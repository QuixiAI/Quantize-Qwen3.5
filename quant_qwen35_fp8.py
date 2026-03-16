#!/usr/bin/env python3
"""
Block-wise FP8 E4M3 quantization for Qwen3.5 finetuned models.
Supports all Qwen3.5 sizes across both MoE and dense families:

  MoE   : 397B-A17B, 122B-A10B, 35B-A3B
  Dense : 27B, 9B, 4B, 2B, 0.8B

Produces a checkpoint compatible with the official Qwen3.5 FP8 format,
loading directly in vLLM and SGLang without engine modifications.

─────────────────────────────────────────────────────────────────────────────
PERFORMANCE MODEL
─────────────────────────────────────────────────────────────────────────────
All calibration modes are fully vectorized over blocks.  There is no Python
loop over the (n_out × n_in) block grid.  The MSE grid search iterates over
alpha values (default: 50), each of which runs a single fused kernel over
all blocks of a weight tensor simultaneously.

  CPU-only (calib_mode=max/rms)      ~fastest
  CUDA    (calib_mode=max/rms)       ~10-20× faster than CPU
  CUDA    (calib_mode=mse, grid=50)  ~recommended; best weight-only quality

CUDA is used automatically when available.  Pass --device cpu to force CPU.

─────────────────────────────────────────────────────────────────────────────
CALIBRATION MODES  (--calib_mode)
─────────────────────────────────────────────────────────────────────────────
  max          Per-block absolute maximum.  Fastest.  Outlier-sensitive.
               scale = max(|block|) / FP8_MAX
               One outlier weight can inflate the scale, compressing all
               other values toward zero.

  rms          RMS-based dynamic range estimate.  Same speed as max but
               much more robust to weight outliers.
               scale = rms(block) * k / FP8_MAX
               where k (--rms_factor, default 3.0) approximates the ratio
               of the block's peak to its RMS.  When outliers push
               max >> k*rms, they are clipped to ±FP8_MAX; the bulk of the
               distribution is preserved at full precision.
               This is the right default if you want a fast export with
               better-than-max quality.  Note: outlier values will saturate
               to ±448 rather than being represented exactly.

  percentile   Per-block percentile clipping.  Handles weight outliers.
               Fully vectorized via torch.quantile over the block dim.
               Tune with --calib_percentile (default: 99.9).

  mse          MSE-optimal scale via vectorized grid search.  Best weight-
               only quality.  Tune with --mse_grid_size (default: 50).

─────────────────────────────────────────────────────────────────────────────
MODE SELECTION GUIDE
─────────────────────────────────────────────────────────────────────────────
  Need fastest possible export, quality secondary  →  max
  Fast export, outlier-robust, no tuning           →  rms                ← default
  Fast + activation-aware (best for 397B MoE)      →  rms + activation_stats
  Good quality, no activation stats                →  percentile
  Best weight-only quality                         →  mse
  Best overall quality (requires calib forward)    →  mse + activation_stats

─────────────────────────────────────────────────────────────────────────────
ACTIVATION-AWARE CALIBRATION  (--activation_stats PATH)
─────────────────────────────────────────────────────────────────────────────
Works with any calib_mode.  Provide a JSON file of per-layer per-channel
input-activation statistics collected from a small calibration forward pass.
When present, the MSE objective is weighted by per-input-channel activation
magnitude, prioritising precision where the matmul output matters most.

Generate the stats file with the companion script collect_activation_stats.py.

Stats file format:
    {
      "model.layers.0.mlp.experts.0.down_proj.weight": {
        "input_channel_max": [0.34, 0.12, ...]   // length == in_features
      },
      ...
    }

─────────────────────────────────────────────────────────────────────────────
VERIFICATION — THREE-TIER HEALTH BANDS
─────────────────────────────────────────────────────────────────────────────
  zone   cosine sim         mean abs err     max abs err
  GOOD   ≥ 0.9997           ≤ 0.0008         ≤ 0.01
  WARN   0.9995 – 0.9997    0.0008 – 0.001   0.01 – 0.02
  FAIL   < 0.9995           > 0.001          > 0.02

─────────────────────────────────────────────────────────────────────────────
RECOMMENDED USAGE
─────────────────────────────────────────────────────────────────────────────
  # Recommended default — fast, outlier-robust
  python quantize_fp8.py \\
      --input_dir  /models/bf16_finetune \\
      --output_dir /models/fp8_finetune

  # Best quality (activation-aware MSE)
  python quantize_fp8.py \\
      --input_dir        /models/bf16_finetune \\
      --output_dir       /models/fp8_finetune  \\
      --calib_mode       mse                   \\
      --activation_stats calib_stats.json

  # Maximum speed (quick sanity check only)
  python quantize_fp8.py \\
      --input_dir  /models/bf16_finetune \\
      --output_dir /models/fp8_finetune  \\
      --calib_mode max --skip_verify
"""

import json
import math
import argparse
import shutil
import logging
import dataclasses
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm

# ── Logging ────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

BLOCK_SIZE  = 128
FP8_DTYPE   = torch.float8_e4m3fn
FP8_MAX     = torch.finfo(FP8_DTYPE).max   # 448.0
SCALE_DTYPE = torch.bfloat16

# ── Three-tier calibration health bands ────────────────────────────────────────
#
# Each metric has two thresholds that define three zones:
#
#   GOOD  — typical healthy range for a well-calibrated BF16 finetune
#   WARN  — outside typical but within hard limit; investigate or re-calibrate
#   FAIL  — hard failure; do not deploy without fixing
#
#                    GOOD          WARN              FAIL
#   cosine sim  :  ≥ 0.9997    [0.9995, 0.9997)    < 0.9995
#   mean abs err:  ≤ 0.0008    (0.0008, 0.001]      > 0.001
#   max abs err :  ≤ 0.01      (0.01,   0.02]        > 0.02
#
# The WARN band is deliberately narrow — it exists to catch cases that look
# like a pass on the hard threshold but are actually degraded.

@dataclasses.dataclass(frozen=True)
class MetricBands:
    name:        str
    good:        float   # value is GOOD  if within good bound
    warn:        float   # value is WARN  if within warn bound (else FAIL)
    higher_is_better: bool

    def evaluate(self, value: float) -> str:
        """Return 'GOOD', 'WARN', or 'FAIL'."""
        if self.higher_is_better:
            if value >= self.good: return "GOOD"
            if value >= self.warn: return "WARN"
            return "FAIL"
        else:
            if value <= self.good: return "GOOD"
            if value <= self.warn: return "WARN"
            return "FAIL"

    def typical_range(self) -> str:
        if self.higher_is_better:
            return f"{self.good}–0.99995"
        else:
            return f"{self.good}–{self.warn}"


COSINE_BANDS  = MetricBands("cosine_sim",    good=0.9997,  warn=0.9995, higher_is_better=True)
MAE_BANDS     = MetricBands("mean_abs_err",  good=0.0008,  warn=0.001,  higher_is_better=False)
MAX_ERR_BANDS = MetricBands("max_abs_err",   good=0.01,    warn=0.02,   higher_is_better=False)

# Keep these as the hard-failure values for backward-compat with CLI defaults
TARGET_COSINE_SIM   = COSINE_BANDS.warn
TARGET_MEAN_ABS_ERR = MAE_BANDS.warn
TARGET_MAX_ABS_ERR  = MAX_ERR_BANDS.warn

# ── Tensor eligibility ─────────────────────────────────────────────────────────

SKIP_PATTERNS = [
    # ── Embeddings & output ────────────────────────────────────────────────────
    "embed_tokens",        # token embedding lookup table
    "lm_head",             # output projection / tied embedding

    # ── Normalization ──────────────────────────────────────────────────────────
    "norm",                # all RMSNorm / LayerNorm weights (input_layernorm,
    "layernorm",           # post_attention_layernorm, linear_attn.norm, ...)

    # ── Gated DeltaNet (linear attention) SSM tensors ─────────────────────────
    "A_log",               # SSM state parameter (kept F32 in all checkpoints)
    "dt_bias",             # SSM delta-time bias
    "conv1d",              # SSM local convolution

    # ── Low-rank / small linear_attn projections ───────────────────────────────
    "in_proj_a",           # [heads, hidden] — e.g. [64, 4096]; too small
    "in_proj_b",           # [heads, hidden] — always < BLOCK_SIZE in first dim

    # ── MoE router ────────────────────────────────────────────────────────────
    "shared_expert_gate",  # router gate  [1, hidden] — always 1 row
    "mlp.gate.",           # expert router [n_experts, hidden] e.g. [512, 4096]
                           # trailing dot ensures we match mlp.gate.weight but
                           # NOT mlp.gate_proj.weight (the dense MLP projection)

    # ── Multi-Token Prediction fusion ─────────────────────────────────────────
    "mtp.fc",              # MTP hidden-state fusion [hidden, 2*hidden], BF16
                           # in every official FP8 checkpoint across all sizes

    # ── Vision encoder ────────────────────────────────────────────────────────
    "visual",              # entire model.visual.* subtree — ViT blocks, merger,
                           # patch_embed, pos_embed; never FP8 in any checkpoint

    # ── Scale tensors themselves ───────────────────────────────────────────────
    "weight_scale_inv",    # the inverse-scale tensors we write

    # ── Bias terms ────────────────────────────────────────────────────────────
    "bias",                # all bias terms (rare in these models but safe)
]


def should_quantize(name: str, tensor: torch.Tensor) -> bool:
    if tensor.ndim != 2:
        return False
    if tensor.shape[0] < BLOCK_SIZE or tensor.shape[1] < BLOCK_SIZE:
        return False
    if not name.endswith(".weight"):
        return False
    for pat in SKIP_PATTERNS:
        if pat in name:
            return False
    return True


def scale_inv_name(weight_name: str) -> str:
    return weight_name.replace(".weight", ".weight_scale_inv")


def expected_scale_shape(weight_shape: Tuple[int, int]) -> Tuple[int, int]:
    out, inp = weight_shape
    return (math.ceil(out / BLOCK_SIZE), math.ceil(inp / BLOCK_SIZE))


# ── Vectorized block-wise FP8 quantization ─────────────────────────────────────

def quantize_fp8_blockwise(
    weight:      torch.Tensor,
    calib_mode:  str                    = "rms",
    percentile:  float                  = 99.9,
    mse_grid:    int                    = 50,
    rms_factor:  float                  = 3.0,
    act_channel: Optional[torch.Tensor] = None,   # [in_features]
    device:      Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fully vectorized block-wise FP8 E4M3 quantization with 128×128 tiles.

    There is NO Python loop over the block grid.  All calibration strategies
    operate on the full [n_out, n_in, 128, 128] block tensor at once.  The
    MSE grid search iterates only over alpha values (mse_grid steps), with
    each step being a single fused tensor operation over all blocks.

    Args:
        weight:      [out, in] any float dtype.
        calib_mode:  "rms" | "max" | "percentile" | "mse".
        percentile:  clip percentile for "percentile" mode.
        mse_grid:    number of alpha candidates for "mse" mode.
        rms_factor:  multiplier k for "rms" mode: scale = rms(block)*k / FP8_MAX.
                     Default 3.0 approximates the peak-to-RMS ratio of a normal
                     distribution.  Increase (e.g. 4.0) to reduce saturation of
                     heavy-tailed blocks; decrease (e.g. 2.5) to clip more
                     aggressively and improve precision for typical values.
        act_channel: per-input-channel activation magnitudes [in_features].
                     Activates activation-weighted MSE regardless of calib_mode.
        device:      compute device. Auto-detected if None.

    Returns:
        weight_fp8:       [out, in]       float8_e4m3fn  (on CPU)
        weight_scale_inv: [B_out, B_in]   bfloat16       (on CPU)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    orig_out, orig_in = weight.shape

    # Move to compute device in float32
    w = weight.float().to(device)

    # ── Pad to exact multiples of BLOCK_SIZE ──────────────────────────────────
    pad_out = (BLOCK_SIZE - orig_out % BLOCK_SIZE) % BLOCK_SIZE
    pad_in  = (BLOCK_SIZE - orig_in  % BLOCK_SIZE) % BLOCK_SIZE
    if pad_out > 0 or pad_in > 0:
        w = F.pad(w, (0, pad_in, 0, pad_out))

    pout, pin = w.shape
    n_out = pout // BLOCK_SIZE
    n_in  = pin  // BLOCK_SIZE

    # ── Reshape to expose all blocks: [n_out, n_in, 128, 128] ─────────────────
    # w layout: [n_out*128, n_in*128]
    #   → [n_out, 128, n_in, 128]   (natural reshape)
    #   → [n_out, n_in, 128, 128]   (permute for cleaner indexing)
    blocks = (
        w.reshape(n_out, BLOCK_SIZE, n_in, BLOCK_SIZE)
        .permute(0, 2, 1, 3)
        .contiguous()
    )   # [n_out, n_in, 128, 128]

    # ── Prepare activation weights if provided ────────────────────────────────
    act_w: Optional[torch.Tensor] = None
    if act_channel is not None:
        a = act_channel.float().to(device)
        if pad_in > 0:
            a = F.pad(a, (0, pad_in), value=0.0)
        # Normalise and reshape to [1, n_in, 1, 128] for broadcasting
        a = a / a.max().clamp(min=1e-12)
        act_w = a.reshape(n_in, BLOCK_SIZE)[None, :, None, :]  # [1, n_in, 1, 128]

    # ── Compute per-block optimal scales (all strategies vectorized) ──────────
    scale = _compute_scales(
        blocks=blocks,
        calib_mode=calib_mode,
        percentile=percentile,
        mse_grid=mse_grid,
        rms_factor=rms_factor,
        act_w=act_w,
        device=device,
    )   # [n_out, n_in]  float32

    scale_inv = (1.0 / scale.clamp(min=1e-12)).to(SCALE_DTYPE).cpu()

    # ── Final quantization with calibrated scales ─────────────────────────────
    # Expand scale: [n_out, n_in] → [n_out, n_in, 1, 1]
    scale_exp = scale[:, :, None, None]
    weight_fp8 = (
        (blocks / scale_exp)
        .clamp(-FP8_MAX, FP8_MAX)
        .to(FP8_DTYPE)                            # [n_out, n_in, 128, 128]
        .permute(0, 2, 1, 3)                      # [n_out, 128, n_in, 128]
        .contiguous()
        .reshape(pout, pin)
        [:orig_out, :orig_in]
        .contiguous()
        .cpu()
    )

    return weight_fp8, scale_inv


def _compute_scales(
    blocks:      torch.Tensor,            # [n_out, n_in, 128, 128]
    calib_mode:  str,
    percentile:  float,
    mse_grid:    int,
    rms_factor:  float,
    act_w:       Optional[torch.Tensor],  # [1, n_in, 1, 128] or None
    device:      torch.device,
) -> torch.Tensor:
    """
    Return per-block scales of shape [n_out, n_in] on `device`.

    All strategies are fully vectorized over the block grid.  The MSE search
    loops over alpha values only (mse_grid iterations), each of which is a
    single fused tensor op over all blocks simultaneously.

    act_w, when provided, is [1, n_in, 1, 128] of normalised per-input-channel
    activation magnitudes.  Its effect depends on the mode:

      rms  + act_w  →  activation-weighted RMS:
                        scale = sqrt(mean_j(a_j · mean_i(W_ij²))) * k / FP8_MAX
                        Channels with large activations dominate the RMS estimate,
                        so the scale tightens around the channels that matter most.
                        Fast (zero grid search) and activation-aware.  Recommended
                        for large MoE models when export speed matters.

      mse  + act_w  →  activation-weighted MSE grid search:
                        Minimises sum_j a_j · sum_i (W_q_ij - W_ij)².
                        Highest quality, but O(mse_grid) passes over the block
                        tensor.

      rms  – act_w  →  plain RMS (fast, outlier-robust baseline).
      mse  – act_w  →  plain MSE grid search (best weight-only quality).
      max  / percentile ignore act_w (activation weighting requires a
                        differentiable loss; these modes don't have one).

    Modes
    -----
    rms        : scale = rms(block, act_w) * rms_factor / FP8_MAX
    max        : scale = max(|block|) / FP8_MAX
    percentile : scale = quantile(|block|, p) / FP8_MAX
    mse        : grid search over alpha ∈ [0.5, 1.0]
    """
    n_out, n_in, _, _ = blocks.shape

    # ── rms ───────────────────────────────────────────────────────────────────
    if calib_mode == "rms":
        sq = blocks ** 2   # [n_out, n_in, 128, 128]
        if act_w is not None:
            # act_w: [1, n_in, 1, 128] — weight each input channel by its
            # activation magnitude before taking the mean.
            # Result: scale reflects the RMS of the channels that matter most.
            block_rms = torch.sqrt((sq * act_w).mean(dim=(-2, -1))).clamp(min=1e-12)
        else:
            block_rms = torch.sqrt(sq.mean(dim=(-2, -1))).clamp(min=1e-12)
        return (block_rms * rms_factor) / FP8_MAX

    # ── max ───────────────────────────────────────────────────────────────────
    if calib_mode == "max":
        return blocks.abs().amax(dim=(-2, -1)).clamp(min=1e-12) / FP8_MAX

    # ── percentile ────────────────────────────────────────────────────────────
    if calib_mode == "percentile":
        flat = blocks.abs().reshape(n_out, n_in, -1)
        p = torch.quantile(flat, percentile / 100.0, dim=-1)   # [n_out, n_in]
        return p.clamp(min=1e-12) / FP8_MAX

    # ── mse (with optional activation weighting) ──────────────────────────────
    # block_max: [n_out, n_in]
    block_max = blocks.abs().amax(dim=(-2, -1)).clamp(min=1e-12)

    best_scale = block_max / FP8_MAX                            # [n_out, n_in]
    best_loss  = torch.full((n_out, n_in), float("inf"), device=device)

    # Loop over alpha candidates only — O(mse_grid) iterations,
    # each fully vectorized over the entire [n_out, n_in, 128, 128] block tensor.
    for alpha in torch.linspace(0.5, 1.0, mse_grid, device=device):
        s     = (block_max * alpha) / FP8_MAX                  # [n_out, n_in]
        s_exp = s[:, :, None, None]                            # [n_out, n_in, 1, 1]

        quant = (
            (blocks / s_exp)
            .clamp(-FP8_MAX, FP8_MAX)
            .to(FP8_DTYPE)
            .float()
        )   # [n_out, n_in, 128, 128]

        err2 = (quant * s_exp - blocks).pow(2)                 # [n_out, n_in, 128, 128]

        if act_w is not None:
            # act_w broadcasts over [n_out, n_in, 128(out), 128(in)]
            loss = (err2 * act_w).mean(dim=(-2, -1))           # [n_out, n_in]
        else:
            loss = err2.mean(dim=(-2, -1))                     # [n_out, n_in]

        improved   = loss < best_loss
        best_loss  = torch.where(improved, loss,  best_loss)
        best_scale = torch.where(improved, s,     best_scale)

    return best_scale


# ── Dequantization (verification only) ────────────────────────────────────────

def dequantize(
    fp8_weight: torch.Tensor,
    scale_inv:  torch.Tensor,
) -> torch.Tensor:
    """Reconstruct approximate BF16 weight from FP8 + inverse-scale tensors."""
    orig_out, orig_in = fp8_weight.shape
    n_out, n_in       = scale_inv.shape

    w = fp8_weight.float()
    pad_out = n_out * BLOCK_SIZE - orig_out
    pad_in  = n_in  * BLOCK_SIZE - orig_in
    if pad_out > 0 or pad_in > 0:
        w = F.pad(w, (0, pad_in, 0, pad_out))

    # Expand scale_inv [n_out, n_in] → [pout, pin]
    scale_exp = (
        scale_inv.float()
        .unsqueeze(1)
        .unsqueeze(3)
        .expand(n_out, BLOCK_SIZE, n_in, BLOCK_SIZE)
        .reshape(n_out * BLOCK_SIZE, n_in * BLOCK_SIZE)
    )
    return (w * scale_exp)[:orig_out, :orig_in].to(torch.bfloat16)


# ── Verification data structures ───────────────────────────────────────────────

@dataclasses.dataclass
class TensorVerifyResult:
    name:              str
    dtype_ok:          bool
    scale_exists:      bool
    scale_dtype_ok:    bool
    scale_shape_ok:    bool
    cosine_sim:        float
    mean_abs_err:      float
    max_abs_err:       float
    # Per-metric health bands
    cosine_band:       str   # "GOOD" | "WARN" | "FAIL"
    mae_band:          str
    max_err_band:      str
    structural_errors: List[str]   # dtype / shape / missing-scale failures

    @property
    def overall_band(self) -> str:
        """Worst of the three metric bands, plus FAIL for any structural error."""
        if self.structural_errors:
            return "FAIL"
        order = {"GOOD": 0, "WARN": 1, "FAIL": 2}
        worst = max(
            [self.cosine_band, self.mae_band, self.max_err_band],
            key=lambda b: order[b],
        )
        return worst

    @property
    def passed(self) -> bool:
        return self.overall_band != "FAIL"


@dataclasses.dataclass
class ShardVerifyResult:
    shard_name:     str
    n_quantized:    int
    n_passthrough:  int
    n_verified:     int
    tensor_results: List[TensorVerifyResult]

    def _count_band(self, band: str) -> int:
        return sum(1 for r in self.tensor_results if r.overall_band == band)

    @property
    def n_good(self)  -> int: return self._count_band("GOOD")
    @property
    def n_warn(self)  -> int: return self._count_band("WARN")
    @property
    def n_fail(self)  -> int: return self._count_band("FAIL")
    @property
    def n_passed(self) -> int: return self.n_good + self.n_warn

    @property
    def all_passed(self) -> bool:
        return self.n_fail == 0

    @property
    def shard_band(self) -> str:
        if self.n_fail > 0:   return "FAIL"
        if self.n_warn > 0:   return "WARN"
        return "GOOD"

    def mean_cosine_sim(self) -> float:
        v = [r.cosine_sim for r in self.tensor_results]
        return sum(v) / len(v) if v else float("nan")

    def mean_mae(self) -> float:
        v = [r.mean_abs_err for r in self.tensor_results]
        return sum(v) / len(v) if v else float("nan")


# ── Verification logic ─────────────────────────────────────────────────────────

def _verify_one_pair(
    name:       str,
    original:   torch.Tensor,
    fp8_weight: torch.Tensor,
    scale_inv:  torch.Tensor,
) -> TensorVerifyResult:
    structural_errors = []

    # 1. dtype checks
    if fp8_weight.dtype != FP8_DTYPE:
        structural_errors.append(f"weight dtype={fp8_weight.dtype}, expected {FP8_DTYPE}")
    if scale_inv.dtype != SCALE_DTYPE:
        structural_errors.append(f"scale_inv dtype={scale_inv.dtype}, expected {SCALE_DTYPE}")

    # 2. scale shape
    exp = expected_scale_shape(fp8_weight.shape)
    if tuple(scale_inv.shape) != exp:
        structural_errors.append(
            f"scale_inv shape={tuple(scale_inv.shape)}, expected {exp}"
        )

    # 3. round-trip metrics
    reconstructed = dequantize(fp8_weight, scale_inv).float()
    orig_f        = original.float()

    abs_err = (reconstructed - orig_f).abs()
    mean_ae = abs_err.mean().item()
    max_ae  = abs_err.max().item()
    cosine  = F.cosine_similarity(
        orig_f.flatten().unsqueeze(0),
        reconstructed.flatten().unsqueeze(0),
    ).item()

    return TensorVerifyResult(
        name=name,
        dtype_ok=(fp8_weight.dtype == FP8_DTYPE),
        scale_exists=True,
        scale_dtype_ok=(scale_inv.dtype == SCALE_DTYPE),
        scale_shape_ok=(tuple(scale_inv.shape) == exp),
        cosine_sim=cosine,
        mean_abs_err=mean_ae,
        max_abs_err=max_ae,
        cosine_band=COSINE_BANDS.evaluate(cosine),
        mae_band=MAE_BANDS.evaluate(mean_ae),
        max_err_band=MAX_ERR_BANDS.evaluate(max_ae),
        structural_errors=structural_errors,
    )


def _dtype_from_slice(slice_obj) -> torch.dtype:
    try:
        return slice_obj[0:1].dtype
    except Exception:
        return torch.float32


def verify_shard(
    input_path:     Path,
    output_path:    Path,
    shard_name:     str,
    verify_samples: int,
) -> ShardVerifyResult:
    """Re-open both shards and verify a random sample of FP8 pairs."""
    with safe_open(str(output_path), framework="pt", device="cpu") as f_out:
        out_keys  = set(f_out.keys())
        out_metas = {k: f_out.get_slice(k) for k in out_keys}

    fp8_weights = [
        k for k in out_keys
        if not k.endswith("weight_scale_inv")
        and _dtype_from_slice(out_metas[k]) == FP8_DTYPE
    ]
    passthrough = [
        k for k in out_keys
        if k not in fp8_weights and not k.endswith("weight_scale_inv")
    ]

    sample_size = min(verify_samples, len(fp8_weights))
    sample_keys = [
        fp8_weights[i]
        for i in torch.randperm(len(fp8_weights))[:sample_size].tolist()
    ]

    originals: Dict[str, torch.Tensor] = {}
    with safe_open(str(input_path), framework="pt", device="cpu") as f_in:
        in_keys = set(f_in.keys())
        for k in sample_keys:
            if k in in_keys:
                originals[k] = f_in.get_tensor(k)

    tensor_results: List[TensorVerifyResult] = []

    with safe_open(str(output_path), framework="pt", device="cpu") as f_out:
        for name in sample_keys:
            sname = scale_inv_name(name)
            if sname not in out_keys:
                tensor_results.append(TensorVerifyResult(
                    name=name,
                    dtype_ok=True, scale_exists=False,
                    scale_dtype_ok=False, scale_shape_ok=False,
                    cosine_sim=0.0, mean_abs_err=float("inf"), max_abs_err=float("inf"),
                    cosine_band="FAIL", mae_band="FAIL", max_err_band="FAIL",
                    structural_errors=["weight_scale_inv tensor missing from shard"],
                ))
                continue

            orig = originals.get(name)
            if orig is None:
                log.warning("  Cannot load original for %s — skipping.", name)
                continue

            tensor_results.append(_verify_one_pair(
                name, orig,
                f_out.get_tensor(name),
                f_out.get_tensor(sname),
            ))

    return ShardVerifyResult(
        shard_name=shard_name,
        n_quantized=len(fp8_weights),
        n_passthrough=len(passthrough),
        n_verified=len(tensor_results),
        tensor_results=tensor_results,
    )


# ── Reporting ──────────────────────────────────────────────────────────────────

# ANSI colours
_C_GREEN  = "\033[92m"
_C_YELLOW = "\033[93m"
_C_RED    = "\033[91m"
_C_RESET  = "\033[0m"
_C_BOLD   = "\033[1m"
_C_DIM    = "\033[2m"

_BAND_COLOUR = {"GOOD": _C_GREEN, "WARN": _C_YELLOW, "FAIL": _C_RED}
_BAND_ICON   = {"GOOD": "✔", "WARN": "⚠", "FAIL": "✘"}


def _fmt_band(band: str, text: str = "") -> str:
    c = _BAND_COLOUR[band]
    icon = _BAND_ICON[band]
    label = text or band
    return f"{c}{icon} {label}{_C_RESET}"


def _fmt_metric(value: float, band: str, fmt: str = ".6f") -> str:
    c = _BAND_COLOUR[band]
    return f"{c}{value:{fmt}}{_C_RESET}"


def _print_shard_report(result: ShardVerifyResult) -> None:
    icon_line = _fmt_band(result.shard_band, result.shard_name)
    log.info(
        "%s  quantized=%d  passthrough=%d  "
        "verified=%d  good=%d warn=%d fail=%d  "
        "cosine=%s  MAE=%s",
        icon_line,
        result.n_quantized, result.n_passthrough,
        result.n_verified,
        result.n_good, result.n_warn, result.n_fail,
        _fmt_metric(result.mean_cosine_sim(), COSINE_BANDS.evaluate(result.mean_cosine_sim())),
        _fmt_metric(result.mean_mae(),        MAE_BANDS.evaluate(result.mean_mae())),
    )
    for r in result.tensor_results:
        band = r.overall_band
        if band == "GOOD":
            log.debug(
                "  %s  cosine=%s  MAE=%s  MaxAE=%s",
                _fmt_band("GOOD", r.name),
                _fmt_metric(r.cosine_sim,    r.cosine_band),
                _fmt_metric(r.mean_abs_err,  r.mae_band),
                _fmt_metric(r.max_abs_err,   r.max_err_band),
            )
        else:
            log.warning(
                "  %s\n"
                "       cosine  %s  (%s)  typical ≥ %s\n"
                "       MAE     %s  (%s)  typical ≤ %s\n"
                "       MaxAE   %s  (%s)  typical ≤ %s%s",
                _fmt_band(band, r.name),
                _fmt_metric(r.cosine_sim,   r.cosine_band),   r.cosine_band,   COSINE_BANDS.good,
                _fmt_metric(r.mean_abs_err, r.mae_band),      r.mae_band,      MAE_BANDS.good,
                _fmt_metric(r.max_abs_err,  r.max_err_band),  r.max_err_band,  MAX_ERR_BANDS.good,
                (f"\n       structural: {'; '.join(r.structural_errors)}"
                 if r.structural_errors else ""),
            )


# Calibration-specific remediation advice, keyed by (metric, current_mode)
_REMEDIATION: Dict[Tuple[str, str], str] = {
    # cosine degraded
    ("cosine", "rms"):         "Try increasing --rms_factor (e.g. 4.0) to reduce "
                               "saturation of heavy-tailed blocks, or switch to "
                               "--calib_mode mse for explicit error minimisation.",
    ("cosine", "max"):         "Switch to --calib_mode rms (same speed, outlier-robust) "
                               "or --calib_mode mse for best quality.",
    ("cosine", "percentile"):  "Try lowering --calib_percentile to 99.5 or 99.0 to clip "
                               "outliers more aggressively.",
    ("cosine", "mse"):         "Add activation stats (collect_activation_stats.py) to "
                               "switch to activation-weighted MSE, or increase "
                               "--mse_grid_size to 100.",
    # MAE degraded
    ("mae", "rms"):            "Switch to --calib_mode mse — MSE search finds the "
                               "optimal clipping factor directly. Or try adjusting "
                               "--rms_factor (lower = more clipping = lower MAE for "
                               "typical values, but more saturation of outliers).",
    ("mae", "max"):            "Switch to --calib_mode rms or mse — max scaling lets "
                               "outliers inflate the scale, raising MAE for all other values.",
    ("mae", "percentile"):     "Try --calib_mode mse for explicit error minimisation, "
                               "or lower --calib_percentile (e.g. 99.5).",
    ("mae", "mse"):            "Increase --mse_grid_size (try 100) or supply "
                               "--activation_stats to weight error by actual activations.",
    # max abs error degraded
    ("maxe", "rms"):           "Outlier blocks are saturating. Lower --rms_factor "
                               "(e.g. 2.5) to clip more aggressively, or switch to "
                               "--calib_mode percentile --calib_percentile 99.9.",
    ("maxe", "max"):           "Outlier blocks are inflating max error. Switch to "
                               "--calib_mode rms (same speed) or "
                               "--calib_mode percentile --calib_percentile 99.9.",
    ("maxe", "percentile"):    "Lower --calib_percentile (99.5 or 99.0) to clip the "
                               "outlier blocks more aggressively.",
    ("maxe", "mse"):           "Max error is dominated by outlier blocks. Try "
                               "--calib_mode percentile --calib_percentile 99.9 "
                               "or combine both: run percentile first, use its scale "
                               "as the upper bound for MSE search.",
}


def _calibration_recommendations(
    all_results: List[ShardVerifyResult],
    calib_mode:  str,
) -> List[str]:
    """
    Inspect aggregate metric bands across all results and return a list of
    actionable recommendation strings.
    """
    all_tensors = [t for r in all_results for t in r.tensor_results]
    if not all_tensors:
        return []

    def agg_band(values, bands_obj):
        mean_val = sum(values) / len(values)
        return bands_obj.evaluate(mean_val), mean_val

    cosines  = [t.cosine_sim   for t in all_tensors]
    maes     = [t.mean_abs_err for t in all_tensors]
    maxerrs  = [t.max_abs_err  for t in all_tensors]

    cos_band,  cos_val  = agg_band(cosines,  COSINE_BANDS)
    mae_band,  mae_val  = agg_band(maes,     MAE_BANDS)
    maxe_band, maxe_val = agg_band(maxerrs,  MAX_ERR_BANDS)

    recs = []

    if cos_band != "GOOD":
        tip = _REMEDIATION.get(("cosine", calib_mode), "")
        recs.append(
            f"Cosine similarity avg={cos_val:.6f} is in the {cos_band} zone "
            f"(typical ≥ {COSINE_BANDS.good}).  {tip}"
        )
    if mae_band != "GOOD":
        tip = _REMEDIATION.get(("mae", calib_mode), "")
        recs.append(
            f"Mean abs error avg={mae_val:.6f} is in the {mae_band} zone "
            f"(typical ≤ {MAE_BANDS.good}).  {tip}"
        )
    if maxe_band != "GOOD":
        tip = _REMEDIATION.get(("maxe", calib_mode), "")
        recs.append(
            f"Max abs error avg={maxe_val:.6f} is in the {maxe_band} zone "
            f"(typical ≤ {MAX_ERR_BANDS.good}).  {tip}"
        )

    # Generic advice when any metric is failing hard
    if any(b == "FAIL" for b in [cos_band, mae_band, maxe_band]):
        recs.append(
            "One or more metrics are in the FAIL zone. Before deploying, "
            "verify the source BF16 model is numerically stable (no NaNs/Infs "
            "in weights), then try the recommended calibration changes above."
        )

    return recs


def _print_final_report(
    all_results: List[ShardVerifyResult],
    calib_desc:  str,
    calib_mode:  str,
) -> None:
    total_q  = sum(r.n_quantized   for r in all_results)
    total_pt = sum(r.n_passthrough for r in all_results)
    total_v  = sum(r.n_verified    for r in all_results)
    total_g  = sum(r.n_good        for r in all_results)
    total_w  = sum(r.n_warn        for r in all_results)
    total_f  = sum(r.n_fail        for r in all_results)

    all_cos  = [t.cosine_sim   for r in all_results for t in r.tensor_results]
    all_mae  = [t.mean_abs_err for r in all_results for t in r.tensor_results]
    all_maxe = [t.max_abs_err  for r in all_results for t in r.tensor_results]

    def safe_mean(lst): return sum(lst) / len(lst) if lst else float("nan")
    def safe_max(lst):  return max(lst)             if lst else float("nan")

    mean_cos  = safe_mean(all_cos)
    mean_mae  = safe_mean(all_mae)
    worst_mae = safe_max(all_mae)
    worst_max = safe_max(all_maxe)

    overall_band = (
        "FAIL" if total_f > 0 else
        "WARN" if total_w > 0 else
        "GOOD"
    )

    bar  = "=" * 66
    bar2 = "-" * 66

    log.info(bar)
    log.info("  %sQUANTIZATION VERIFICATION SUMMARY%s", _C_BOLD, _C_RESET)
    log.info("  Calibration         : %s", calib_desc)
    log.info(bar2)
    log.info("  Shards processed    : %d", len(all_results))
    log.info("  FP8 weight tensors  : %d", total_q)
    log.info("  Passthrough tensors : %d", total_pt)
    log.info("  Spot-checked        : %d  (%s%d GOOD%s  %s%d WARN%s  %s%d FAIL%s)",
             total_v,
             _C_GREEN,  total_g, _C_RESET,
             _C_YELLOW, total_w, _C_RESET,
             _C_RED,    total_f, _C_RESET)
    log.info(bar2)

    # Header row
    log.info(
        "  %-14s  %-10s  %-12s  %-10s  %-10s",
        "metric", "mean value", "typical range", "band", "hard limit",
    )
    log.info("  " + "-" * 62)

    def _metric_row(label, value, bands_obj, hard_limit_str):
        band = bands_obj.evaluate(value)
        c    = _BAND_COLOUR[band]
        icon = _BAND_ICON[band]
        log.info(
            "  %-14s  %-10.6f  %-12s  %s%-10s%s  %s",
            label, value, bands_obj.typical_range(),
            c, f"{icon} {band}", _C_RESET,
            hard_limit_str,
        )

    _metric_row("cosine sim",   mean_cos,  COSINE_BANDS,  f"≥ {COSINE_BANDS.warn}")
    _metric_row("mean abs err", mean_mae,  MAE_BANDS,     f"≤ {MAE_BANDS.warn}")
    _metric_row("worst MAE",    worst_mae, MAE_BANDS,     f"≤ {MAE_BANDS.warn}")
    _metric_row("worst MaxAE",  worst_max, MAX_ERR_BANDS, f"≤ {MAX_ERR_BANDS.warn}")

    log.info(bar2)

    # Verdict
    if overall_band == "GOOD":
        log.info("%s  All metrics within typical range. Model ready for inference.",
                 _fmt_band("GOOD", "PASS"))
    elif overall_band == "WARN":
        log.warning(
            "%s  Metrics are within hard limits but outside the typical range.\n"
            "     Model may deploy, but quality could be slightly degraded.\n"
            "     See calibration recommendations below.",
            _fmt_band("WARN", "WARN"),
        )
    else:
        log.error(
            "%s  One or more metrics exceeded hard limits.\n"
            "     Do NOT deploy without investigating. See recommendations below.",
            _fmt_band("FAIL", "FAIL"),
        )

    # Calibration recommendations
    recs = _calibration_recommendations(all_results, calib_mode)
    if recs:
        log.info(bar2)
        log.info("  %sCALIBRATION RECOMMENDATIONS%s", _C_BOLD, _C_RESET)
        for i, rec in enumerate(recs, 1):
            # Word-wrap at 80 chars with indent
            words, line, lines = rec.split(), "", []
            for word in words:
                if len(line) + len(word) + 1 > 74:
                    lines.append(line)
                    line = word
                else:
                    line = (line + " " + word).lstrip()
            if line:
                lines.append(line)
            log.info("  %d. %s", i, lines[0])
            for extra in lines[1:]:
                log.info("     %s", extra)

    log.info(bar)


# ── Activation stats ───────────────────────────────────────────────────────────

ActivationStats = Dict[str, torch.Tensor]   # weight_name → [in_features]


def load_activation_stats(path: Path) -> ActivationStats:
    log.info("Loading activation stats from %s ...", path)
    with open(path) as f:
        raw = json.load(f)

    stats: ActivationStats = {}
    for name, data in raw.items():
        if "input_channel_max" not in data:
            log.warning("  No 'input_channel_max' for %s — skipping.", name)
            continue
        stats[name] = torch.tensor(data["input_channel_max"], dtype=torch.float32)

    log.info("  Loaded stats for %d layers.", len(stats))
    if not stats:
        log.warning(
            "  Activation stats file empty or malformed. "
            "Falling back to weight-only calibration."
        )
    return stats


# ── Model size detection & calib stats auto-discovery ─────────────────────────

# (hidden_size, num_hidden_layers) → size label used in calib_stats_*.json
_SIZE_LOOKUP: Dict[Tuple[int, int], str] = {
    (4096, 60): "397b",
    (3072, 48): "122b",
    (2048, 40): "35b",
    (5120, 64): "27b",
    (4096, 32): "9b",
    (2560, 32): "4b",
    (2048, 24): "2b",
    (1024, 24): "0.8b",
}


def _detect_model_size(input_dir: Path) -> Optional[str]:
    """
    Read config.json from input_dir and return the Qwen3.5 size label
    (e.g. '397b', '9b', '0.8b'), or None if unrecognised.

    Qwen3.5 vision-language models wrap the LM config under a 'text_config'
    key.  Falls back to the top-level keys for plain language models.
    """
    config_path = input_dir / "config.json"
    if not config_path.exists():
        return None

    try:
        with open(config_path) as f:
            cfg = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None

    # VL models nest LM config under text_config
    lm_cfg = cfg.get("text_config", cfg)

    hidden  = lm_cfg.get("hidden_size")
    layers  = lm_cfg.get("num_hidden_layers")

    if hidden is None or layers is None:
        return None

    return _SIZE_LOOKUP.get((hidden, layers))


def _find_calib_stats(input_dir: Path) -> Optional[Path]:
    """
    Auto-discover a calib_stats_*.json file that matches the model in
    input_dir.  Searches (in order):
      1. The current working directory
      2. The directory containing this script

    Returns the Path if found, None otherwise.
    """
    size_label = _detect_model_size(input_dir)
    if size_label is None:
        return None

    filename = f"calib_stats_{size_label}.json"
    search_dirs = [Path.cwd(), Path(__file__).parent.resolve()]

    for directory in search_dirs:
        candidate = directory / filename
        if candidate.exists():
            return candidate

    return None


# ── Shard processing ───────────────────────────────────────────────────────────

def process_shard(
    input_path:  Path,
    output_path: Path,
    index:       Optional[Dict],
    shard_name:  str,
    calib_mode:  str,
    percentile:  float,
    mse_grid:    int,
    rms_factor:  float,
    act_stats:   ActivationStats,
    device:      torch.device,
) -> List[str]:
    """Quantize one shard. Returns list of FP8-quantized weight names."""
    tensors_out:    Dict[str, torch.Tensor] = {}
    quantized_keys: List[str]               = []
    act_hits        = 0

    with safe_open(str(input_path), framework="pt", device="cpu") as f:
        keys = list(f.keys())
        for name in tqdm(keys, desc=f"  {shard_name}", leave=False):
            tensor = f.get_tensor(name)

            if should_quantize(name, tensor):
                act_channel = act_stats.get(name)
                if act_channel is not None:
                    if act_channel.shape[0] != tensor.shape[1]:
                        log.warning(
                            "  Act stat shape mismatch for %s: "
                            "expected [%d] got [%d]. Using weight-only.",
                            name, tensor.shape[1], act_channel.shape[0],
                        )
                        act_channel = None
                    else:
                        act_hits += 1

                fp8_w, scale_i = quantize_fp8_blockwise(
                    tensor,
                    calib_mode=calib_mode,
                    percentile=percentile,
                    mse_grid=mse_grid,
                    rms_factor=rms_factor,
                    act_channel=act_channel,
                    device=device,
                )
                sname = scale_inv_name(name)
                tensors_out[name]  = fp8_w
                tensors_out[sname] = scale_i
                quantized_keys.append(name)

                if index is not None and "weight_map" in index:
                    index["weight_map"][sname] = shard_name
            else:
                tensors_out[name] = tensor

    save_file(tensors_out, str(output_path))

    if act_stats:
        log.info(
            "    Activation stats applied to %d / %d quantized tensors.",
            act_hits, len(quantized_keys),
        )
    return quantized_keys


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )

    io = parser.add_argument_group("I/O")
    io.add_argument("--input_dir",  required=True, type=Path)
    io.add_argument("--output_dir", required=True, type=Path)

    cal = parser.add_argument_group("Calibration")
    cal.add_argument("--calib_mode", default="rms",
                     choices=["rms", "max", "percentile", "mse"],
                     help="Per-block scale calibration strategy. (default: rms)")
    cal.add_argument("--rms_factor", type=float, default=3.0,
                     help="Peak-to-RMS multiplier k for 'rms' mode: "
                          "scale = rms(block)*k / FP8_MAX. "
                          "3.0 suits near-Gaussian weights; increase to 4.0 for "
                          "heavy-tailed blocks, decrease to 2.5 for more clipping. "
                          "(default: 3.0)")
    cal.add_argument("--calib_percentile", type=float, default=99.9,
                     help="Clip percentile for 'percentile' mode. (default: 99.9)")
    cal.add_argument("--mse_grid_size", type=int, default=50,
                     help="Alpha candidates for MSE grid search. (default: 50)")
    cal.add_argument("--activation_stats", type=Path, default=None,
                     help="JSON of per-layer activation statistics from "
                          "collect_activation_stats.py.")
    cal.add_argument("--device", type=str, default=None,
                     help="Compute device: 'cuda', 'cuda:1', 'cpu', etc. "
                          "Auto-detected if omitted.")

    ver = parser.add_argument_group("Verification")
    ver.add_argument("--verify_samples", type=int, default=5,
                     help="FP8 tensors to spot-check per shard. (default: 5)")
    ver.add_argument("--skip_verify", action="store_true",
                     help="Skip the verification pass.")

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Resolve compute device ─────────────────────────────────────────────────
    if args.device is not None:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    log.info("Compute device : %s", device)

    # ── Activation stats ───────────────────────────────────────────────────────
    act_stats: ActivationStats = {}

    if args.activation_stats is not None:
        # Explicit path provided — use it directly.
        act_stats = load_activation_stats(args.activation_stats)
    else:
        # No explicit path — try to auto-discover calib_stats_{size}.json.
        auto_path = _find_calib_stats(args.input_dir)
        if auto_path is not None:
            log.info("Auto-detected activation stats: %s", auto_path)
            act_stats = load_activation_stats(auto_path)
        else:
            size_label = _detect_model_size(args.input_dir)
            if size_label is not None:
                log.info(
                    "No calib_stats_%s.json found in cwd or script dir — "
                    "running without activation stats. "
                    "Run collect_activation_stats.py to generate it.",
                    size_label,
                )
            else:
                log.info(
                    "Model size unrecognised (config.json missing or unknown) — "
                    "running without activation stats."
                )

    calib_desc = args.calib_mode
    if args.calib_mode == "rms":
        calib_desc += f"(k={args.rms_factor})"
    if act_stats:
        calib_desc += " + activation-weighted"
    log.info("Calibration    : %s  (grid=%d)", calib_desc, args.mse_grid_size)

    # ── Locate shards ──────────────────────────────────────────────────────────
    index_path = args.input_dir / "model.safetensors.index.json"
    if not index_path.exists():
        shards = ["model.safetensors"]
        index  = None
        log.info("Single-shard model detected.")
    else:
        with open(index_path) as f:
            index = json.load(f)
        shards = sorted(set(index["weight_map"].values()))
        log.info("Found %d shards in index.", len(shards))

    # ── Process shards ─────────────────────────────────────────────────────────
    all_verify_results: List[ShardVerifyResult] = []

    for shard_name in shards:
        input_path  = args.input_dir  / shard_name
        output_path = args.output_dir / shard_name

        log.info("Quantizing %s ...", shard_name)
        quantized_keys = process_shard(
            input_path, output_path, index, shard_name,
            calib_mode=args.calib_mode,
            percentile=args.calib_percentile,
            mse_grid=args.mse_grid_size,
            rms_factor=args.rms_factor,
            act_stats=act_stats,
            device=device,
        )
        log.info("  -> %d tensors quantized to FP8.", len(quantized_keys))

        if not args.skip_verify:
            log.info("  Verifying (sampling %d tensors) ...", args.verify_samples)
            result = verify_shard(
                input_path, output_path, shard_name,
                verify_samples=args.verify_samples,
            )
            _print_shard_report(result)
            all_verify_results.append(result)

    # ── Write updated index ────────────────────────────────────────────────────
    if index is not None:
        out_index = args.output_dir / "model.safetensors.index.json"
        with open(out_index, "w") as f:
            json.dump(index, f, indent=2)
        log.info("Wrote updated model.safetensors.index.json.")

    # ── Copy non-weight files ──────────────────────────────────────────────────
    skip_names = set(shards) | {"model.safetensors.index.json"}
    for item in args.input_dir.iterdir():
        if item.name in skip_names:
            continue
        dst = args.output_dir / item.name
        if item.is_file():
            shutil.copy2(item, dst)
        elif item.is_dir():
            shutil.copytree(item, dst, dirs_exist_ok=True)
    log.info("Copied config, tokenizer, and auxiliary files.")

    # ── Final summary ──────────────────────────────────────────────────────────
    if not args.skip_verify and all_verify_results:
        _print_final_report(all_verify_results, calib_desc, args.calib_mode)
    elif args.skip_verify:
        log.info("Verification skipped (--skip_verify).")

    log.info("Done. FP8 model written to: %s", args.output_dir)


if __name__ == "__main__":
    main()