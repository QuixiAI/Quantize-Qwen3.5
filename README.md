# Qwen3.5-397B FP8 Quantization Toolkit

Block-wise FP8 E4M3 quantization for Qwen3.5 BF16 finetuned models.
Produces a checkpoint that is **drop-in compatible** with the official
`Qwen/Qwen3.5-397B-A17B-FP8` format and loads directly in vLLM and SGLang
without any engine modifications.

---

## Output format

Every eligible weight tensor becomes a matched pair:

```
layer.weight            →  float8_e4m3fn   [out, in]
layer.weight_scale_inv  →  bfloat16        [ceil(out/128), ceil(in/128)]
```

Scales are block-wise: one BF16 inverse-scale per 128×128 tile.
At inference the engine dequantizes as `W_fp32 ≈ W_fp8 × scale_inv`.

Numerically sensitive tensors (embeddings, norms, SSM parameters, router
gates, low-rank projections) are passed through unchanged in BF16/F32.

---

## Scripts

| Script | Purpose |
|---|---|
| `quantize_fp8.py` | Main quantization script. Processes one shard at a time — never loads the full model. |
| `collect_activation_stats.py` | Optional. Collects per-layer input-activation statistics for activation-aware calibration. Requires loading the full BF16 model. |

---

## Installation

```bash
pip install torch safetensors transformers tqdm
```

CUDA is auto-detected and used when available. CPU-only also works, but is
significantly slower for MSE mode.

---

## Quick start

### Fastest — RMS scaling, no activation stats

```bash
python quantize_fp8.py \
    --input_dir  bf16_model \
    --output_dir fp8_model
```

`rms` is the default mode. It is outlier-robust, requires no calibration
forward pass, and runs at the same speed as raw max scaling.

### With verification (recommended)

Verification is **on by default**. The script re-opens each output shard
after writing it, spot-checks a random sample of FP8 tensor pairs, and
prints a health report at the end. No extra flags needed:

```bash
python quantize_fp8.py \
    --input_dir      bf16_model \
    --output_dir     fp8_model \
    --calib_mode     rms \
    --verify_samples 20
```

`--verify_samples` controls how many tensors are checked per shard
(default: 5). 20 gives a thorough check with modest overhead.

### Best quality — activation-aware RMS

Collect activation statistics first (requires loading the full BF16 model),
then pass the stats file to the quantizer:

```bash
# Step 1 — collect activation statistics
python collect_activation_stats.py \
    --model_dir bf16_model \
    --output    calib_stats.json \
    --max_tokens 128

# Step 2 — quantize with activation-aware RMS
python quantize_fp8.py \
    --input_dir        bf16_model \
    --output_dir       fp8_model \
    --calib_mode       rms \
    --activation_stats calib_stats.json
```

### Best possible quality — activation-aware MSE

Slower but highest accuracy. Use when export time is not a constraint:

```bash
python quantize_fp8.py \
    --input_dir        bf16_model \
    --output_dir       fp8_model \
    --calib_mode       mse \
    --activation_stats calib_stats.json
```

---

## Calibration modes

All modes are **fully vectorized** over the block grid — there is no Python
loop over the `(n_out × n_in)` block tensor. MSE iterates over alpha values
only (`--mse_grid_size` steps), each of which is a single fused kernel over
all blocks simultaneously.

| Mode | Speed | Outlier robustness | Quality | Notes |
|---|---|---|---|---|
| `rms` | ★★★★★ | ★★★★☆ | ★★★☆☆ | **Default.** `scale = rms(block) × k / FP8_MAX`. Outliers saturate to ±448 instead of compressing everything else. |
| `max` | ★★★★★ | ★☆☆☆☆ | ★★☆☆☆ | `scale = max(|block|) / FP8_MAX`. One outlier inflates the scale and collapses all other values. |
| `percentile` | ★★★★☆ | ★★★★☆ | ★★★☆☆ | Clips at `--calib_percentile` (default 99.9) before scaling. Good explicit control. |
| `mse` | ★★☆☆☆ | ★★★☆☆ | ★★★★☆ | Grid search over clipping factors. Best weight-only quality. |

### Activation stats combinations

| Mode | Activation stats | What runs |
|---|---|---|
| `rms` | no | Plain RMS |
| `rms` | yes | **Activation-weighted RMS** — channels with large activations dominate the RMS estimate, tightening the scale where it matters most. Fast, no grid search. |
| `mse` | no | Plain MSE grid search |
| `mse` | yes | Activation-weighted MSE — minimises `Σ_j a_j · Σ_i (W_q_ij − W_ij)²`. Highest quality. |
| `max` / `percentile` | yes/no | Activation stats ignored for these modes (no differentiable loss to weight). |

### `--rms_factor` tuning

The RMS factor `k` (`--rms_factor`, default `3.0`) approximates the
peak-to-RMS ratio of the weight distribution.

- `3.0` — suits near-Gaussian weights (most finetuned models)
- `4.0` — heavier tails; reduces saturation of outlier blocks
- `2.5` — more aggressive clipping; lower MAE for typical values but more
  outlier saturation

The verification report will guide you: if `maxe` is WARN, increase `k`;
if `mae` is WARN, decrease `k`.

---

## Verification

After each shard is written the script re-opens both the input and output
files, dequantizes a random sample of FP8 tensor pairs back to BF16, and
measures three metrics against the original weights.

### Three-tier health bands

| Zone | Cosine similarity | Mean abs error | Max abs error |
|---|---|---|---|
| ✔ **GOOD** | ≥ 0.9997 | ≤ 0.0008 | ≤ 0.01 |
| ⚠ **WARN** | 0.9995 – 0.9997 | 0.0008 – 0.001 | 0.01 – 0.02 |
| ✘ **FAIL** | < 0.9995 | > 0.001 | > 0.02 |

WARN means the model can technically deploy but quality may be slightly
degraded. FAIL means do not deploy.

### Example output — healthy model

```
================================================================
  QUANTIZATION VERIFICATION SUMMARY
  Calibration         : rms(k=3.0)
----------------------------------------------------------------
  Shards processed    : 95  (0 failed)
  FP8 weight tensors  : 18240
  Passthrough tensors : 420
  Spot-checked        : 1900  (✔ 1900 GOOD  ⚠ 0 WARN  ✘ 0 FAIL)
----------------------------------------------------------------
  metric          mean value  typical range  band        hard limit
  ---------------------------------------------------------------
  cosine sim      0.999812    0.9997–...     ✔ GOOD      ≥ 0.9995
  mean abs err    0.000612    0.0004–0.0008  ✔ GOOD      ≤ 0.001
  worst MAE       0.000891                   ✔ GOOD      ≤ 0.001
  worst MaxAE     0.009240                   ✔ GOOD      ≤ 0.02
----------------------------------------------------------------
✔  All metrics within typical range. Model ready for inference.
================================================================
```

### Example output — degraded (with recommendations)

```
================================================================
  QUANTIZATION VERIFICATION SUMMARY
  Calibration         : max
----------------------------------------------------------------
  ...
  cosine sim      0.999521    0.9997–...     ⚠ WARN      ≥ 0.9995
  mean abs err    0.000923    0.0004–0.0008  ⚠ WARN      ≤ 0.001
----------------------------------------------------------------
⚠  Metrics within hard limits but outside typical range.
   See calibration recommendations below.
----------------------------------------------------------------
  CALIBRATION RECOMMENDATIONS
  1. Cosine similarity avg=0.999521 is in the WARN zone (typical ≥ 0.9997).
     Switch to --calib_mode rms (same speed, outlier-robust) or
     --calib_mode mse for best quality.
================================================================
```

---

## Hardware requirements

### `quantize_fp8.py`

Processes **one shard at a time** — never loads the full model. Peak memory
is roughly one shard plus the CUDA workspace (~8–16 GB for a 95-shard
397B layout). Runs comfortably on a single GPU or CPU-only.

### `collect_activation_stats.py`

Requires loading the **full BF16 model** to run calibration forward passes.
At 397B × 2 bytes ≈ ~800 GB VRAM. Options:

- Multi-GPU node with `--device_map auto` (recommended)
- CPU offloading — works but slow
- Skip entirely and use `--calib_mode rms` without activation stats

---

## Runtime estimates (single H100 80 GB)

| Mode | ~Time for 397B |
|---|---|
| `rms` | 20–40 min |
| `rms` + verify (20 samples) | 30–60 min |
| `mse` (grid=50) | 3–6 hours |
| `mse` + verify (20 samples) | 4–8 hours |

---

## All flags

### `quantize_fp8.py`

#### I/O
| Flag | Default | Description |
|---|---|---|
| `--input_dir` | *(required)* | Directory containing BF16 safetensors shards |
| `--output_dir` | *(required)* | Output directory for FP8 shards |

#### Calibration
| Flag | Default | Description |
|---|---|---|
| `--calib_mode` | `rms` | `rms` / `max` / `percentile` / `mse` |
| `--rms_factor` | `3.0` | Peak-to-RMS multiplier `k` for `rms` mode |
| `--calib_percentile` | `99.9` | Clip percentile for `percentile` mode |
| `--mse_grid_size` | `50` | Alpha candidates for MSE grid search |
| `--activation_stats` | *(none)* | JSON file from `collect_activation_stats.py` |
| `--device` | *(auto)* | `cuda`, `cuda:1`, `cpu`, etc. |

#### Verification
| Flag | Default | Description |
|---|---|---|
| `--verify_samples` | `5` | FP8 tensors to spot-check per shard |
| `--skip_verify` | off | Skip verification entirely |

### `collect_activation_stats.py`

| Flag | Default | Description |
|---|---|---|
| `--model_dir` | *(required)* | BF16 model directory |
| `--output` | *(required)* | Output JSON path |
| `--prompts_file` | *(built-in)* | Text file, one prompt per line |
| `--max_tokens` | `256` | Max new tokens per calibration prompt |
| `--device_map` | `auto` | Passed to `from_pretrained` |

---

## Inference

The output is directly loadable with vLLM and SGLang using the same commands
as the official Qwen FP8 checkpoint:

```bash
# vLLM
vllm serve /path/to/fp8_model \
    --tensor-parallel-size 8 \
    --max-model-len 262144 \
    --reasoning-parser qwen3

# SGLang
python -m sglang.launch_server \
    --model-path /path/to/fp8_model \
    --tp-size 8 \
    --context-length 262144 \
    --reasoning-parser qwen3
```

---

## What gets quantized

Eligible tensors (large 2D weight matrices in attention and MoE layers)
are stored as FP8 E4M3. Everything else passes through unchanged:

| Tensor | Dtype | Reason |
|---|---|---|
| `*in_proj_qkv.weight`, `*out_proj.weight`, `*in_proj_z.weight` | FP8 | Large attention projections |
| `*experts.*.{down,gate,up}_proj.weight` | FP8 | Expert MLP weights |
| `*shared_expert.{down,gate,up}_proj.weight` | FP8 | Shared expert weights |
| `embed_tokens.weight` | BF16 | Embedding lookup, no matmul |
| `*norm*`, `*layernorm*` | BF16 | Numerically sensitive |
| `*in_proj_a.weight`, `*in_proj_b.weight` | BF16 | Small low-rank [64, 4096] |
| `*A_log*` | F32 | SSM state parameter |
| `*dt_bias*`, `*conv1d*` | BF16 | SSM auxiliary tensors |
| `*shared_expert_gate.weight` | BF16 | Router gate [1, 4096] |
