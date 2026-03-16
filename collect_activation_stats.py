#!/usr/bin/env python3
"""
Collect per-layer input-activation statistics from a BF16 Qwen3.5 model
for use with the activation-aware calibration pass in quantize_fp8.py.

Supports all Qwen3.5 sizes across both MoE and dense families:

  MoE   : 397B-A17B, 122B-A10B, 35B-A3B
  Dense : 27B, 9B, 4B, 2B, 0.8B

What this does
──────────────
Registers forward hooks on every eligible linear layer, runs a small set of
calibration prompts through the model in BF16, and records the per-input-
channel maximum absolute activation seen across all tokens and all prompts.

After collection, stats keys are normalised to exactly match the safetensors
key format used in the model's own weight files.  This means quantize_fp8.py
will always find exact matches — no prefix-stripping fallback needed.

Output format
─────────────
    {
      "model.language_model.layers.0.mlp.experts.0.down_proj.weight": {
        "input_channel_max": [0.34, 0.12, ...]   // length == in_features
      },
      ...
    }

Keys always match the safetensors weight names exactly.

Hardware requirements
─────────────────────
Running a 397B model requires a multi-GPU setup (8× H100 80 GB or similar).
Use device_map="auto" with accelerate for multi-GPU loading.

For smaller models (9B and below) this script runs on a single GPU.

Usage
─────
  # Standard (auto multi-GPU, built-in prompts)
  python collect_activation_stats.py \\
      --model_dir ~/models/bf16_finetune \\
      --output    calib_stats_397b.json

  # Short tokens to save time — activation magnitudes stabilise quickly
  python collect_activation_stats.py \\
      --model_dir  ~/models/bf16_finetune \\
      --output     calib_stats_397b.json  \\
      --max_tokens 128

  # Custom prompts from a text file (one prompt per line)
  python collect_activation_stats.py \\
      --model_dir    ~/models/bf16_finetune \\
      --output       calib_stats_397b.json  \\
      --prompts_file my_prompts.txt
"""

import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional

import torch

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

# ── Patterns that identify ineligible weight tensors ───────────────────────────
# Must stay in sync with SKIP_PATTERNS in quantize_fp8.py.
# We only want hooks on layers that will actually be quantized.

SKIP_PATTERNS = [
    # Embeddings & output
    "embed_tokens", "lm_head",
    # Normalization
    "norm", "layernorm",
    # Gated DeltaNet SSM tensors
    "A_log", "dt_bias", "conv1d",
    # Low-rank linear_attn projections (first dim < BLOCK_SIZE in quantizer)
    "in_proj_a", "in_proj_b",
    # MoE router weights (kept BF16 in all checkpoints)
    "shared_expert_gate", "mlp.gate.",
    # MTP fusion weight (kept BF16 in all checkpoints)
    "mtp.fc",
    # Entire vision encoder — never quantized
    "visual",
    # Misc
    "weight_scale_inv", "bias",
]

BLOCK_SIZE = 128


def _is_eligible(name: str, module: torch.nn.Module) -> bool:
    """Return True if this module's weight will be FP8-quantized."""
    if not hasattr(module, "weight"):
        return False
    w = module.weight
    if w is None or w.ndim != 2:
        return False
    if w.shape[0] < BLOCK_SIZE or w.shape[1] < BLOCK_SIZE:
        return False
    for pat in SKIP_PATTERNS:
        if pat in name:
            return False
    return True


# ── Key normalization ──────────────────────────────────────────────────────────

def _load_safetensors_weight_names(model_dir: Path) -> Optional[List[str]]:
    """
    Return the list of weight tensor names from the model's safetensors index.
    Returns None if the index cannot be read (single-file or non-standard layout).
    """
    index_path = model_dir / "model.safetensors.index.json"
    if not index_path.exists():
        # Single-shard model — try to open it directly
        shard_path = model_dir / "model.safetensors"
        if shard_path.exists():
            try:
                from safetensors import safe_open
                with safe_open(str(shard_path), framework="pt", device="cpu") as f:
                    return list(f.keys())
            except Exception:
                return None
        return None

    try:
        with open(index_path) as f:
            index = json.load(f)
        return list(index.get("weight_map", {}).keys())
    except Exception:
        return None


def _normalise_keys(
    raw_stats:    Dict[str, torch.Tensor],
    sf_names_set: set,
) -> Dict[str, torch.Tensor]:
    """
    Remap stats keys to exactly match the safetensors key format.

    named_modules() may produce keys with or without a leading "model."
    component depending on the model class.  For example:

      named_modules key : "language_model.layers.0.mlp.experts.0.down_proj.weight"
      safetensors key   : "model.language_model.layers.0.mlp.experts.0.down_proj.weight"

    Strategy: for each stats key, try the key as-is, then try stripping or
    adding the common prefix variants until we find an exact match in the
    safetensors key set.
    """
    normalised: Dict[str, torch.Tensor] = {}
    unmatched: List[str] = []

    for stats_key, val in raw_stats.items():
        # 1. Exact match
        if stats_key in sf_names_set:
            normalised[stats_key] = val
            continue

        # 2. Try adding prefixes (stats key is shorter than safetensors key)
        matched = None
        for prefix in ("model.", "model.language_model."):
            candidate = prefix + stats_key
            if candidate in sf_names_set:
                matched = candidate
                break

        if matched:
            normalised[matched] = val
            continue

        # 3. Try stripping leading components from the stats key
        parts = stats_key.split(".")
        for i in range(1, len(parts) - 1):
            candidate = ".".join(parts[i:])
            if candidate in sf_names_set:
                matched = candidate
                break
            for prefix in ("model.", "model.language_model."):
                full = prefix + candidate
                if full in sf_names_set:
                    matched = full
                    break
            if matched:
                break

        if matched:
            normalised[matched] = val
        else:
            unmatched.append(stats_key)

    return normalised, unmatched


# ── Default calibration prompts ────────────────────────────────────────────────

DEFAULT_PROMPTS = [
    # Reasoning
    "Explain the difference between induction and deduction in logic, "
    "and give one example of each.",
    "A train leaves city A at 60 km/h. Another leaves city B at 80 km/h "
    "toward A. The cities are 280 km apart. When do they meet?",
    # Coding
    "Write a Python function that merges two sorted lists into one sorted list "
    "without using the built-in sort.",
    "Explain what a race condition is in concurrent programming and show a "
    "minimal Python example using threading.",
    # Math
    "Prove that the square root of 2 is irrational.",
    "What is the derivative of f(x) = x^3 * sin(x)? Show the steps.",
    # Summarisation
    "Summarise the key differences between transformer-based and recurrent "
    "architectures for sequence modelling.",
    # Instruction following
    "List five strategies for reducing hallucinations in large language models. "
    "For each strategy, briefly explain the intuition behind it.",
    # Multilingual
    "Translate the following sentence to French and explain any idiomatic "
    "differences: 'It's raining cats and dogs outside.'",
    # Tool use / agent
    "You are an AI assistant with access to a calculator tool. "
    "The user asks: 'What is 17 factorial?' Explain your plan and compute it.",
    # Domain diversity
    "Describe the mechanism by which CRISPR-Cas9 introduces a double-strand "
    "break into target DNA.",
    "What are the main differences between TCP and UDP? When would you use each?",
]


# ── Hook infrastructure ────────────────────────────────────────────────────────

class _ActivationCollector:
    """
    Accumulates per-input-channel maximum absolute activation across all
    forward passes and all token positions.

    For a linear layer weight [out, in], inputs arrive as [batch, seq, in]
    or [tokens, in].  We keep a running max per input channel (last dim).
    """

    def __init__(self, in_features: int):
        self.channel_max = torch.zeros(in_features, dtype=torch.float32)

    def update(self, x: torch.Tensor) -> None:
        flat = x.detach().float().reshape(-1, x.shape[-1])
        batch_max = flat.abs().max(dim=0).values.cpu()
        self.channel_max = torch.maximum(self.channel_max, batch_max)


def _make_hook(collector: _ActivationCollector):
    def hook(module, inputs, output):
        collector.update(inputs[0])
    return hook


# ── Main collection logic ──────────────────────────────────────────────────────

def collect_stats(
    model_dir:   Path,
    prompts:     List[str],
    max_tokens:  int,
    output_path: Path,
    device_map:  str,
) -> None:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        raise SystemExit(
            "transformers is required: pip install transformers"
        )

    # ── Load safetensors key list for later normalisation ──────────────────────
    log.info("Reading safetensors weight names from %s ...", model_dir)
    sf_names = _load_safetensors_weight_names(model_dir)
    if sf_names is None:
        log.warning(
            "  Could not read safetensors index — "
            "stats keys will use named_modules() format as-is."
        )
        sf_names_set = set()
    else:
        sf_names_set = set(sf_names)
        log.info("  Found %d safetensors weight names.", len(sf_names_set))

    # ── Load model ─────────────────────────────────────────────────────────────
    log.info("Loading tokenizer from %s ...", model_dir)
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir), trust_remote_code=True
    )

    log.info("Loading model (device_map=%s) — this may take several minutes ...",
             device_map)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        trust_remote_code=True,
    )
    model.eval()

    # ── Register hooks on eligible linear layers ───────────────────────────────
    collectors: Dict[str, _ActivationCollector] = {}
    hooks = []

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and _is_eligible(name, module):
            weight_key = name + ".weight"
            collector  = _ActivationCollector(module.in_features)
            collectors[weight_key] = collector
            hooks.append(module.register_forward_hook(_make_hook(collector)))

    log.info("Registered hooks on %d eligible layers.", len(collectors))

    # Show a sample of what named_modules() produced so mismatches are visible
    sample = list(collectors.keys())[:5]
    log.info("  named_modules() key sample (first 5):")
    for k in sample:
        log.info("    %s", k)

    # ── Forward pass over calibration prompts ──────────────────────────────────
    log.info("Running %d calibration prompts (max_new_tokens=%d) ...",
             len(prompts), max_tokens)

    with torch.inference_mode():
        for i, prompt in enumerate(prompts, 1):
            log.info("  Prompt %d/%d ...", i, len(prompts))
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048,
            )
            # Move each input tensor to the model's first device
            first_device = next(model.parameters()).device
            inputs = {k: v.to(first_device) for k, v in inputs.items()}

            model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
            )

    for h in hooks:
        h.remove()
    log.info("Hooks removed.")

    # ── Build raw stats dict ───────────────────────────────────────────────────
    raw_stats: Dict[str, torch.Tensor] = {
        k: col.channel_max for k, col in collectors.items()
    }

    # ── Normalise keys to safetensors format ───────────────────────────────────
    if sf_names_set:
        log.info("Normalising stats keys to safetensors format ...")
        normalised, unmatched = _normalise_keys(raw_stats, sf_names_set)

        if unmatched:
            log.warning(
                "  %d stats keys could not be matched to any safetensors key "
                "(they will be dropped):", len(unmatched)
            )
            for k in unmatched[:10]:
                log.warning("    %s", k)
            if len(unmatched) > 10:
                log.warning("    ... and %d more", len(unmatched) - 10)

        matched_count = len(normalised)
        log.info(
            "  Normalised %d / %d keys successfully.",
            matched_count, len(raw_stats),
        )

        # Show a sample of normalised keys
        norm_sample = list(normalised.keys())[:5]
        log.info("  Normalised key sample (first 5):")
        for k in norm_sample:
            log.info("    %s", k)

        final_stats = normalised
    else:
        log.info("  Skipping normalisation (no safetensors index available).")
        final_stats = raw_stats

    # ── Write output ───────────────────────────────────────────────────────────
    output = {
        weight_name: {"input_channel_max": val.tolist()}
        for weight_name, val in final_stats.items()
    }

    with open(output_path, "w") as f:
        json.dump(output, f)

    log.info("Wrote activation stats for %d layers to %s.",
             len(output), output_path)

    # Warn about layers with all-zero activations
    zero_layers = [
        k for k, col in collectors.items()
        if col.channel_max.max().item() == 0.0
    ]
    if zero_layers:
        log.warning(
            "%d layers had all-zero activations (likely never activated "
            "by the calibration prompts). Consider adding more diverse prompts.",
            len(zero_layers),
        )


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    parser.add_argument("--model_dir", required=True, type=Path,
                        help="Directory containing the BF16 model.")
    parser.add_argument("--output", required=True, type=Path,
                        help="Output JSON file for activation stats.")
    parser.add_argument("--prompts_file", type=Path, default=None,
                        help="Text file with one calibration prompt per line. "
                             "Uses built-in defaults when omitted.")
    parser.add_argument("--max_tokens", type=int, default=256,
                        help="Max new tokens per prompt. (default: 256) "
                             "128 is sufficient for activation stats.")
    parser.add_argument("--device_map", default="auto",
                        help="Transformers device_map: 'auto' for multi-GPU, "
                             "'cpu' for CPU-only. (default: auto)")
    args = parser.parse_args()

    if args.prompts_file is not None:
        prompts = [
            line.strip()
            for line in args.prompts_file.read_text().splitlines()
            if line.strip()
        ]
        log.info("Loaded %d prompts from %s.", len(prompts), args.prompts_file)
    else:
        prompts = DEFAULT_PROMPTS
        log.info("Using %d built-in default prompts.", len(prompts))

    collect_stats(
        model_dir=args.model_dir,
        prompts=prompts,
        max_tokens=args.max_tokens,
        output_path=args.output,
        device_map=args.device_map,
    )


if __name__ == "__main__":
    main()