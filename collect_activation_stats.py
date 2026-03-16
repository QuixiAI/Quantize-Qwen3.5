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

The result is a JSON file consumed by quantize_fp8.py --activation_stats.

Output format
─────────────
    {
      "model.layers.0.mlp.experts.0.down_proj.weight": {
        "input_channel_max": [0.34, 0.12, ...]   // length == in_features
      },
      ...
    }

Hardware requirements
─────────────────────
Running a 397B model requires a multi-GPU setup (8× H100 80 GB or similar).
Use tensor_parallel via accelerate / vLLM offline batching / FSDP for loading.

For smaller models (9B and below) this script runs on a single GPU.

Usage
─────
  # Minimal (uses built-in default prompts)
  python collect_activation_stats.py \\
      --model_dir /models/bf16_finetune \\
      --output    calib_stats.json

  # Custom calibration prompts from a text file (one prompt per line)
  python collect_activation_stats.py \\
      --model_dir    /models/bf16_finetune \\
      --output       calib_stats.json      \\
      --prompts_file my_prompts.txt        \\
      --max_tokens   512

  # Multi-GPU via accelerate
  accelerate launch --multi_gpu collect_activation_stats.py \\
      --model_dir /models/bf16_finetune \\
      --output    calib_stats.json
"""

import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List

import torch

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

# ── Patterns that identify ineligible weight tensors ───────────────────────────
# MUST stay in sync with SKIP_PATTERNS in quantize_fp8.py.
# We only want hooks on layers that will actually be quantized.
# Hooking extra layers wastes forward-pass memory and bloats the stats file.

SKIP_PATTERNS = [
    # Embeddings & output
    "embed_tokens", "lm_head",
    # Normalization
    "norm", "layernorm",
    # Gated DeltaNet SSM tensors
    "A_log", "dt_bias", "conv1d",
    # Low-rank linear_attn projections (first dim < BLOCK_SIZE)
    "in_proj_a", "in_proj_b",
    # MoE router weights (kept BF16)
    "shared_expert_gate", "mlp.gate.",
    # MTP fusion weight (kept BF16)
    "mtp.fc",
    # Entire vision encoder — ViT blocks, merger, patch embed, pos embed
    # (~100+ linear layers that are never quantized; skip to save hooks + memory)
    "visual",
    # Misc
    "weight_scale_inv", "bias",
]


def _is_eligible(name: str, module: torch.nn.Module) -> bool:
    if not hasattr(module, "weight"):
        return False
    w = module.weight
    if w is None or w.ndim != 2:
        return False
    if w.shape[0] < 128 or w.shape[1] < 128:
        return False
    for pat in SKIP_PATTERNS:
        if pat in name:
            return False
    return True


# ── Default calibration prompts ────────────────────────────────────────────────
# A short, diverse set that covers reasoning, code, math, multilingual, and
# instruction-following — representative of Qwen3.5's expected workloads.

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
    # Long context / summarisation
    "Summarise the key differences between transformer-based and recurrent "
    "architectures for sequence modelling.",
    # Instruction following
    "List five strategies for reducing hallucinations in large language models. "
    "For each strategy, briefly explain the intuition behind it.",
    # Multilingual
    "Translate the following sentence to French and explain any idiomatic "
    "differences: 'It's raining cats and dogs outside.'",
    # Vision-language (text description)
    "Describe what information you would need to determine the depth of an "
    "object in a monocular photograph.",
    # Tool use / agent
    "You are an AI assistant with access to a calculator tool. "
    "The user asks: 'What is 17 factorial?' Explain your plan and compute it.",
    # Diverse domain — biology
    "Describe the mechanism by which CRISPR-Cas9 introduces a double-strand "
    "break into target DNA.",
]


# ── Hook infrastructure ────────────────────────────────────────────────────────

class _ActivationCollector:
    """
    Accumulates per-input-channel maximum absolute activations across
    all forward passes and all sequence positions.

    For a linear layer with weight [out, in], the input to the layer has
    shape [batch, seq_len, in].  We reduce over batch and seq_len, keeping
    the running max per input channel (dim=-1).
    """

    def __init__(self, in_features: int):
        self.channel_max = torch.zeros(in_features, dtype=torch.float32)

    def update(self, x: torch.Tensor) -> None:
        # x: [batch, seq, in] or [tokens, in]
        flat = x.detach().float().reshape(-1, x.shape[-1])
        batch_max = flat.abs().max(dim=0).values.cpu()
        self.channel_max = torch.maximum(self.channel_max, batch_max)


def _make_hook(collector: _ActivationCollector):
    def hook(module, inputs, output):
        x = inputs[0]
        collector.update(x)
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
            "transformers is required. Install with: pip install transformers"
        )

    log.info("Loading tokenizer from %s ...", model_dir)
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)

    log.info("Loading model from %s (device_map=%s) ...", model_dir, device_map)
    log.info("  This may take several minutes for large models.")
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
            weight_name = name + ".weight"
            collector = _ActivationCollector(module.in_features)
            collectors[weight_name] = collector
            hooks.append(module.register_forward_hook(_make_hook(collector)))

    log.info("Registered hooks on %d eligible layers.", len(collectors))

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
            ).to(model.device)

            model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
            )

    # ── Remove hooks ───────────────────────────────────────────────────────────
    for h in hooks:
        h.remove()
    log.info("Hooks removed.")

    # ── Serialise stats ────────────────────────────────────────────────────────
    stats = {
        weight_name: {
            "input_channel_max": col.channel_max.tolist()
        }
        for weight_name, col in collectors.items()
    }

    with open(output_path, "w") as f:
        json.dump(stats, f)

    log.info("Wrote activation stats for %d layers to %s.", len(stats), output_path)

    # Sanity: warn about layers with zero activation (likely never reached)
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
                        help="Max new tokens per prompt during calibration. "
                             "(default: 256)")
    parser.add_argument("--device_map", default="auto",
                        help="Transformers device_map. Use 'auto' for multi-GPU, "
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
