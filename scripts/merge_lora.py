"""Merge MLX LoRA adapter into base weights (transformers format) for GGUF export.

MLX <-> PEFT mapping (verified against adapters.safetensors):
  mlx "model.layers.N.<mod>.lora_a" (in, r)  -> peft "lora_A.weight" (r, in)  [transpose]
  mlx "model.layers.N.<mod>.lora_b" (r, out) -> peft "lora_B.weight" (out, r) [transpose]
  mlx scale 20.0, rank 8  -> peft lora_alpha=160 (alpha/rank == scale)

Usage:
    ./.venv/bin/python scripts/merge_lora.py [--verify]
"""
import argparse
import sys

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from safetensors.numpy import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = "Qwen/Qwen2.5-3B-Instruct"
RANK, ALPHA = 8, 160
LAYERS = list(range(20, 36))
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def load_mlx_adapter(path: str) -> dict:
    raw = load_file(path)
    out = {}
    for k, v in raw.items():
        tail = k.rsplit(".", 1)[1]  # lora_a | lora_b
        peft_tail = {"lora_a": "lora_A", "lora_b": "lora_B"}[tail]
        new_k = f"base_model.model.{k.rsplit('.', 1)[0]}.{peft_tail}.default.weight"
        out[new_k] = torch.from_numpy(v.T.copy())  # (in,r)/(r,out) -> (r,in)/(out,r)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="./adapters/adapters.safetensors")
    ap.add_argument("--out", default="./exports/merged_hf")
    ap.add_argument("--verify", action="store_true",
                    help="compare one merged weight against mlx-fused model in ./exports/fused")
    args = ap.parse_args()

    print("loading base", BASE, flush=True)
    model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.float32)
    tok = AutoTokenizer.from_pretrained(BASE)

    cfg = LoraConfig(r=RANK, lora_alpha=ALPHA, target_modules=TARGETS,
                     layers_to_transform=LAYERS, bias="none",
                     task_type="CAUSAL_LM")
    model = get_peft_model(model, cfg)
    state = load_mlx_adapter(args.adapter)
    missing, unexpected = model.load_state_dict(state, strict=False)
    still_missing = [k for k in missing if ".lora_" in k]
    assert not still_missing and not unexpected, f"adapter load issue: {still_missing[:3]} {unexpected[:3]}"
    print(f"adapter loaded: {len(state)} tensors", flush=True)

    merged = model.merge_and_unload()
    merged = merged.to(torch.bfloat16)
    merged.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    print("saved merged model to", args.out, flush=True)

    if args.verify:
        from safetensors.torch import load_file as load_torch
        fused = load_torch("./exports/fused/model-00001-of-00002.safetensors")
        name = "model.layers.20.self_attn.q_proj.weight"
        a = fused[name].float()
        merged_sd = merged.state_dict()
        if name not in merged_sd:  # sharded save
            from safetensors.torch import load_file as _lf
            import glob
            merged_sd = {}
            for sh in glob.glob(f"{args.out}/model-*.safetensors"):
                merged_sd.update(_lf(sh))
        b = merged_sd[name].float()
        diff = (a - b).abs().max().item()
        print(f"max abs diff on {name}: {diff:.2e}")
        assert diff < 5e-2, "fused weights diverge — mapping wrong?"
        print("verify OK: transformers merge matches mlx fuse")
    return 0


if __name__ == "__main__":
    sys.exit(main())
