"""Tiny Arabic eval for Wamda-3B-Reasoning.

Scores how many answers contain the expected string. Uses mlx_lm.generate CLI
so it works with or without the trained adapter.

Usage:
    ./.venv/bin/python eval/eval.py [--adapter-path ./adapters | --base-only] [--limit 5]
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

SYSTEM = "أنت مساعد رياضيات بارع. فكّر خطوة بخطوة داخل <think> قبل إعطاء الإجابة النهائية."
ROOT = Path(__file__).resolve().parent.parent


def run_prompt(model: str, question: str, extra: list[str]) -> str:
    prompt = (f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
              f"<|im_start|>user\n{question}<|im_end|>\n"
              f"<|im_start|>assistant\n<think>\n")
    cmd = [str(ROOT / ".venv" / "bin" / "python"), "-m", "mlx_lm.generate",
           "--model", model, "--prompt", prompt, "--max-tokens", "256", "--temp", "0.0"] + extra
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    return (r.stdout or "") + (r.stderr or "")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--adapter-path", default=str(ROOT / "adapters"))
    ap.add_argument("--base-only", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    extra = [] if args.base_only else ["--adapter-path", args.adapter_path]
    rows = [json.loads(l) for l in open(ROOT / "eval" / "eval_set.jsonl", encoding="utf-8") if l.strip()]
    if args.limit:
        rows = rows[:args.limit]

    ok, outs = 0, []
    for i, row in enumerate(rows, 1):
        out = run_prompt(args.model, row["question"], extra)
        tail = out.split("</think>")[-1] if "</think>" in out else out
        hit = row["answer_contains"] in tail or row["answer_contains"] in out
        ok += hit
        outs.append({"q": row["question"], "expected": row["answer_contains"], "hit": hit,
                     "output_tail": tail[-300:]})
        print(f"[{i}/{len(rows)}] {'PASS' if hit else 'FAIL'} expected={row['answer_contains']!r}")

    print(f"\nSCORE {ok}/{len(rows)} = {100 * ok / len(rows):.1f}% ({'base' if args.base_only else 'adapter'})")
    json.dump(outs, open(ROOT / "eval" / f"results_{'base' if args.base_only else 'adapter'}.json",
                         "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
