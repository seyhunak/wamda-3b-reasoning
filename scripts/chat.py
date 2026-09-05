"""Single-prompt inference for Wamda-3B-Reasoning (MLX).

Usage:
    ./.venv/bin/python scripts/chat.py "كم يساوي 47 × 36؟" [--base-only]
"""
import argparse
import sys

SYSTEM = "أنت مساعد رياضيات بارع. فكّر خطوة بخطوة داخل <think> قبل إعطاء الإجابة النهائية."


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--adapter-path", default="./adapters")
    ap.add_argument("--base-only", action="store_true")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--temp", type=float, default=0.0)
    args = ap.parse_args()

    from mlx_lm import load, generate
    from mlx_lm.sample_utils import make_sampler

    if args.base_only:
        model, tokenizer = load(args.model)
    else:
        model, tokenizer = load(args.model, adapter_path=args.adapter_path)

    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": args.question},
        {"role": "assistant", "content": "<think>\n"},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    sampler = make_sampler(temp=args.temp)
    out = generate(model, tokenizer, prompt=prompt, max_tokens=args.max_tokens,
                   sampler=sampler)
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
