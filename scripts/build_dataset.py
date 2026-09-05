"""Build Arabic <think> SFT dataset for Wamda-3B-Reasoning.

Generates deterministic, arithmetically-verified Arabic math/logic examples in
mlx-lm chat format: {"messages": [system, user, assistant]} where the assistant
response wraps step-by-step reasoning in <think>...</think> followed by the final answer.

Usage:
    python3 scripts/build_dataset.py --train-n 300 --valid-n 40 --seed 7
"""
import argparse
import json
import random
from pathlib import Path

SYSTEM = "أنت مساعد رياضيات بارع. فكّر خطوة بخطوة داخل <think> قبل إعطاء الإجابة النهائية."

AR_NAMES = ["أحمد", "سارة", "خالد", "فاطمة", "محمد", "ليلى", "عمر", "نورة", "يوسف", "مريم"]


def ex(user: str, think_lines: list[str], final: str) -> dict:
    think = "\n".join(f"الخطوة {i + 1}: {line}" for i, line in enumerate(think_lines))
    return {
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user},
            {"role": "assistant", "content": f"<think>\n{think}\n</think>\n\n{final}"},
        ]
    }


def gen_boxes(rng: random.Random) -> dict:
    n, per = rng.randint(2, 5), rng.randint(12, 30)
    total = n * per
    give = total // 2
    left = total - give
    sell = left // 3
    price = round(rng.uniform(1.0, 3.0), 2)
    earned = round(sell * price, 2)
    name = rng.choice(AR_NAMES)
    user = (f"اشترى {name} {n} صناديق من التفاح، يحتوي كل صندوق على {per} تفاحة. "
            f"إذا تبرع بنصف التفاح، وباع ثلث ما تبقى بسعر {price} ريال للتفاحة، كم ريالاً جنى؟")
    return ex(user, [
        f"احسب العدد الإجمالي: {n} × {per} = {total} تفاحة",
        f"احسب عدد التفاح المتبرع به: {total} / 2 = {give} تفاحة",
        f"احسب المتبقي: {total} − {give} = {left} تفاحة",
        f"احسب المباع: {left} / 3 = {sell} تفاحة",
        f"احسب المبلغ: {sell} × {price} = {earned} ريالًا",
    ], f"{earned} ريالًا")


def gen_cups(rng: random.Random) -> dict:
    pairs = rng.randint(3, 10)
    n = pairs * 2
    full, pct = rng.randint(4, 8), rng.choice([50, 60, 75])
    disc = round(full * pct / 100, 2)
    total = round(pairs * full + pairs * disc, 2)
    user = (f"تكلفة الكوب الواحد {full} دولارات، لكن كل كوب ثانٍ يكلف {pct}٪ فقط من السعر. "
            f"كم يدفع من يشتري {n} كوبًا؟")
    return ex(user, [
        f"سعر الكوب الأول {full} دولارات",
        f"سعر الكوب الثاني {pct}٪ من {full} = {disc} دولارات",
        f"عدد الأزواج في {n} كوبًا = {pairs}",
        f"الأكواب الكاملة: {pairs} × {full} = {round(pairs * full, 2)}",
        f"الأكواب المخفضة: {pairs} × {disc} = {round(pairs * disc, 2)}",
        f"المجموع: {round(pairs * full, 2)} + {round(pairs * disc, 2)} = {total} دولارات",
    ], f"{total} دولارات")


def gen_discount_vat(rng: random.Random) -> dict:
    base = rng.randint(1000, 9000)
    d = rng.choice([10, 15, 20, 25])
    v = rng.choice([5, 15])
    after_d = round(base * (100 - d) / 100, 2)
    vat = round(after_d * v / 100, 2)
    final = round(after_d + vat, 2)
    user = (f"جهاز سعره الأصلي {base} ريال، عليه خصم {d}٪، ثم ضريبة قيمة مضافة {v}٪ بعد الخصم. ما السعر النهائي؟")
    return ex(user, [
        f"قيمة الخصم: {base} × {d}٪ = {round(base * d / 100, 2)}",
        f"السعر بعد الخصم: {base} − {round(base * d / 100, 2)} = {after_d}",
        f"الضريبة: {after_d} × {v}٪ = {vat}",
        f"السعر النهائي: {after_d} + {vat} = {final}",
    ], f"{final} ريال")


def gen_age(rng: random.Random) -> dict:
    x = rng.randint(4, 12)
    mult = rng.choice([2, 3])
    yrs = rng.randint(3, 8)
    sara = mult * x
    total = (x + yrs) + (sara + yrs)
    name = rng.choice(["سارة", "فاطمة", "ليلى", "نورة", "مريم"])
    user = (f"عمر {name} الآن {mult} أضعاف عمر أخيها الصغير. بعد {yrs} سنوات سيكون مجموع عمريهما {total} سنة. كم عمر {name} الآن؟")
    return ex(user, [
        f"ليكن عمر الأخ الصغير x، إذن عمر {name} هي {mult}x",
        f"بعد {yrs} سنوات: الأخ x + {yrs}، و{name} {mult}x + {yrs}",
        f"المجموع: (x + {yrs}) + ({mult}x + {yrs}) = {mult + 1}x + {2 * yrs}",
        f"إذن {mult + 1}x + {2 * yrs} = {total}، ومنها {mult + 1}x = {total - 2 * yrs}، و x = {x}",
        f"عمر {name} الآن = {mult} × {x} = {sara}",
    ], f"{sara}")


def gen_speed(rng: random.Random) -> dict:
    v = rng.choice([60, 80, 90, 100, 120])
    t = rng.randint(2, 5)
    d = v * t
    user = f"قطار يسير بسرعة {v} كم/سا، كم يستغرق ليقطع {d} كم؟"
    return ex(user, [
        "الوقت = المسافة / السرعة",
        f"الوقت = {d} ÷ {v}",
        f"الوقت = {t} ساعات",
    ], f"{t} ساعات")


def gen_apples(rng: random.Random) -> dict:
    start = rng.choice([20, 24, 30, 36])
    buy = rng.randint(5, 15)
    half = start // 2
    final = half + buy
    name = rng.choice(AR_NAMES)
    user = (f"عند {name} {start} تفاحة، أعطى نصفها لصديقه ثم اشترى {buy} تفاحات إضافية. كم تفاحة أصبح عنده؟")
    return ex(user, [
        f"نصف {start} = {start} / 2 = {half}",
        f"بعد الإعطاء: {start} − {half} = {half}",
        f"بعد الشراء: {half} + {buy} = {final}",
    ], f"{final}")


def gen_mult(rng: random.Random) -> dict:
    a, b = rng.randint(12, 99), rng.randint(11, 49)
    ones, tens = b % 10, b - b % 10
    p1, p2 = a * ones, a * tens
    ans = a * b
    user = f"كم يساوي {a} × {b}؟"
    return ex(user, [
        f"اضرب {a} × {ones} = {p1}",
        f"اضرب {a} × {tens} = {p2}",
        f"اجمع {p1} + {p2} = {ans}",
    ], f"{ans}")


TRAP_PILLS = ex(
    "إذا كان طبيب يعطيك 3 حبوب وطلب منك أخذ حبة كل نصف ساعة، فكم من الوقت تستغرق حتى تنهي الحبوب؟",
    ["الحبة الأولى تؤخذ فورًا في الدقيقة 0",
     "الحبة الثانية بعد نصف ساعة",
     "الحبة الثالثة بعد ساعة كاملة من البداية",
     "الفترات بين 3 حبات هي فترتان فقط، أي ساعة واحدة"],
    "ساعة واحدة",
)
TRAP_MONTHS = ex(
    "كم عدد الأشهر التي فيها 28 يومًا؟",
    ["كل شهر ميلادي يحتوي على 28 يومًا على الأقل",
     "فبراير هو الوحيد الذي قد يقتصر على 28 أو 29 يومًا، لكن بقية الأشهر تتجاوز ذلك",
     "إذن جميع الأشهر الاثني عشر فيها 28 يومًا"],
    "12",
)
LOGIC_RACE = ex(
    "في سباق جري، أنهى خالد السباق قبل سعد ولكن بعد فيصل. وأنهى ماجد السباق قبل فيصل. ما الترتيب من الأول إلى الأخير؟",
    ["خالد قبل سعد وبعد فيصل، إذن فيصل > خالد > سعد",
     "ماجد قبل فيصل، إذن ماجد > فيصل",
     "الترتيب: ماجد > فيصل > خالد > سعد"],
    "ماجد، فيصل، خالد، سعد",
)

GENERATORS = [gen_boxes, gen_cups, gen_discount_vat, gen_age, gen_speed, gen_apples, gen_mult]
HANDWRITTEN = [TRAP_PILLS, TRAP_MONTHS, LOGIC_RACE]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-n", type=int, default=300)
    ap.add_argument("--valid-n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=str, default="data")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    train = [rng.choice(GENERATORS)(rng) for _ in range(args.train_n - len(HANDWRITTEN))] + HANDWRITTEN
    rng.shuffle(train)
    valid = [rng.choice(GENERATORS)(rng) for _ in range(args.valid_n)]

    for name, rows in (("train.jsonl", train), ("valid.jsonl", valid)):
        with open(out / name, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {out/'train.jsonl'} ({len(train)}) + {out/'valid.jsonl'} ({len(valid)})")


if __name__ == "__main__":
    main()
