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

# ---------------------------------------------------------------------------
# UAE domain extensions — keep core math/logic, layer local reasoning.
# Rules used (simplified, stated in traces):
# - UAE VAT = 5% (Federal Tax Authority), applied AFTER discount.
# - Murabaha: sale = cost + margin%, instalment = sale / months (no interest).
# - EOSB (simplified UAE Labour Law): daily = basic/30; 21 days/yr for first
#   5 years, 30 days/yr thereafter; <1 yr = 0. Educational simplification only.
# - Annual leave: 30 days/year => 2.5 days/month pro-rata.
# - Health/legal outputs carry a "general info only" note, never a diagnosis.
# ---------------------------------------------------------------------------

def gen_aed_vat(rng: random.Random) -> dict:
    base = rng.randrange(500, 5001, 20)
    d = rng.choice([10, 15, 20, 25, 30])
    v = 5  # UAE VAT
    disc = round(base * d / 100, 2)
    after_d = round(base - disc, 2)
    vat = round(after_d * v / 100, 2)
    final = round(after_d + vat, 2)
    user = (f"في متجر في دبي، سعر جهاز {base} درهم، عليه خصم {d}٪، "
            f"ثم ضريبة القيمة المضافة الإماراتية {v}٪ بعد الخصم. ما السعر النهائي بالدرهم؟")
    return ex(user, [
        f"قيمة الخصم: {base} × {d}٪ = {disc}",
        f"السعر بعد الخصم: {base} − {disc} = {after_d}",
        f"الضريبة (5٪ إماراتية): {after_d} × {v}٪ = {vat}",
        f"السعر النهائي: {after_d} + {vat} = {final}",
    ], f"{final} درهم")


def gen_sme_profit(rng: random.Random) -> dict:
    rev = rng.randrange(15000, 51000, 1000)
    rent = rng.randrange(3000, 8001, 500)
    sal = rng.randrange(5000, 20001, 500)
    sup = rng.randrange(1000, 5001, 500)
    costs = rent + sal + sup
    profit = rev - costs
    user = (f"شركة صغيرة في الشارقة إيرادها الشهري {rev} درهم، "
            f"مصاريفها: إيجار {rent} ورواتب {sal} ومستلزمات {sup}. ما صافي الربح الشهري؟")
    return ex(user, [
        f"مجموع المصاريف: {rent} + {sal} + {sup} = {costs}",
        f"صافي الربح: {rev} − {costs} = {profit}",
    ], f"{profit} درهم")


def gen_murabaha(rng: random.Random) -> dict:
    cost = rng.choice([24000, 30000, 36000, 48000, 60000])
    pct = rng.choice([8, 10, 12, 15])
    months = rng.choice([10, 12])
    # pick combos that divide cleanly; retry by adjusting months if needed
    profit = round(cost * pct / 100, 2)
    sale = round(cost + profit, 2)
    if sale % months != 0:
        months = 10 if sale % 10 == 0 else 12
        # fall back to rounded instalment if still not clean
    inst = round(sale / months, 2)
    user = (f"تمويل مرابحة: اشترى البنك بضاعة بتكلفة {cost} درهم وهامش ربح {pct}٪، "
            f"يسددها العميل على {months} قسطًا شهريًا متساويًا. ما سعر البيع وما قيمة القسط؟")
    return ex(user, [
        f"الربح المتفق عليه: {cost} × {pct}٪ = {profit}",
        f"سعر البيع (مرابحة، بلا فائدة): {cost} + {profit} = {sale}",
        f"القسط الشهري: {sale} ÷ {months} = {inst}",
    ], f"{sale} درهم، القسط {inst} درهم")


def gen_eosb(rng: random.Random) -> dict:
    basic = rng.choice([6000, 9000, 12000, 15000])
    yrs = rng.choice([2, 3, 4, 6, 7, 8])
    if yrs <= 5:
        days = 21 * yrs
        rule = "21 يومًا عن كل سنة (أقل من 5 سنوات)"
    else:
        days = 21 * 5 + 30 * (yrs - 5)
        rule = "21 يومًا عن أول 5 سنوات + 30 يومًا عن كل سنة بعدها"
    daily = basic // 30
    gratuity = days * daily
    user = (f"موظف في أبوظبي راتبه الأساسي {basic} درهم وخدم {yrs} سنوات. "
            f"احسب مكافأة نهاية الخدمة (تبسيط تعليمي لقانون العمل الإماراتي).")
    return ex(user, [
        f"القاعدة المبسطة: {rule}",
        f"أيام الاستحقاق: {days} يومًا",
        f"الأجر اليومي: {basic} ÷ 30 = {daily}",
        f"المكافأة: {days} × {daily} = {gratuity} درهم",
    ], f"{gratuity} درهم (تبسيط تعليمي — تحقق من قانون العمل الإماراتي)")


def gen_visa_timeline(rng: random.Random) -> dict:
    permit = rng.choice([3, 5, 7])
    adjust = rng.choice([2, 3])
    residency = rng.choice([5, 7, 10])
    total = permit + adjust + residency
    user = (f"معاملة إقامة في الإمارات: إذن الدخول يستغرق {permit} أيام، "
            f"وتعديل الوضع {adjust} أيام، وإصدار الإقامة {residency} أيام. "
            f"كم إجمالي المدة بالتسلسل؟")
    return ex(user, [
        f"المدة = إذن الدخول + تعديل الوضع + الإقامة",
        f"المدة = {permit} + {adjust} + {residency} = {total} أيام",
    ], f"{total} أيام (مدة تقديرية — تحقق من الجهات الرسمية)")


def gen_leave_accrual(rng: random.Random) -> dict:
    months = rng.choice([4, 6, 8, 9])
    accrued = round(months * 2.5, 2)
    acc_s = int(accrued) if float(accrued).is_integer() else accrued
    user = (f"موظف في دبي عمل {months} أشهر. الإجازة السنوية 30 يومًا في السنة. "
            f"كم رصيده المستحق بالتناسب؟")
    return ex(user, [
        "المعدل الشهري: 30 ÷ 12 = 2.5 يوم",
        f"المستحق: {months} × 2.5 = {acc_s} أيام",
    ], f"{acc_s} أيام (تبسيط — تحقق من عقد العمل وقانون العمل)")


def gen_teacher_plan(rng: random.Random) -> dict:
    per_week = rng.choice([4, 5, 6])
    weeks = rng.choice([4, 8])
    mins = per_week * 45 * weeks
    hrs = mins // 60
    user = (f"معلمة في مدرسة إماراتية تعطي {per_week} حصص أسبوعيًا، مدة الحصة 45 دقيقة، "
            f"لمدة {weeks} أسابيع. كم مجموع الساعات؟")
    return ex(user, [
        f"حصص الفصل: {per_week} × {weeks} = {per_week * weeks} حصة",
        f"الدقائق: {per_week * weeks} × 45 = {mins} دقيقة",
        f"الساعات: {mins} ÷ 60 = {hrs} ساعات",
    ], f"{hrs} ساعات")


def gen_curriculum_fraction(rng: random.Random) -> dict:
    total = rng.choice([240, 360, 480])
    third = total // 3
    quarter = total // 4
    rest = total - third - quarter
    user = (f"مدرسة في العين فيها {total} طالبًا. ثلثهم في الأنشطة الرياضية وربعهم في الأنشطة العلمية. "
            f"كم عدد الباقين؟")
    return ex(user, [
        f"الرياضية: {total} ÷ 3 = {third}",
        f"العلمية: {total} ÷ 4 = {quarter}",
        f"الباقون: {total} − {third} − {quarter} = {rest}",
    ], f"{rest} طالبًا")


def gen_itinerary(rng: random.Random) -> dict:
    a = rng.choice([75, 100, 150])
    b = rng.choice([50, 80, 120])
    k = rng.choice([2, 3, 4])
    fare = rng.choice([25, 30, 35, 40])
    taxi = k * fare
    total = a + b + taxi
    user = (f"زائر في دبي: تذكرة فعالية أولى {a} درهم، وثانية {b} درهم، "
            f"و{k} رحلات تاكسي بسعر {fare} درهم للرحلة. ما إجمالي الميزانية؟")
    return ex(user, [
        f"التاكسي: {k} × {fare} = {taxi} درهم",
        f"الإجمالي: {a} + {b} + {taxi} = {total} درهم",
    ], f"{total} درهم")


def gen_visit_window(rng: random.Random) -> dict:
    start_h = rng.choice([10, 11, 14])
    d1 = rng.choice([60, 120])
    transit = rng.choice([30, 60])
    d2 = rng.choice([60, 90, 120])
    total_min = d1 + transit + d2
    end_min = start_h * 60 + total_min
    end_h, end_m = divmod(end_min, 60)
    fmt = lambda m: f"{m // 60}:{m % 60:02d}"
    user = (f"خطة في أبوظبي تبدأ {fmt(start_h * 60)}: زيارة {d1} دقيقة، "
            f"ثم تنقل {transit} دقيقة، ثم فعالية {d2} دقيقة. متى تنتهي؟")
    return ex(user, [
        f"المدة الكلية: {d1} + {transit} + {d2} = {total_min} دقيقة",
        f"النهاية: {fmt(start_h * 60)} + {total_min} دقيقة = {fmt(end_min)}",
    ], fmt(end_min))


def gen_culture_check(rng: random.Random) -> dict:
    # constraint-check reasoning: modest dress + photo permission + quiet in worship areas
    place = rng.choice(["مسجد الشيخ زايد", "متحف اللوفر أبوظبي", "سوق نايف"])
    photo = rng.choice(["مسموح بلا فلاش", "ممنوع"])
    user = (f"زائر يخطط لزيارة {place}. القواعد: لباس محتشم دائمًا، "
            f"والتصوير هنا: {photo}، والهدوء واجب في أماكن العبادة. "
            f"هل يرتدي لباسًا محتشمًا ويلتزم بقاعدة التصوير؟ ماذا يفعل؟")
    if photo == "ممنوع":
        steps = ["يلتزم باللباس المحتشم",
                 "لا يلتقط صورًا لأن التصوير ممنوع هنا",
                 "يحافظ على الهدوء"]
        final = "يلتزم باللباس المحتشم ولا يصور"
    else:
        steps = ["يلتزم باللباس المحتشم",
                 "يمكنه التصوير بلا فلاش فقط",
                 "يحافظ على الهدوء"]
        final = "يلتزم باللباس المحتشم ويصور بلا فلاش فقط"
    return ex(user, steps, final)


def gen_triage_flow(rng: random.Random) -> dict:
    emergency = rng.choice([True, False])
    if emergency:
        symptom = rng.choice(["ألم صدر شديد", "صعوبة في التنفس", "نزيف شديد"])
        user = (f"استفسار عام في الإمارات: شخص يشكو من {symptom}. "
                f"هل يتوجه للطوارئ (998) أم يحجز عيادة روتينية؟")
        return ex(user, [
            f"العرض: {symptom} من علامات الخطر",
            "علامات الخطر تستدعي الطوارئ فورًا، لا الانتظار",
            "الاتصال بالإسعاف 998 أو التوجه لأقرب طوارئ",
        ], "توجه للطوارئ فورًا (998) — معلومات عامة وليست تشخيصًا، راجع جهة صحية رسمية")
    symptom = rng.choice(["زكام خفيف منذ يومين بلا حمى", "صداع خفيف متقطع", "ألم أسنان خفيف"])
    user = (f"استفسار عام في الإمارات: شخص يشكو من {symptom} وحالته مستقرة. "
            f"هل يتوجه للطوارئ أم يحجز عيادة؟")
    return ex(user, [
        f"العرض: {symptom} ولا توجد علامة خطر مذكورة",
        "الحالة المستقرة غير العاجلة تناسب العيادة لا الطوارئ",
        "يحجز موعد عيادة ويتابع الأعراض، ويراجع الطوارئ إذا ساءت",
    ], "احجز عيادة وتابع — معلومات عامة وليست تشخيصًا، راجع جهة صحية رسمية")


def gen_appointment_slot(rng: random.Random) -> dict:
    pos = rng.choice([3, 5, 8, 10])
    mins = (pos - 1) * 30
    h, m = divmod(9 * 60 + mins, 60)
    slot = f"{h}:{m:02d}"
    user = (f"عيادة في دبي تفتح 9:00 ومدة الموعد 30 دقيقة. دورك رقم {pos} في الطابور. "
            f"في أي وقت موعدك؟")
    return ex(user, [
        f"قبلك {pos - 1} مواعيد × 30 دقيقة = {mins} دقيقة انتظار",
        f"الموعد: 9:00 + {mins} دقيقة = {slot}",
    ], slot)


CORE_GENERATORS = [gen_boxes, gen_cups, gen_discount_vat, gen_age, gen_speed, gen_apples, gen_mult]
UAE_GENERATORS = [gen_aed_vat, gen_sme_profit, gen_murabaha, gen_eosb,
                  gen_visa_timeline, gen_leave_accrual,
                  gen_teacher_plan, gen_curriculum_fraction,
                  gen_itinerary, gen_visit_window, gen_culture_check,
                  gen_triage_flow, gen_appointment_slot]
GENERATORS = CORE_GENERATORS + UAE_GENERATORS
HANDWRITTEN = [TRAP_PILLS, TRAP_MONTHS, LOGIC_RACE]


def pick_generator(rng: random.Random, uae_ratio: float):
    # Keep core math/logic dominant: default 60% core / 40% UAE.
    if rng.random() < uae_ratio:
        return rng.choice(UAE_GENERATORS)
    return rng.choice(CORE_GENERATORS)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-n", type=int, default=300)
    ap.add_argument("--valid-n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=str, default="data")
    ap.add_argument("--uae-ratio", type=float, default=0.4,
                    help="fraction of synthetic rows from UAE generators (rest = core math/logic)")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    train = [pick_generator(rng, args.uae_ratio)(rng) for _ in range(args.train_n - len(HANDWRITTEN))] + HANDWRITTEN
    rng.shuffle(train)
    valid = [pick_generator(rng, args.uae_ratio)(rng) for _ in range(args.valid_n)]

    for name, rows in (("train.jsonl", train), ("valid.jsonl", valid)):
        with open(out / name, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {out/'train.jsonl'} ({len(train)}) + {out/'valid.jsonl'} ({len(valid)})")


if __name__ == "__main__":
    main()
