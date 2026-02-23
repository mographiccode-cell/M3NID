# NIDS Desktop (Linux) - تجربة سريعة بدون إعادة تدريب

هذا تطبيق سطح مكتب بسيط (Tkinter) يعمل على لينكس ويستخدم **مودل جاهز `.pth`** من نفس المستودع مباشرة.

## ما الذي يفعله؟
- تحميل مودل جاهز (UNSW / CIC / NSL).
- إدخال متجه خصائص التدفق كقيم CSV.
- إخراج فوري:
  - `predicted_class`
  - `confidence`
  - `risk_level`

> ملاحظة: هذه تجربة فورية (POC) وليست نظام إنتاجي كامل.

## التشغيل على لينكس
من جذر المشروع:

```bash
python desktop_app/app.py
```

أو اختيار مودل مباشرة:

```bash
python desktop_app/app.py --model UNSW-PCNN-AttBiLSTM.pth
```

## فحص سريع بدون واجهة (Self Test)
```bash
python desktop_app/app.py --self-test --model UNSW-PCNN-AttBiLSTM.pth
```

## المتطلبات
- Python 3.10+
- PyTorch (CPU أو GPU)
- Tkinter (غالباً يأتي مع Python على لينكس، وإن لم يوجد:
  - Ubuntu/Debian: `sudo apt install python3-tk`)

## ملاحظة مهمة
إذا ظهر تنبيه أن PyTorch غير مثبت، قم بتثبيته في جهازك المحلي ثم أعد التشغيل.
