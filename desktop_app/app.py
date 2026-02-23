import argparse
import json
import threading
import tkinter as tk
from pathlib import Path
from tkinter import ttk, messagebox

try:
    import torch
    TORCH_IMPORT_ERROR = None
except Exception as e:
    torch = None
    TORCH_IMPORT_ERROR = e



MODEL_CONFIGS = {
    "UNSW-PCNN-AttBiLSTM.pth": {
        "input_dim": 196,
        "labels": [
            "Analysis",
            "Backdoor",
            "DoS",
            "Exploits",
            "Fuzzers",
            "Generic",
            "Normal",
            "Reconnaissance",
            "Shellcode",
            "Worms",
        ],
    },
    "CIC-PCNN-AttBiLSTM-Transformer.pth": {
        "input_dim": 77,
        "labels": [str(i) for i in range(18)],
    },
    "NSL-PCNN-AttBiLSTM-Transformer.pth": {
        "input_dim": 122,
        "labels": ["Dos", "Normal", "Probe", "R2L", "U2R"],
    },
}

ROOT_DIR = Path(__file__).resolve().parents[1]


def parse_feature_vector(raw: str, expected_dim: int):
    if torch is None:
        raise RuntimeError(f"PyTorch غير متوفر: {TORCH_IMPORT_ERROR}")
    values = [v.strip() for v in raw.replace("\n", ",").split(",") if v.strip()]
    if len(values) != expected_dim:
        raise ValueError(f"عدد الخصائص المدخل ({len(values)}) لا يساوي المطلوب ({expected_dim}).")
    try:
        floats = [float(v) for v in values]
    except ValueError as exc:
        raise ValueError("تأكد أن كل القيم أرقام (float/int).") from exc
    return torch.tensor([floats], dtype=torch.float32)


def risk_level(confidence: float, attack_label: str):
    if attack_label.lower() == "normal":
        return "منخفض"
    if confidence >= 0.90:
        return "حرج"
    if confidence >= 0.75:
        return "مرتفع"
    return "متوسط"


class DesktopNIDSApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("NIDS Desktop (Linux)")
        self.root.geometry("980x680")

        self.model_name = tk.StringVar(value="UNSW-PCNN-AttBiLSTM.pth")
        self.status_text = tk.StringVar(value="جاهز.")

        self.model = None
        self.device = torch.device("cpu") if torch is not None else None

        self._build_ui()

    def _build_ui(self):
        top = ttk.Frame(self.root, padding=12)
        top.pack(fill="x")

        ttk.Label(top, text="اختر المودل الجاهز:", font=("Arial", 12, "bold")).pack(side="left")
        model_combo = ttk.Combobox(
            top,
            width=45,
            textvariable=self.model_name,
            values=list(MODEL_CONFIGS.keys()),
            state="readonly",
        )
        model_combo.pack(side="left", padx=10)

        ttk.Button(top, text="تحميل المودل", command=self.load_model).pack(side="left", padx=5)
        ttk.Button(top, text="تعبئة بيانات تجريبية", command=self.fill_random).pack(side="left", padx=5)

        info = ttk.LabelFrame(self.root, text="تعليمات", padding=10)
        info.pack(fill="x", padx=12, pady=8)
        self.instructions = ttk.Label(info, justify="left")
        self.instructions.pack(anchor="w")

        editor_frame = ttk.LabelFrame(self.root, text="خصائص التدفق (CSV)", padding=10)
        editor_frame.pack(fill="both", expand=True, padx=12, pady=8)

        self.feature_text = tk.Text(editor_frame, height=15, wrap="word", font=("Courier", 10))
        self.feature_text.pack(fill="both", expand=True)

        run_frame = ttk.Frame(self.root, padding=12)
        run_frame.pack(fill="x")
        ttk.Button(run_frame, text="تشغيل الفحص", command=self.run_prediction, style="Accent.TButton").pack(side="left")

        result = ttk.LabelFrame(self.root, text="النتيجة", padding=10)
        result.pack(fill="x", padx=12, pady=8)
        self.result_box = tk.Text(result, height=9, font=("Arial", 11))
        self.result_box.pack(fill="x")

        status = ttk.Label(self.root, textvariable=self.status_text, relief="sunken", anchor="w")
        status.pack(fill="x", side="bottom")

        self.refresh_instruction()

    def refresh_instruction(self):
        conf = MODEL_CONFIGS[self.model_name.get()]
        txt = (
            f"- المودل الحالي: {self.model_name.get()}\n"
            f"- عدد الخصائص المطلوبة: {conf['input_dim']}\n"
            "- أدخل القيم كأرقام مفصولة بفواصل.\n"
            "- لا تحتاج إعادة تدريب؛ فقط تحميل المودل وتشغيل الفحص."
        )
        self.instructions.config(text=txt)

    def load_model(self):
        if torch is None:
            messagebox.showerror("PyTorch Missing", f"PyTorch غير مثبت.\n{TORCH_IMPORT_ERROR}")
            return

        model_file = ROOT_DIR / self.model_name.get()
        if not model_file.exists():
            messagebox.showerror("خطأ", f"الملف غير موجود:\n{model_file}")
            return

        self.status_text.set("جارِ تحميل المودل...")

        def _load():
            try:
                model = torch.load(model_file, map_location=self.device)
                model.eval()
                self.model = model
                self.status_text.set(f"تم تحميل: {model_file.name}")
                self.refresh_instruction()
            except Exception as exc:
                self.status_text.set("فشل تحميل المودل")
                messagebox.showerror("Load Error", str(exc))

        threading.Thread(target=_load, daemon=True).start()

    def fill_random(self):
        if torch is None:
            messagebox.showerror("PyTorch Missing", f"PyTorch غير مثبت.\n{TORCH_IMPORT_ERROR}")
            return
        conf = MODEL_CONFIGS[self.model_name.get()]
        vec = torch.rand(conf["input_dim"]).tolist()
        self.feature_text.delete("1.0", tk.END)
        self.feature_text.insert("1.0", ", ".join(f"{v:.6f}" for v in vec))
        self.status_text.set("تم تعبئة متجه تجريبي عشوائي.")
        self.refresh_instruction()

    def run_prediction(self):
        if torch is None:
            messagebox.showerror("PyTorch Missing", f"PyTorch غير مثبت.\n{TORCH_IMPORT_ERROR}")
            return

        if self.model is None:
            messagebox.showwarning("تنبيه", "حمّل المودل أولاً.")
            return

        conf = MODEL_CONFIGS[self.model_name.get()]
        raw = self.feature_text.get("1.0", tk.END).strip()

        try:
            x = parse_feature_vector(raw, conf["input_dim"])  # [1, D]
        except Exception as exc:
            messagebox.showerror("خطأ إدخال", str(exc))
            return

        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)
            pred_idx = int(torch.argmax(probs, dim=1).item())
            conf_score = float(probs[0, pred_idx].item())

        labels = conf["labels"]
        pred_label = labels[pred_idx] if pred_idx < len(labels) else str(pred_idx)
        level = risk_level(conf_score, pred_label)

        output = {
            "model": self.model_name.get(),
            "predicted_class": pred_label,
            "class_index": pred_idx,
            "confidence": round(conf_score, 4),
            "risk_level": level,
            "note": "هذه نتيجة تجربة فورية باستخدام مودل جاهز فقط.",
        }

        self.result_box.delete("1.0", tk.END)
        self.result_box.insert("1.0", json.dumps(output, ensure_ascii=False, indent=2))
        self.status_text.set("اكتمل الفحص بنجاح.")


def self_test(model_name: str):
    if torch is None:
        raise RuntimeError(f"PyTorch غير متوفر: {TORCH_IMPORT_ERROR}")
    conf = MODEL_CONFIGS[model_name]
    model_file = ROOT_DIR / model_name
    model = torch.load(model_file, map_location="cpu")
    model.eval()
    x = torch.rand(1, conf["input_dim"])
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        idx = int(torch.argmax(probs, dim=1).item())
        score = float(probs[0, idx].item())
    label = conf["labels"][idx] if idx < len(conf["labels"]) else str(idx)
    print(json.dumps({"model": model_name, "pred": label, "confidence": round(score, 4)}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true", help="run CLI self-test without GUI")
    parser.add_argument("--model", default="UNSW-PCNN-AttBiLSTM.pth", choices=list(MODEL_CONFIGS.keys()))
    args = parser.parse_args()

    if args.self_test:
        self_test(args.model)
        return

    root = tk.Tk()
    if torch is None:
        messagebox.showwarning("تنبيه", f"PyTorch غير مثبت حالياً.\n{TORCH_IMPORT_ERROR}\n\nثبت PyTorch ثم أعد التشغيل.")
    app = DesktopNIDSApp(root)
    app.model_name.set(args.model)
    app.refresh_instruction()
    root.mainloop()


if __name__ == "__main__":
    main()
