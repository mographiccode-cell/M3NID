import argparse
import json
import random
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

try:
    import torch
    TORCH_IMPORT_ERROR = None
except Exception as e:
    torch = None
    TORCH_IMPORT_ERROR = str(e)

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent

MODEL_CONFIGS = {
    "UNSW-PCNN-AttBiLSTM.pth": {
        "input_dim": 196,
        "labels": [
            "Analysis", "Backdoor", "DoS", "Exploits", "Fuzzers",
            "Generic", "Normal", "Reconnaissance", "Shellcode", "Worms",
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

MODEL_CACHE = {}


def parse_features(raw: str, expected_dim: int):
    values = [v.strip() for v in raw.replace("\n", ",").split(",") if v.strip()]
    if len(values) != expected_dim:
        raise ValueError(f"features count {len(values)} != expected {expected_dim}")
    return [float(v) for v in values]


def risk_level(confidence: float, attack_label: str):
    if attack_label.lower() == "normal":
        return "Low"
    if confidence >= 0.90:
        return "Critical"
    if confidence >= 0.75:
        return "High"
    return "Medium"


def load_model(model_name: str):
    if model_name in MODEL_CACHE:
        return MODEL_CACHE[model_name]
    model_path = ROOT_DIR / model_name
    if not model_path.exists():
        raise FileNotFoundError(f"model not found: {model_path}")
    if torch is None:
        raise RuntimeError(f"PyTorch unavailable: {TORCH_IMPORT_ERROR}")
    model = torch.load(model_path, map_location="cpu")
    model.eval()
    MODEL_CACHE[model_name] = model
    return model


def infer(model_name: str, feature_text: str):
    conf = MODEL_CONFIGS[model_name]
    features = parse_features(feature_text, conf["input_dim"])

    if torch is None:
        idx = random.randint(0, len(conf["labels"]) - 1)
        confidence = round(random.uniform(0.65, 0.98), 4)
        label = conf["labels"][idx]
        return {
            "mode": "mock",
            "warning": f"PyTorch unavailable in current environment: {TORCH_IMPORT_ERROR}",
            "model": model_name,
            "predicted_class": label,
            "class_index": idx,
            "confidence": confidence,
            "risk_level": risk_level(confidence, label),
            "features_used": len(features),
        }

    model = load_model(model_name)
    with torch.no_grad():
        x = torch.tensor([features], dtype=torch.float32)
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        idx = int(torch.argmax(probs, dim=1).item())
        confidence = float(probs[0, idx].item())
    label = conf["labels"][idx] if idx < len(conf["labels"]) else str(idx)
    return {
        "mode": "real",
        "model": model_name,
        "predicted_class": label,
        "class_index": idx,
        "confidence": round(confidence, 4),
        "risk_level": risk_level(confidence, label),
        "features_used": len(features),
    }


class Handler(BaseHTTPRequestHandler):
    def _send_json(self, data, code=HTTPStatus.OK):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path, content_type: str):
        if not path.exists():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        data = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/":
            return self._send_file(BASE_DIR / "templates" / "index.html", "text/html; charset=utf-8")
        if self.path == "/static/style.css":
            return self._send_file(BASE_DIR / "static" / "style.css", "text/css; charset=utf-8")
        if self.path == "/static/app.js":
            return self._send_file(BASE_DIR / "static" / "app.js", "application/javascript; charset=utf-8")
        if self.path == "/api/models":
            payload = {
                "models": [
                    {"name": k, "input_dim": v["input_dim"], "num_labels": len(v["labels"])}
                    for k, v in MODEL_CONFIGS.items()
                ],
                "torch_available": torch is not None,
                "torch_error": TORCH_IMPORT_ERROR,
            }
            return self._send_json(payload)
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self):
        if self.path != "/api/predict":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            model = payload.get("model")
            features = payload.get("features", "")
            if model not in MODEL_CONFIGS:
                return self._send_json({"error": "unknown model"}, HTTPStatus.BAD_REQUEST)
            result = infer(model, features)
            return self._send_json(result)
        except ValueError as e:
            return self._send_json({"error": str(e)}, HTTPStatus.BAD_REQUEST)
        except Exception as e:
            return self._send_json({"error": str(e)}, HTTPStatus.INTERNAL_SERVER_ERROR)


def run_server(host: str, port: int):
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"Web UI running on http://{host}:{port}")
    server.serve_forever()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    run_server(args.host, args.port)
