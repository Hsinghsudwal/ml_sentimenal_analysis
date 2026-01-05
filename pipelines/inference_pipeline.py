import numpy as np
from datetime import datetime
import joblib
from pathlib import Path


class InferencePipeline:

    def __init__(self, artifact_path):
        artifact_path = Path(artifact_path)

        artifact = joblib.load(artifact_path)

        self.model = artifact["model"]
        self.label_encoder = artifact["label_encoder"]
        self.thresholds = artifact.get("thresholds")

        if not hasattr(self.model, "predict_proba"):
            raise ValueError("Model must support predict_proba()")

        print(f"Artifact loaded from {artifact_path}")

    def predict(self, texts):
        if isinstance(texts, str):
            texts = [texts]

        start_time = datetime.utcnow().timestamp()

        probs = self.model.predict_proba(texts)

        if self.thresholds is not None:
            adjusted = probs - self.thresholds
            pred_idx = np.argmax(adjusted, axis=1)
        else:
            pred_idx = np.argmax(probs, axis=1)

        labels = self.label_encoder.inverse_transform(pred_idx)
        confidences = probs[np.arange(len(probs)), pred_idx]

        latency = datetime.utcnow().timestamp() - start_time

        return [
            {
                "text": text,
                "sentiment": str(label),
                "confidence": float(conf),
                "latency_sec": latency,
            }
            for text, label, conf in zip(texts, labels, confidences)
        ]
