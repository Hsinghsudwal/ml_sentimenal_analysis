import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

from pipelines.inference_pipeline import InferencePipeline

ARTIFACT_PATH = "artifact/pipeline.joblib"

app = FastAPI(
    title="Sentiment Analysis API",
    version="1.0.0",
)


# Load model
try:
    inference = InferencePipeline(ARTIFACT_PATH)
except Exception as e:
    inference = None
    print(f"Model load failed: {e}")


class SinglePredictRequest(BaseModel):
    text: str


class BatchPredictRequest(BaseModel):
    texts: List[str]



# Point
@app.post("/predict/single")
def predict_single(req: SinglePredictRequest):
    if inference is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    result = inference.predict(req.text)
    return result[0]


@app.post("/predict/batch")
def predict_batch(req: BatchPredictRequest):
    if inference is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    results = inference.predict(req.texts)
    return {
        "count": len(results),
        "results": results
    }


@app.get("/health")
def health():
    return {"status": "ok"}

# if __name__ == "__main__":
#     uvicorn.run("app:app", host="0.0.0.0", port=9696, reload=True)