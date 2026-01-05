from fastapi import FastAPI, Request
from serverless.lambda_handler import lambda_handler
import json

app = FastAPI(title="Local Serverless Inference Gateway")


def build_event(request: Request, body: dict):
    return {
        "path": request.url.path,
        "httpMethod": request.method,
        "headers": dict(request.headers),
        "body": json.dumps(body),
    }


@app.post("/predict/single")
async def predict_single(request: Request):
    body = await request.json()
    event = build_event(request, body)
    response = lambda_handler(event)
    return json.loads(response["body"])


@app.post("/predict/batch")
async def predict_batch(request: Request):
    body = await request.json()
    event = build_event(request, body)
    response = lambda_handler(event)
    return json.loads(response["body"])
