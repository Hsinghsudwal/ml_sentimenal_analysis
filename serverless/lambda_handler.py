# lambda_handler
import json
from pipelines.inference_pipeline import InferencePipeline

ARTIFACT_PATH = "artifact/pipeline.joblib"

pipeline = InferencePipeline(artifact_path=ARTIFACT_PATH)


def lambda_handler(event, context=None):
    """
    API Gateway:
    {
        "path": "/predict/single" | "/predict/batch",
        "httpMethod": "POST",
        "body": "{...json...}"
    }
    """

    try:
        path = event.get("path")
        body = json.loads(event.get("body", "{}"))

        if path == "/predict/single":
            text = body["text"]
            predictions = pipeline.predict(text)

        elif path == "/predict/batch":
            texts = body["texts"]
            predictions = pipeline.predict(texts)

        else:
            return {
                "statusCode": 404,
                "body": json.dumps({"error": "Invalid route"})
            }

        return {
            "statusCode": 200,
            "body": json.dumps({
                "count": len(predictions),
                "results": predictions
            })
        }

    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({
                "error": str(e),
                "type": type(e).__name__
            })
        }
