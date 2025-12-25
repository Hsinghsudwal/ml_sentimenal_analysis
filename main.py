import argparse
import os
import subprocess
from pipelines.training_pipeline import MLPipeline


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--serve", action="store_true", help="Serve the API")
    parser.add_argument("--deploy", action="store_true", help="Build and run Docker")
    args = parser.parse_args()

    if args.train:
        path_data = "data/twitter_training.csv"
        pipe = MLPipeline()
        pipe.train_pipeline(path_data)

    elif args.serve:
        # Run FastAPI
        print(f"Starting API")
        uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
    elif args.deploy:
        # Build and run image
        subprocess.run(["docker", "build", "-t", "nlp-api", "."], check=True)
        subprocess.run(["docker", "run", "-p", "8000:8000", "nlp-api"], check=True)


if __name__ == "__main__":
    main()
