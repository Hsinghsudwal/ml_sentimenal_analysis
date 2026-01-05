import argparse
import os
import sys
import subprocess
from pipelines.training_pipeline import MLPipeline


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--serve", action="store_true", help="Serve the API")
    parser.add_argument("--serverless", action="store_true", help="Serve serverless")
    args = parser.parse_args()

    if args.train:
        print("Training model...")
        path_data = "data/twitter_training.csv"
        pipe = MLPipeline()
        pipe.train_pipeline(path_data)

    elif args.serve:
        print(f"Starting API")
        subprocess.run([
            sys.executable,
            "-m", "uvicorn","app:app",
            "--host", "0.0.0.0",
            "--port", "9696",
            "--reload"
        ])
        
        
        # uvicorn.run("app:app", host="0.0.0.0", port=9696, reload=True)

    elif args.serverless:
        print("Starting serverless")
        subprocess.run([
            sys.executable,
            "-m", "uvicorn","serverless.gateway:app",
            "--host", "0.0.0.0",
            "--port", "8080",
            "--reload"
        ])

        # Build and run image
        # subprocess.run(["docker", "build", "-f", "docker/Dockerfile.app", "-t", "api", "."], check=True)
        # subprocess.run(["docker", "run", "-p", "9696:9696", "api"], check=True)


if __name__ == "__main__":
    main()
