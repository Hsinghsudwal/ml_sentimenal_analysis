# ML Sentiment Analysis 

## Overview
This project implements an end-to-end **Tweet Sentiment Analysis** system using NLP and ML best practices. It covers the full ML lifecycle: data processing, model training, evaluation, API serving, containerization, Kubernetes scaling, and serverless deployment. The system classifies text into positive, negative, neutral or irrelevant, sentiment and is designed to scale from local development to Kubernetes and serverless environments.

---

## Table of Contents
1. [Problem Statement](#problem-statement)  
2. [Installation](#installation)  
3. [Research Notebook](#research-notebook)  
4. [Components](#components)  
5. [Pipeline](#pipeline)
6. [Running the Project](#running-the-project)
7. [Deployment](#deployment)
8. [Proof](#proof)  
9. [Best Practices](#best-practices)
9. [Next Steps](#next-steps)  

---

## Problem Statement
Every day, users post millions of tweets—some positive, some negative, and some neutral. Understanding sentiment is crucial for brands, governments, and researchers to monitor opinions, detect trends, and respond appropriately. Manually analyzing tweets is inefficient and prone to error.

**Objective:**  
Build an intelligent system to automatically classify tweets into positive, negative, neutral or irrelevant sentiment.

**Solution Approach:**
- **Data Understanding & Cleaning:** Remove noise, special characters, URLs, mentions, hashtags, and standardize text.  
- **Feature Engineering:** Transform tweets into numerical representations (TF-IDF, word embeddings, or transformer embeddings).  
- **Model Training:** Train models such as Logistic Regression, Linear SVM with GridSearch.  
- **Evaluation:** Measure performance using metrics like F1-score, precision, recall, and confusion matrices.  
- **Deployment:** Serve the model via a FastAPI for real-time predictions. Deploy using Docker, Kubernetes, and serverless patterns

---
### Tech Stack
* NLP -scikit-learn, NLTK                                           
* API - FastAPI, Uvicorn           
* Containerization - Docker                    
* Serverless - Lambda_handler and gateway
* Orchestration - Kubernetes (Kind)         

---

### Project Structure

```bash
ml_sentiment_analysis/
├── app.py                      # FastAPI application
├── main.py                     # CLI entrypoint
├── pipelines/
│   ├── training_pipeline.py
│   └── inference_pipeline.py
├── src/
│   ├── data_prep/
│   └── train_prep/
├── artifact/                     # Saved model artifacts
├── data/
│   └── twitter_training.csv
├── notebooks/                  # Research & experiments
├── docker/
│   ├── Dockerfile.app
│   ├── Dockerfile.k8s
│   └── Dockerfile.serverless
|
├── serverless/
│   ├── lambda_handler.py
│   ├── gateway.py
│   └── requirements.txt
|
├── k8s/
│   ├── deploy.yaml
│   ├── service.yaml
│   └── hpa.yaml
|
├── test_app_single.py
├── test_serverless_single.py
└── test_k8s_single.py
└── test_k8s_hpa.py
|
├── requirements.txt
└── README.md

```
---
## Installation

1. Clone the repository:
```bash
git clone https://github.com/Hsinghsudwal/ml_sentiment_analysis.git
cd ml_sentiment_analysis
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Install dependencies:

`pip install -r requirements.txt`

## Research Notebook

The notebooks/ directory contains Jupyter notebooks for:
  * Data exploration (EDA)
  * Text preprocessing experiments
  * Feature engineering experiments
  * Model training and evaluation
  * Inference testing

## Components
End-to-End Flow:
```bash
Data Load
   ↓
Text Cleaning & Normalization
   ↓
Feature Engineering (TF-IDF)
   ↓
Model Training & Validation
   ↓
Artifact
   ↓
Inference API
   ↓
Serverless / k8s
```
---
### Data Prep:
* Load raw tweet data
* Text cleaning process & normalization (data cleaning & preprocessing encoder )

### Feature Engineering
* Data train test split
* Tfidf-Vectorizer (text sparse) 

### Models Used:

* Logistic Regression
* Linear SVM
* Hyperparameter tuning via GridSearch

### Evaluation Metrics:
* Precision, Recall, F1-score
* Ensemble on validation, threshold tuning, voting_classifier
* Report
* Artifact

## Pipelines

### Training pipeline:

* Data & process: where it brings the entire sentiment analysis workflow together in one place. It starts by loading raw tweet data, performing cleaning and preprocessing, and preparing the dataset for modeling in a consistent and reproducible way.

* Model & Evaluation: Once the data is ready, the pipeline splits it into training and testing sets, applies label encoding, and trains multiple machine learning models. These models are evaluated, combined into an ensemble, and measured using meaningful performance metrics to ensure reliable predictions.

* Artifact & Reporting: After training, the pipeline packages everything needed for inference—including the trained model, encoders, thresholds, and evaluation results and saves them as artifacts. Which makes the model easy to deploy, and it is logged for traceability. 

### Inference pipeline
* This `inference_pipeline`: ensuring the model is ready for reliable and consistent inference.

* When prediction requests arrive, the pipeline safely handles single or batch inputs, computes class probabilities, and applies calibrated thresholds when available. This allows the system to make balanced predictions rather than relying on raw scores alone.

* Each prediction returns not just the sentiment label, but also confidence and latency information, making the output transparent and observable. This design will track real-world behavior with ease.

## Running the Project

* Training
```bash
python main.py --train
```

* Serve API Locally
```bash
python main.py --serve
```

* Serverless API
```bash
python main.py --serverless
```

* Pull Pre-built Image
```bash
docker pull hsinghsudwal/sentiment-api:v1
docker run -p 9696:9696 hsinghsudwal/sentiment-api:v1

# new terminal
python test_app_single.py or python test_app_batch.py
```

* API Endpoints
Predict Sentiment (Single / Batch)

`POST /predict`

### Request 

```json
POST /predict
{
  "texts": [
    "This is an excellent product!",
    "Terrible experience, would not recommend"
  ]
}
```

### Response

```json
{
  "predictions": [
    {
      "text": "This is an excellent product!",
      "sentiment_label": "positive",
      "confidence": 0.95,
      "latency_sec":"Time in sec",

    },
    {
      "text": "Terrible experience, would not recommend",
      "sentiment_label": "negative",
      "confidence": 0.92,
      "latency_sec":"Time in sec",
    }
  ],
}
```

## Deployment
* The `app`: This FastAPI service exposes the trained sentiment model as a simple web API so you can send text and get instant sentiment results. It safely loads the model and provides health checks to ensure the system is running properly.

* The API supports both single and batch predictions, making it easy to use for quick checks

* Serve API Locally
```bash
python main.py --serve
```
* Testing
```bash
python test_app_single.py or python test_app_batch.py
```

### Docker Local API
```bash
docker build -f docker/Dockerfile.app -t api .
docker run -p 9696:9696 api
```
* Testing
```bash
python test_app_single.py or python test_app_batch.py
```

### Serverless Image
```bash
docker build -f docker/Dockerfile.serverless -t serverless .
docker run -p 8080:8080 serverless
```
* Testing
```bash
python test_serverless_single.py or python test_serverless_batch.py
```

### Kubernetes Deployment (Kind)
1. Create Cluster
```bash
kind create cluster --name sentiment-cluster
```

2. Build & Test Image
```bash
docker build -f docker/Dockerfile.k8s -t app-k8s:v1 .
docker run -p 8000:8000 app-k8s:v1
```

* Testing
```bash
python test_k8s_local.py 
```
3. Load Image
```bash
kind load docker-image app-k8s:v1 --name sentiment-cluster
```

4. Deploy to Kubernetes
```bash
kubectl apply -f k8s/deploy.yaml
kubectl get pods
kubectl apply -f k8s/service.yaml
kubectl get service
kubectl apply -f k8s/hpa.yaml
kubectl get hpa
```
* cpu metrics:
```bash
kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml
```

5. Access Service
```bash
kubectl port-forward service/sentiment-service 30001:80
```
6. Load HPA
```bash
python test_k8s_hpa.py
```
* Watch scaling:
```bash
kubectl get hpa sentiment-hpa --watch
```
### K8s-docker registry
* checkout `docker/Docker_README.md`
---

### Horizontal Pod Autoscaling (HPA)

* CPU-based autoscaling
* Automatic scale-up under load
* Safe scale-down strategy
* Production-ready configuration

## Proof
#### Dir PNG/ show png of docker pull image and apply run with port and run test

## Best Practices

* Modular pipeline architecture
* Reproducible experiments
* Model artifact
* Infrastructure-ready deployments
* Separation of training vs inference
* CLI-driven entrypoint
* Scalable services

## Next Steps
* Artifact versioning
* Integrate Deep-Learning and transformer models (BERT)
* Add real-time Twitter API ingestion
* Implement model drift detection
* Enable automated retraining
* CI/CD with GitHub Actions
* Canary & blue-green deployments
* Multilingual sentiment support


## License

MIT License

---
## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request

## License

MIT License

## Additional Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Kubernetes HPA](https://kubernetes.io/docs/tasks/run-application/horizontal-pod-autoscale/)

## Contact

For questions or issues, please open a GitHub issue.

---