## ML Sentiment Analysis Deployment

This guide shows how to build, run, and deploy the ML Sentiment Analysis project using Docker, serverless containers, and Kubernetes (Kind).

# Local Docker Development

Build and run the local API for testing:
```bash
# Build the Docker image
docker build -f docker/Dockerfile.app -t api .

# Run the container
docker run -p 9696:9696 api

# Test
python test_app_single.py
```
# Push Image to Docker Registry
```bash
# Push Image to Docker Registry
docker build -f docker/Dockerfile.app -t hsinghsudwal/sentiment-api:v1 .
docker push hsinghsudwal/sentiment-api:v1

# Pull Pre-built Image
docker pull hsinghsudwal/sentiment-api:v1
docker run -p 9696:9696 hsinghsudwal/sentiment-api:v1
python test_app_single.py
```

# Serverless Deployment

Build and run the serverless container:
```bash
# Build serverless image local
docker build -f docker/Dockerfile.serverless -t serverless .
docker run -p 8080:8080 serverless
python test_serverless_single.py

# Docker pull container
docker build -f docker/Dockerfile.serverless -t <user/app:tag .>
docker push <user/app:tag>
docker pull <user/app:tag>
docker run -p 8080:8080 <user/app:tag>
python test_serverless_single.py or python test_serverless_batch.py
```
# Kubernetes Deployment (Kind)

1. Create Kind cluster
```bash
kind create cluster --name sentiment-cluster
```

2. Build Docker image
```bash
docker build -f docker/Dockerfile.k8s -t app-k8s:v1 .
# 2.1. docker run local dev (test)
docker run -p 8000:8000 app-k8s:v1
```

3. Load image into Kind cluster
```bash
kind load docker-image app-k8s:v1 --name sentiment-cluster
```

# Deploy Application
```bash
# Deploy app
kubectl apply -f k8s/deploy.yaml

# Deploy service
kubectl apply -f k8s/service.yaml

# Deploy Horizontal Pod Autoscaler
kubectl apply -f k8s/hpa.yaml
```

# Verify Deployment
```bash
# Check pods
kubectl get pods

# Check deployments
kubectl get deploy

# Check services
kubectl get service

# Check HPA
kubectl get hpa

# View logs of a pod
kubectl logs -f <pod_name>
```

# Port Forward (Local Access)
```bash
kubectl port-forward service/sentiment-service 30001:80`
```

# Push Image to Docker Registry (k8s)
```bash
# Push Image to Docker Registry
docker build -f docker/Dockerfile.k8s -t hsinghsudwal/sentiment-k8s:v1 .
docker push hsinghsudwal/sentiment-k8s:v1
# Local Development Test (Docker Only)
docker pull hsinghsudwal/sentiment-k8s:v1
docker run -p 8000:8000 hsinghsudwal/sentiment-k8s:v1
python test_k8s_local.py  # local
```


# K8s-Docker Registry
```bash
# Create a kind cluster
kind create cluster --name sentiment-cluster

# Check cluster context
kubectl config current-context

# Verify nodes
kubectl get nodes 

# Load the Docker Hub image into the kind node
kind load docker-image hsinghsudwal/sentiment-k8s:v1 --name sentiment-cluster

# Optional: If you want to test locally, build image for kind:
docker build -f docker/Dockerfile.k8s -t app-k8s:v1 .
kind load docker-image app-k8s:v1 --name sentiment-cluster

# Apply the Deployment
      containers:
        - name: sentiment
          image: hsinghsudwal/sentiment-k8s:v1
          imagePullPolicy: Always
          ports:
            - containerPort: 8000

kubectl apply -f k8s/deploy_image.yaml
kubectl get deployments
kubectl get pods -w   # Running

# Apply the Service

kubectl apply -f k8s/service.yaml
kubectl get services

kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml

# Check if metrics server is running
kubectl get pods -n kube-system

# Apply the HPA

kubectl apply -f k8s/hpa.yaml
kubectl get hpa

# Port-forward for testing
kubectl port-forward service/sentiment-service 30001:80

python test_k8s_hpa.py

kubectl get hpa sentiment-hpa --watch
kubectl get pods -w

# Clean up cluster
kubectl delete -f k8s/hpa.yaml
kubectl delete -f k8s/service.yaml
kubectl delete -f k8s/deploy_image.yaml

kind delete cluster --name sentiment-cluster
```