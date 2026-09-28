# Deployment Guide

This guide provides comprehensive instructions for deploying the LLM Evaluation Platform in various environments.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Local Development Setup](#local-development-setup)
- [Docker Deployment](#docker-deployment)
- [Cloud Deployment](#cloud-deployment)
- [API Server Deployment](#api-server-deployment)
- [Environment Configuration](#environment-configuration)
- [Troubleshooting](#troubleshooting)

## Prerequisites

### Required Software

- **Python 3.11+**: [Download here](https://www.python.org/downloads/)
- **Docker & Docker Compose**: [Install Docker](https://docs.docker.com/get-docker/)
- **Git**: [Install Git](https://git-scm.com/downloads)

### Required API Keys

You'll need API keys for at least one of the following providers:

- **Anthropic**: [Get API Key](https://console.anthropic.com/)
- **OpenAI**: [Get API Key](https://platform.openai.com/api-keys)
- **Google AI**: [Get API Key](https://makersuite.google.com/app/apikey)
- **Cohere**: [Get API Key](https://dashboard.cohere.com/api-keys)

### Optional for Local Models

- **Ollama**: [Install Ollama](https://ollama.ai/) for running local models
- **vLLM**: [Install vLLM](https://github.com/vllm-project/vllm) for production local inference

## Local Development Setup

### 1. Clone the Repository

```bash
git clone <repository-url>
cd llm-harness
```

### 2. Create Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

```bash
cp .env.example .env
```

Edit `.env` and add your API keys:

```env
ANTHROPIC_API_KEY=your_anthropic_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
GOOGLE_API_KEY=your_google_api_key_here
COHERE_API_KEY=your_cohere_api_key_here
LOCAL_API_BASE=http://localhost:8000/v1
```

### 5. Run the Dashboard

```bash
streamlit run dashboard.py
```

The dashboard will be available at `http://localhost:8501`

### 6. Run Evaluation

```bash
python run_evaluation.py --models claude-3-5-sonnet-20241022 gpt-4o
```

## Docker Deployment

### Quick Start with Docker Compose

1. **Configure Environment Variables**

```bash
cp .env.example .env
# Edit .env with your API keys
```

2. **Build and Run**

```bash
docker-compose up -d
```

The dashboard will be available at `http://localhost:8501`

3. **View Logs**

```bash
docker-compose logs -f
```

4. **Stop Services**

```bash
docker-compose down
```

### Manual Docker Build

1. **Build the Image**

```bash
docker build -t llm-evaluation-harness .
```

2. **Run the Container**

```bash
docker run -d \
  -p 8501:8501 \
  --env-file .env \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/results:/app/results \
  -v $(pwd)/prompts:/app/prompts \
  --name llm-harness \
  llm-evaluation-harness
```

### Docker for API Server

1. **Build API Server Image**

```bash
docker build -f Dockerfile.api -t llm-evaluation-api .
```

2. **Run API Server**

```bash
docker run -d \
  -p 8000:8000 \
  --env-file .env \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/results:/app/results \
  --name llm-api \
  llm-evaluation-api
```

The API will be available at `http://localhost:8000`

API documentation: `http://localhost:8000/docs`

## Cloud Deployment

### AWS Deployment

#### Using AWS ECS (Elastic Container Service)

1. **Push to ECR (Elastic Container Registry)**

```bash
# Login to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com

# Build and tag
docker build -t llm-harness .
docker tag llm-harness:latest <account-id>.dkr.ecr.us-east-1.amazonaws.com/llm-harness:latest

# Push
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/llm-harness:latest
```

2. **Create ECS Task Definition**

```json
{
  "family": "llm-harness",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "2048",
  "memory": "4096",
  "containerDefinitions": [
    {
      "name": "llm-harness",
      "image": "<account-id>.dkr.ecr.us-east-1.amazonaws.com/llm-harness:latest",
      "portMappings": [
        {
          "containerPort": 8501,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "ANTHROPIC_API_KEY",
          "value": "your_key_here"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/llm-harness",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

3. **Deploy to ECS**

```bash
aws ecs register-task-definition --cli-input-json file://task-definition.json
aws ecs run-task --cluster llm-harness-cluster --task-definition llm-harness
```

#### Using AWS EC2

1. **Launch EC2 Instance**

- AMI: Ubuntu 22.04 LTS
- Instance Type: t3.medium or larger
- Security Group: Allow ports 8501 (HTTP) and 22 (SSH)

2. **SSH into Instance**

```bash
ssh -i your-key.pem ubuntu@your-instance-ip
```

3. **Install Docker**

```bash
sudo apt-get update
sudo apt-get install -y docker.io docker-compose
sudo usermod -aG docker ubuntu
```

4. **Deploy Application**

```bash
git clone <repository-url>
cd llm-harness
cp .env.example .env
# Edit .env with your API keys
docker-compose up -d
```

### Google Cloud Platform (GCP)

#### Using Cloud Run

1. **Build and Push to GCR**

```bash
# Configure gcloud
gcloud auth configure-docker

# Build
docker build -t gcr.io/your-project-id/llm-harness .

# Push
docker push gcr.io/your-project-id/llm-harness
```

2. **Deploy to Cloud Run**

```bash
gcloud run deploy llm-harness \
  --image gcr.io/your-project-id/llm-harness \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars ANTHROPIC_API_KEY=your_key_here
```

### Microsoft Azure

#### Using Azure Container Instances

1. **Build and Push to ACR**

```bash
# Login to ACR
az acr login --name your-registry

# Build
docker build -t your-registry.azurecr.io/llm-harness .

# Push
docker push your-registry.azurecr.io/llm-harness
```

2. **Deploy to ACI**

```bash
az container create \
  --resource-group llm-harness-rg \
  --name llm-harness \
  --image your-registry.azurecr.io/llm-harness \
  --dns-name-label llm-harness-unique \
  --ports 8501 \
  --environment-variables ANTHROPIC_API_KEY=your_key_here
```

## API Server Deployment

### Running the API Server

1. **Start the API Server**

```bash
python api_server.py
```

The API will be available at `http://localhost:8000`

2. **API Documentation**

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### API Usage Examples

#### Start Evaluation

```bash
curl -X POST "http://localhost:8000/api/v1/evaluations" \
  -H "Content-Type: application/json" \
  -d '{
    "models": ["claude-3-5-sonnet-20241022", "gpt-4o"],
    "categories": ["factual_qa", "reasoning"],
    "bias_check": true
  }'
```

#### Check Status

```bash
curl "http://localhost:8000/api/v1/evaluations/{evaluation_id}"
```

#### Get Results

```bash
curl "http://localhost:8000/api/v1/evaluations/{evaluation_id}/results"
```

#### Export Results

```bash
# JSON
curl "http://localhost:8000/api/v1/evaluations/{evaluation_id}/export/json" -o results.json

# CSV
curl "http://localhost:8000/api/v1/evaluations/{evaluation_id}/export/csv" -o results.csv
```

### Production API Deployment

For production, use a process manager like **systemd** or **gunicorn**:

#### Using Gunicorn

```bash
pip install gunicorn uvicorn

gunicorn api_server:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --timeout 300
```

#### Using Systemd

Create `/etc/systemd/system/llm-api.service`:

```ini
[Unit]
Description=LLM Evaluation API
After=network.target

[Service]
Type=notify
User=www-data
WorkingDirectory=/path/to/llm-harness
Environment="PATH=/path/to/llm-harness/venv/bin"
ExecStart=/path/to/llm-harness/venv/bin/gunicorn api_server:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
Restart=always

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl enable llm-api
sudo systemctl start llm-api
sudo systemctl status llm-api
```

## Environment Configuration

### Environment Variables

| Variable | Required | Description | Example |
|----------|----------|-------------|---------|
| `ANTHROPIC_API_KEY` | No* | Anthropic API key | `sk-ant-...` |
| `OPENAI_API_KEY` | No* | OpenAI API key | `sk-...` |
| `GOOGLE_API_KEY` | No* | Google AI API key | `AIza...` |
| `COHERE_API_KEY` | No* | Cohere API key | `abc123...` |
| `LOCAL_API_BASE` | No | Local model API endpoint | `http://localhost:8000/v1` |

*At least one API key is required.

### Security Best Practices

1. **Never commit `.env` files** to version control
2. **Use secret management** in production (AWS Secrets Manager, Azure Key Vault, etc.)
3. **Rotate API keys** regularly
4. **Use environment-specific** configuration files
5. **Limit API key permissions** to only necessary scopes

### Production Configuration

For production, consider:

```env
# Production settings
PYTHONUNBUFFERED=1
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
STREAMLIT_SERVER_ENABLE_CORS=true
STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=true
STREAMLIT_LOGGER_LEVEL=warning
```

## Troubleshooting

### Common Issues

#### 1. API Key Errors

**Problem**: `Invalid API key` errors

**Solution**:
- Verify API keys in `.env` file
- Check API key hasn't expired
- Ensure API key has necessary permissions

#### 2. Docker Build Failures

**Problem**: Docker build fails with dependency errors

**Solution**:
```bash
# Clear Docker cache
docker system prune -a

# Rebuild without cache
docker build --no-cache -t llm-harness .
```

#### 3. Port Already in Use

**Problem**: Port 8501 or 8000 already in use

**Solution**:
```bash
# Find process using the port
lsof -i :8501  # Mac/Linux
netstat -ano | findstr :8501  # Windows

# Kill the process or use different port
docker run -p 8502:8501 llm-harness
```

#### 4. Memory Issues

**Problem**: Out of memory errors during evaluation

**Solution**:
- Reduce jury size in `src/config.py`
- Use smaller models
- Increase system memory
- Process categories sequentially

#### 5. Import Errors

**Problem**: `ModuleNotFoundError` for packages

**Solution**:
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall

# Or in Docker
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

### Debug Mode

Enable debug logging:

```bash
# Set environment variable
export STREAMLIT_LOGGER_LEVEL=debug

# Or in Python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Health Checks

#### Docker Health Check

```bash
docker ps --format "table {{.Names}}\t{{.Status}}"
```

#### API Health Check

```bash
curl http://localhost:8000/docs
```

#### Dashboard Health Check

```bash
curl http://localhost:8501/_stcore/health
```

## Performance Optimization

### For Large Evaluations

1. **Use Parallel Judge Execution** (default)
2. **Batch Process Categories**
3. **Use SSD Storage** for results
4. **Increase Memory** for concurrent requests
5. **Use Redis** for caching (advanced)

### For Production

1. **Load Balancing**: Use Nginx or AWS ALB
2. **Monitoring**: Set up Prometheus/Grafana
3. **Logging**: Use ELK Stack or CloudWatch
4. **CDN**: For static assets
5. **Database**: PostgreSQL for result storage (advanced)

## Support

For issues and questions:
- Check the [README.md](README.md) for usage documentation
- Review logs in `data/` directory
- Open an issue on GitHub
- Contact support via the repository
