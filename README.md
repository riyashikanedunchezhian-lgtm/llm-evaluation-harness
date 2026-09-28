# LLM Evaluation Platform ⚡

A comprehensive, production-ready evaluation system for LLM outputs using LLM-as-a-Judge methodology with jury-style evaluation, bias mitigation, and detailed metrics tracking.

## 🌟 Features

### Core Evaluation Capabilities
- **LLM-as-a-Judge**: Multi-dimensional scoring with detailed justifications
- **Jury-style Evaluation**: 3-judge aggregation with variance tracking and consensus detection
- **Bias Mitigation**: Position bias detection and automated mitigation
- **Comprehensive Metrics**: Quality, latency, token usage, and cost tracking per evaluation

### Multi-Provider Support
- **Anthropic**: Claude 3.5 Sonnet, Claude 3 Haiku
- **OpenAI**: GPT-4o, GPT-4o Mini
- **Google AI**: Gemini 1.5 Pro, Gemini 1.5 Flash
- **Cohere**: Command R+, Command R
- **Local Models**: Llama 3, Mistral (via OpenAI-compatible APIs)

### Modern User Interface
- **Polished Dashboard**: Inspired by Linear, Sarvam, and OpenAI design aesthetics
- **Real-time Progress Tracking**: Live evaluation progress with detailed status updates
- **Interactive Visualizations**: Scatter plots, radar charts, heatmaps, and more
- **Export Functionality**: CSV and JSON exports for further analysis

### Developer Experience
- **REST API**: FastAPI-based programmatic access with async evaluation
- **Docker Support**: Containerized deployment with Docker Compose
- **Error Handling**: Comprehensive retry logic with exponential backoff
- **Test Set Management**: Built-in UI for managing custom test cases

### Deployment Ready
- **Zero Configuration**: Quick start with startup scripts (Windows/Mac/Linux)
- **Cloud Native**: Support for AWS, GCP, and Azure deployment
- **Production Optimized**: Health checks, logging, and monitoring ready
- **Secure**: Environment-based configuration with secret management support

## 🚀 Quick Start

### Option 1: Using Startup Scripts (Recommended)

**Windows:**
```bash
start.bat
```

**Mac/Linux:**
```bash
chmod +x start.sh
./start.sh
```

The script will:
- Create necessary directories
- Set up virtual environment
- Install dependencies
- Prompt you to configure API keys
- Start the dashboard and/or API server

### Option 2: Manual Setup

1. **Clone and Setup**
```bash
git clone <repository-url>
cd llm-harness
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. **Configure API Keys**
```bash
cp .env.example .env
# Edit .env with your API keys
```

3. **Start Dashboard**
```bash
streamlit run dashboard.py
```

4. **Run Evaluation**
```bash
python run_evaluation.py --models claude-3-5-sonnet-20241022 gpt-4o
```

### Option 3: Docker Deployment

```bash
docker-compose up -d
```

Access the dashboard at `http://localhost:8501`

## 📖 Usage

### Running Evaluations

**Basic evaluation:**
```bash
python run_evaluation.py
```

**Specify custom models:**
```bash
python run_evaluation.py --models claude-3-5-sonnet-20241022 gpt-4o gemini-1.5-pro
```

**Run specific categories:**
```bash
python run_evaluation.py --categories factual_qa reasoning
```

**Enable position bias checking:**
```bash
python run_evaluation.py --bias-check
```

**Export results:**
```bash
python run_evaluation.py --export-csv results.csv --export-excel results.xlsx
```

### Using the Dashboard

Launch the Streamlit dashboard:
```bash
streamlit run dashboard.py
```

The dashboard provides:
- **Overview Tab**: Performance scatter plots and trends
- **Model Comparison**: Bar charts, radar plots, and performance summaries
- **Category Analysis**: Heatmaps and grouped comparisons
- **Detailed Results**: Searchable table with drill-down capabilities
- **Judge Analysis**: Consensus metrics and variance distribution
- **Settings**: Configuration and methodology documentation
- **Test Set Manager**: Add and manage custom test cases

### Using the API

Start the API server:
```bash
python api_server.py
```

API documentation available at `http://localhost:8000/docs`

**Start evaluation:**
```bash
curl -X POST "http://localhost:8000/api/v1/evaluations" \
  -H "Content-Type: application/json" \
  -d '{
    "models": ["claude-3-5-sonnet-20241022", "gpt-4o"],
    "categories": ["factual_qa", "reasoning"]
  }'
```

**Check status:**
```bash
curl "http://localhost:8000/api/v1/evaluations/{evaluation_id}"
```

**Get results:**
```bash
curl "http://localhost:8000/api/v1/evaluations/{evaluation_id}/results"
```

## 🏗️ Architecture

### Components

```
llm-harness/
├── src/
│   ├── config.py           # Model configurations and pricing
│   ├── models.py           # Multi-provider API client with retry logic
│   ├── judge.py            # LLM-as-a-Judge implementation
│   └── harness.py          # Main evaluation orchestration
├── dashboard.py            # Modern Streamlit dashboard
├── api_server.py           # FastAPI REST API
├── run_evaluation.py       # CLI entry point
├── prompts/
│   └── test_set.json       # Test prompts and reference answers
├── data/                   # Results storage
├── results/                # Export files
├── tests/                  # Test suite
├── Dockerfile              # Dashboard container
├── Dockerfile.api          # API server container
├── docker-compose.yml      # Multi-container orchestration
├── start.sh / start.bat    # Startup scripts
└── DEPLOYMENT.md           # Comprehensive deployment guide
```

### Evaluation Pipeline

1. **Test Selection**: Load test cases from configurable test set
2. **Model Invocation**: Call candidate models with test prompts
3. **Jury Evaluation**: 3 independent judges evaluate each response
4. **Aggregation**: Average scores with variance tracking
5. **Bias Detection**: Check for position bias (optional)
6. **Metrics Collection**: Track tokens, latency, and cost
7. **Results Export**: Save to JSON/CSV/Excel formats

## 🔧 Configuration

### Supported Models

**Anthropic:**
- Claude 3.5 Sonnet (high-performance)
- Claude 3 Haiku (fast, cost-effective)

**OpenAI:**
- GPT-4o (flagship model)
- GPT-4o Mini (cost-effective)

**Google AI:**
- Gemini 1.5 Pro (high-performance)
- Gemini 1.5 Flash (fast, cost-effective)

**Cohere:**
- Command R+ (high-performance)
- Command R (cost-effective)

**Local Models:**
- Llama 3 8B (via Ollama/vLLM)
- Mistral 7B (via Ollama/vLLM)

### Adding Custom Models

Edit `src/config.py`:

```python
MODEL_CONFIGS["your-model-id"] = ModelConfig(
    name="Your Model Name",
    provider="anthropic",  # or "openai", "google", "cohere", "local"
    model_id="your-model-id",
    input_price_per_1k=1.0,
    output_price_per_1k=2.0,
    max_tokens=4096,
    temperature=0.7,
    api_base="http://localhost:8000/v1"  # For local models
)
```

## 🌐 Deployment

### Docker Deployment

```bash
docker-compose up -d
```

### Cloud Deployment

See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed instructions on:
- AWS ECS/EC2 deployment
- Google Cloud Run deployment
- Azure Container Instances deployment
- Production configuration
- Security best practices

### API Server Deployment

For production API deployment:

```bash
gunicorn api_server:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000
```

## 📊 Why Jury Evaluation?

### The Problem with Single-Judge LLM Evaluation

Single-judge LLM evaluation is fundamentally unreliable due to:

1. **Position Bias**: LLM judges tend to favor answers presented first
2. **Verbosity Bias**: Longer responses often receive higher scores
3. **Stochastic Variance**: Same judge, same response, different scores
4. **Context Sensitivity**: Scores influenced by subtle prompt variations
5. **Lack of Reproducibility**: Difficult to compare across runs

### What Jury Aggregation Buys You

Jury-style evaluation (3+ independent judges) addresses these issues:

1. **Variance Reduction**: Random noise cancels out via averaging
2. **Outlier Mitigation**: Individual judge outliers have less impact
3. **Confidence Estimation**: Standard deviation provides confidence measure
4. **Bias Detection**: Can detect and quantify position bias
5. **Improved Reliability**: Better correlation with human judgments

## 🧪 Testing

Run the test suite:
```bash
pytest tests/ -v
```

Specific test files:
```bash
pytest tests/test_aggregation.py -v
pytest tests/test_position_bias.py -v
```

## 📚 Documentation

- [DEPLOYMENT.md](DEPLOYMENT.md) - Comprehensive deployment guide
- [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) - Project architecture and methodology
- [QUICKSTART.md](QUICKSTART.md) - Quick start guide
- [API Documentation](http://localhost:8000/docs) - Interactive API docs (when running)

## 🤝 Contributing

This is a production-ready evaluation platform. For improvements, consider:
- Adding more diverse test cases
- Implementing additional bias mitigation strategies
- Adding support for more LLM providers
- Improving the dashboard with more visualizations
- Adding A/B testing capabilities for prompt engineering
- Implementing result storage in databases
- Adding authentication and authorization

## 📄 License

MIT License - feel free to use this for learning and evaluation purposes.

## 🙏 Acknowledgments

This project implements techniques from research on LLM evaluation, including:
- "LLM-as-a-Judge" methodology from various academic papers
- Jury evaluation approaches for reducing variance
- Position bias mitigation strategies in automated evaluation
- Design inspiration from Linear, Sarvam, and OpenAI product pages

## 📞 Support

For issues and questions:
- Check the [DEPLOYMENT.md](DEPLOYMENT.md) for deployment issues
- Review logs in `data/` directory
- Open an issue on GitHub
- Contact support via the repository

---

**Built with ❤️ for professional LLM evaluation**

**⚡ Zero Configuration • Production Ready • Multi-Provider**
