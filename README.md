# LLM Model Specification Generator

Extract structured scientific models from qualitative survey data using LLMs, semantic chunking, and RAG.

## Installation

```bash
git clone <repository-url>
cd llm_model_spec
pip install -r requirements.txt
```

## Quick Start

```bash
# Run with sample data
python main.py --input data/raw/synthetic_workplace_survey.csv --api-key YOUR_OPENROUTER_KEY

# Launch dashboard
python -m streamlit run ui/dashboard.py
```

## Usage

### Command Line

```bash
python main.py --input your_data.csv --llm-model google/gemma-4-31b-it
```

### Python API

```python
from rag_pipeline import RAGModelExtractor

extractor = RAGModelExtractor(openai_api_key="your-openrouter-key")
processed_chunks = extractor.process_and_store_data("data.csv")
results = extractor.extract_models_from_all_chunks()
```

## Output

Extracts structured models in YAML format:

```yaml
Variables:
  - Workload: Number of tasks vs time available
  - Stress: Emotional state due to workload

Relationships:
  - If workload is high, then stress increases

Hypotheses:
  - H1: Workload has a positive effect on stress
```

## Requirements

- Python 3.10+
- OpenRouter API key
- Dependencies in requirements.txt 