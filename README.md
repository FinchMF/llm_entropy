# LLM Entropy Analysis

Python package analyzing GPT-2 and BERT internal behavior through information-theoretic metrics across layers.

## Installation

```bash
git clone https://github.com/FinchMF/llm_entropy.git
cd llm_entropy
pip install -e .
pip install -r requirements.txt
```

## Usage

### CLI
```bash
# default
python -m llm_entropy
# override
python -m llm_entropy --num-sentences 20 --temperature 0.8 --use-sampling
# configuration
python -m llm_entropy --config path/to/config.yaml
```

### Config (yaml)
```yaml
analysis:
  num_sentences: 10
  temperature: 1.0
  use_sampling: false

models:
  gpt2:
    name: "gpt2"
    output_hidden_states: true
  bert:
    name: "google-bert/bert-base-uncased"
    output_hidden_states: true
```

### Python API
```python
from llm_entropy import run_analysis

run_analysis(
    num_sentences=20,
    temperature=0.8,
    use_sampling=True
)
```

## Output

### Plots (`plots/`)
- Layer-wise entropy curves
- Token probability trajectories
- Attention entropy patterns

### Results (`results/`)
- Token predictions with probabilities
- Analysis CSVs

## Metrics
- Layer Entropies
- Layer Perplexities
- Logit Margins
- KL Divergence
- Attention Entropies
- Token Surprisal

## Applications

### Research
- Track information flow through model layers
- Compare autoregressive (GPT-2) vs masked (BERT) architectures
- Analyze layer-wise probability trajectories

### Engineering
- Debug model behavior
- Probe layer-specific knowledge
- Guide model distillation
- Optimize training

### Benefits
- Model compression insights
- Better debugging capabilities
- Enhanced interpretability
- Architecture improvement

## Target Users
- ML researchers
- ML engineers
- Model optimization teams
- Interpretability researchers
