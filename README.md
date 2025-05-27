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
# default analysis
python -m llm_entropy
# comparative analysis with control vectors
python -m llm_entropy --config llm_entropy/config/comparative.yaml
# custom comparative analysis
python -m llm_entropy --compare-control --control-vector path/to/vector.pt
```

### Config (yaml)
```yaml
analysis:
  num_sentences: 10
  temperature: 1.0
  use_sampling: false
  compare_control: true  # Enable comparative analysis
  control:
    enabled: true
    control_vectors:
      positive: [0.1, 0.2, 0.3]  # Control direction
      negative: [-0.1, -0.2, -0.3]
    layer_weights: [1.0, 1.2, 1.5]  # Layer-wise scaling
    token_weights: [1.0, 1.2, 1.5]  # Position-based scaling

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
import torch

# Basic usage
run_analysis(
    num_sentences=20,
    temperature=0.8,
    use_sampling=True
)

# With control vectors
control_vector = torch.randn(768)  # Example control vector
layer_weights = torch.ones(12)     # Example layer weights
token_weights = torch.ones(50)     # Example token weights

run_analysis(
    num_sentences=20,
    temperature=0.8,
    use_sampling=True,
    control_vector=control_vector,
    layer_weights=layer_weights,
    token_weights=token_weights
)

# Comparative analysis
control_vector = torch.randn(768)
run_analysis(
    num_sentences=20,
    compare_control=True,  # Enable comparison
    control_vector=control_vector,
    layer_weights=torch.ones(12),
    token_weights=torch.ones(50)
)
```

## Output

The analysis creates a structured output directory for each run:
```
analysis_outputs/
└── run_{timestamp}_{id}/
    ├── standard/              # Base analysis results
    │   ├── data/             # Metric data in CSV format
    │   │   ├── {model}_{metric}.csv
    │   │   └── ...
    │   ├── plots/            # Visualizations
    │   │   ├── {model}_{metric}.png
    │   │   └── ...
    │   └── predictions/      # Model predictions
    │       └── {model}_predictions.csv
    │
    ├── control/              # Control vector analysis
    │   ├── data/
    │   ├── plots/
    │   └── predictions/
    │
    └── comparative/          # Comparative analysis
        ├── data/            
        │   └── {model}_{metric}_comparative.csv
        ├── plots/
        │   ├── {model}_{metric}_comparison.png
        │   └── {model}_{metric}_differences.png
        └── predictions/
            └── {model}_comparative_predictions.csv
```

### Data Files
Each CSV file contains structured data:

1. Standard/Control Metrics (`data/{model}_{metric}.csv`):
```csv
sentence,values
"Input text 1","[v1,v2,...,vn]"  # Layer-wise values
"Input text 2","[v1,v2,...,vn]"
```

2. Comparative Metrics (`comparative/data/{model}_{metric}_comparative.csv`):
```csv
sentence,values
"Input text 1 (standard)","[v1,v2,...,vn]"
"Input text 1 (control)","[v1,v2,...,vn]"
"Input text 1 (difference)","[d1,d2,...,dn]"
"Input text 1 (percent_change)","[p1,p2,...,pn]"
```

3. Predictions (`predictions/{model}_predictions.csv`):
```csv
Sentence,Token,Probability,Analysis
"Input text 1","token1",0.85,"standard"
"Input text 1","token2",0.10,"standard"
"Input text 1","token1",0.75,"control"
```

### Plots

1. Standard/Control Plots:
- Distribution plots showing metric values across layers
- Individual sequence trajectories (light lines)
- Mean trajectory (black line)
- Standard deviation band (gray area)

2. Comparative Plots:
- Side-by-side comparison of standard vs control
- Difference distribution plots
- Layer-wise impact visualization
- Color-coded trajectories for easy comparison

### Metrics Available

For each analysis type (standard, control, comparative):

1. **Layer Entropies**
   - Uncertainty measure per layer
   - Lower values = more confident predictions

2. **Layer Perplexities**
   - Model performance measure
   - Lower values = better predictions

3. **Layer Margins**
   - Confidence measure between top predictions
   - Higher values = more decisive predictions

4. **KL Divergence**
   - Distance from uniform distribution
   - Higher values = more focused predictions

5. **Attention Entropy**
   - Attention pattern spread
   - Lower values = more focused attention

6. **Surprisal**
   - Unexpectedness of predictions
   - Higher values = more surprising tokens

## Applications

### Research
- Track information flow through model layers
- Compare autoregressive (GPT-2) vs masked (BERT) architectures
- Analyze layer-wise probability trajectories
- Study controlled generation through vector manipulation
- Analyze intervention effects on layer behavior

### Engineering
- Debug model behavior
- Probe layer-specific knowledge
- Guide model distillation
- Optimize training
- Test controlled text generation
- Evaluate steering vectors

## Features
- Layer-wise entropy analysis
- Token probability tracking
- Control vector intervention
- Layer-specific weighting
- Token-specific weighting
- Comparative analysis of control effects
- Side-by-side visualization of controlled vs uncontrolled
- Metric difference computation
- Control vector impact assessment

## Target Users
- ML researchers
- ML engineers
- Model optimization teams
- Interpretability researchers
