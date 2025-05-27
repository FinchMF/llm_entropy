from .conftest import BaseMockDict
from typing import Dict, Any
import torch

# Replace DictLikeMock with BaseMockDict
DictLikeMock = BaseMockDict

def create_mock_outputs(vocab_size: int = 50257, seq_len: int = 10):
    return DictLikeMock({
        'input_ids': torch.tensor([[1, 2, 3, 4, 5]]),
        'attention_mask': torch.ones(1, seq_len).bool(),
        'logits': torch.randn(1, seq_len, vocab_size),
        'hidden_states': tuple(torch.randn(1, seq_len, 768) for _ in range(12)),
        'attentions': tuple(torch.randn(1, 12, seq_len, seq_len) for _ in range(12))
    })

def create_mock_result():
    return DictLikeMock({
        'layer_entropies': [0.5] * 12,
        'predictions': [('token', 0.8)],
        'token_trajectories': {'token': [0.1] * 12},
        'layer_perplexities': [2.0] * 12,
        'layer_margins': [0.3] * 12,
        'layer_kl_divs': [0.1] * 12,
        'layer_attn_entropies': [0.4] * 12,
        'layer_surprisals': [1.0] * 12
    })

def create_analysis_result():
    """Create a mock analysis result with all required fields"""
    return DictLikeMock({
        'layer_entropies': [0.5] * 12,
        'avg_layer_entropies': [0.4] * 12,
        'layer_perplexities': [2.0] * 12,
        'layer_margins': [0.3] * 12,
        'layer_kl_divs': [0.1] * 12,
        'layer_attn_entropies': [0.4] * 12,
        'layer_surprisals': [1.0] * 12,
        'predictions': [('token', 0.8)],
        'token_trajectories': DictLikeMock({'token': [0.1] * 12})
    })

def create_model_forward_output(vocab_size: int = 50257, seq_len: int = 5):
    base_output = {
        'logits': torch.randn(1, seq_len, vocab_size),
        'hidden_states': tuple(torch.randn(1, seq_len, 768) for _ in range(12)),
        'attentions': tuple(torch.randn(1, 12, seq_len, seq_len) for _ in range(12))
    }
    return DictLikeMock(base_output)

# Add alias for backwards compatibility
MockDict = DictLikeMock
