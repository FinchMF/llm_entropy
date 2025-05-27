import pytest
import torch
from collections.abc import MutableMapping
from typing import Dict, Any, List

class BaseMockDict(MutableMapping):
    def __init__(self, data: Dict[str, Any]):
        self._data = data
        # Add all items as attributes
        for key, value in data.items():
            setattr(self, key, value)
    
    def __getitem__(self, key): return self._data[key]
    def __setitem__(self, key, value): 
        self._data[key] = value
        setattr(self, key, value)
    def __delitem__(self, key): 
        del self._data[key]
        delattr(self, key)
    def __iter__(self): return iter(self._data)
    def __len__(self): return len(self._data)
    def __contains__(self, key): return key in self._data
    def keys(self): return list(self._data.keys())
    def values(self): return list(self._data.values())
    def items(self): return list(self._data.items())

class MockModelOutput(BaseMockDict):
    """Mock for transformer model outputs"""
    pass

class MockResultDict(BaseMockDict):
    """Mock for analysis results"""
    pass

@pytest.fixture
def mock_analysis_result():
    """Fixture for mock analysis results"""
    return MockResultDict({
        'layer_entropies': [0.0] * 12,
        'predictions': [('test', 0.5)],
        'token_trajectories': {'token1': [0.1] * 12},
        'layer_perplexities': [1.0] * 12,
        'layer_margins': [0.5] * 12,
        'layer_kl_divs': [0.3] * 12,
        'layer_attn_entropies': [0.2] * 12,
        'layer_surprisals': [1.5] * 12
    })

def create_model_outputs(vocab_size: int, hidden_size: int = 768, seq_len: int = 10):
    """Create mock model outputs with proper tensor shapes"""
    return MockModelOutput({
        'logits': torch.randn(1, seq_len, vocab_size),
        'hidden_states': [torch.randn(1, seq_len, hidden_size) for _ in range(12)],
        'attentions': [torch.randn(1, 12, seq_len, seq_len) for _ in range(12)]
    })
