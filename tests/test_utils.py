from typing import Dict, Any, List
from unittest.mock import Mock
import torch

class MockModelOutput:
    """Custom mock class for transformer model outputs"""
    def __init__(self, tensor_dict: Dict[str, Any]):
        self._tensors = tensor_dict
        for key, value in tensor_dict.items():
            setattr(self, key, value)
    
    def __getitem__(self, key):
        return self._tensors[key]
    
    def __contains__(self, key):
        return key in self._tensors
    
    def keys(self) -> List[str]:
        return list(self._tensors.keys())
    
    def values(self):
        return self._tensors.values()
    
    def items(self):
        return self._tensors.items()

def create_model_outputs(hidden_size: int = 768, vocab_size: int = 50257, seq_len: int = 10):
    """Create mock model outputs with proper tensor shapes"""
    return {
        'logits': torch.randn(1, seq_len, vocab_size),
        'hidden_states': [torch.randn(1, seq_len, hidden_size) for _ in range(12)],
        'attentions': [torch.randn(1, 12, seq_len, seq_len) for _ in range(12)]
    }
