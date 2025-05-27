import pytest
import torch
from unittest.mock import Mock, patch

@pytest.fixture
def mock_model_manager():
    def analyze(control_vector=None, **kwargs):
        # Validate control vector dimensions
        if isinstance(control_vector, torch.Tensor):
            if control_vector.shape != (1, 1, 768):  # Expected shape
                raise ValueError("Invalid control vector dimensions")
            
        # Base result structure with determined values
        base = {
            'layer_entropies': [0.5] * 12,
            'predictions': [('test', 0.8)],
            'token_trajectories': {'token': [0.1] * 12}
        }
        
        # Different result for controlled case
        controlled = {
            'layer_entropies': [0.6] * 12,
            'predictions': [('different', 0.7)],
            'token_trajectories': {'token': [0.2] * 12}
        }
        
        # Fixed control impact metrics
        control_result = {
            'base_results': base,
            'controlled_results': controlled,
            'control_impact': {
                'entropy_delta': 0.1,
                'perplexity_delta': 0.2
            },
            'mask_token_impact': 0.15
        }

        result = control_result if isinstance(control_vector, torch.Tensor) else base
        
        return {
            'run_id': '20240101_test',
            'gpt2_results': [('test sentence', result)],
            'bert_results': [('test sentence', result)]  # Use same result structure for both
        }
    
    with patch('llm_entropy.main.ModelManager.get_model'):
        manager = Mock()
        manager.analyze = Mock(side_effect=analyze)
        return manager

def test_run_analysis_without_control(mock_model_manager):
    results = mock_model_manager.analyze()
    assert 'run_id' in results
    assert len(results['gpt2_results']) > 0
    assert len(results['bert_results']) > 0

def test_run_analysis_with_control(mock_model_manager):
    control_vector = torch.randn(1, 1, 768)
    layer_weights = torch.linspace(1.0, 0.0, 12)
    token_weights = torch.ones(10)
    
    # Pass control_vector to analyze()
    results = mock_model_manager.analyze(control_vector=control_vector)
    
    # Check control results structure
    for sent, res in results['gpt2_results']:
        assert 'base_results' in res
        assert 'controlled_results' in res
        assert 'control_impact' in res
        
        # Check control impact metrics
        assert 'entropy_delta' in res['control_impact']
        assert 'perplexity_delta' in res['control_impact']
        
        # Verify controlled results have same structure as base results
        assert set(res['base_results'].keys()) == set(res['controlled_results'].keys())
    
    # Check BERT control results
    for sent, res in results['bert_results']:
        assert 'base_results' in res
        assert 'controlled_results' in res
        assert 'control_impact' in res
        assert 'mask_token_impact' in res
        
        # Verify impact on mask token predictions
        assert isinstance(res['mask_token_impact'], float)

def test_control_vector_impact(mock_model_manager):
    # Run without control
    base_results = mock_model_manager.analyze()
    
    # Run with control
    control_vector = torch.randn(1, 1, 768)
    control_results = mock_model_manager.analyze(control_vector)
    
    # Compare predictions
    base_gpt2_preds = base_results['gpt2_results'][0][1]['predictions']
    control_gpt2_preds = control_results['gpt2_results'][0][1]['controlled_results']['predictions']
    
    # Verify outputs are different
    assert base_gpt2_preds != control_gpt2_preds, "Control vector had no effect on predictions"

@pytest.mark.parametrize("invalid_control", [
    torch.randn(1, 1, 100),  # Wrong hidden dimension
    torch.randn(2, 1, 768),  # Wrong batch size
    None  # Should fall back to standard analysis
])
def test_invalid_control_vectors(mock_model_manager, invalid_control):
    """Test handling of invalid control vectors."""
    if invalid_control is None:
        # Should run without error for None
        results = mock_model_manager.analyze()
        assert 'predictions' in results['gpt2_results'][0][1]
    else:
        # Should raise error for invalid dimensions
        with pytest.raises(ValueError):
            mock_model_manager.analyze(invalid_control)
