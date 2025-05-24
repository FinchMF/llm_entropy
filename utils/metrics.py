"""Metrics computation utilities for language model analysis.

This module provides a collection of information-theoretic and probability-based
metrics for analyzing language model behavior. These metrics help quantify model
uncertainty, attention patterns, and prediction confidence across model layers.

The metrics fall into several categories:
    - Information Theory: entropy, perplexity, KL divergence
    - Confidence Metrics: logit margins, prediction probabilities
    - Attention Analysis: attention entropy
    - Token-level Metrics: surprisal, probability trajectories

Typical usage example:
    logits = model(inputs).logits
    entropy = compute_entropy(logits)
    perplexity = compute_perplexity_per_token(logits)
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict, Union

def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Compute Shannon entropy of probability distribution from logits.
    
    Args:
        logits: Raw model output logits of shape (..., vocab_size).
    
    Returns:
        Entropy values of same shape as input except for last dimension.
        Lower values indicate more confident predictions.
    
    Note:
        Uses softmax to convert logits to probabilities before computation.
    """
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    return -(probs * log_probs).sum(-1)

def compute_perplexity_per_token(logits: torch.Tensor) -> torch.Tensor:
    """Compute per-token perplexity from logits.
    
    Args:
        logits: Raw model output logits of shape (..., vocab_size).
    
    Returns:
        Perplexity values of same shape as input except for last dimension.
        Lower values indicate better model performance.
    
    Note:
        Perplexity is the exponential of entropy.
    """
    return torch.exp(compute_entropy(logits))

def compute_logit_margin(logits: torch.Tensor) -> torch.Tensor:
    """Compute margin between highest and second highest logits.
    
    Args:
        logits: Raw model output logits of shape (..., vocab_size).
    
    Returns:
        Margin values of same shape as input except for last dimension.
        Larger margins indicate more confident predictions.
    """
    sorted_logits, _ = torch.sort(logits, dim=-1, descending=True)
    return (sorted_logits[..., 0] - sorted_logits[..., 1])

def compute_kl_divergence_from_uniform(logits: torch.Tensor) -> torch.Tensor:
    """Compute KL divergence between model distribution and uniform distribution.
    
    Args:
        logits: Raw model output logits of shape (..., vocab_size).
    
    Returns:
        KL divergence values of same shape as input except for last dimension.
        Higher values indicate distributions further from uniform.
    
    Note:
        Uses small epsilon (1e-10) to prevent log(0).
    """
    probs = F.softmax(logits, dim=-1)
    uniform = torch.ones_like(probs) / probs.size(-1)
    return (probs * (torch.log(probs + 1e-10) - torch.log(uniform + 1e-10))).sum(-1)

def compute_attention_entropy(attention_weights: torch.Tensor) -> torch.Tensor:
    """Compute entropy of attention weight distributions.
    
    Args:
        attention_weights: Attention weights of shape (..., seq_length).
        Should sum to 1 along last dimension.
    
    Returns:
        Entropy values of same shape as input except for last dimension.
        Lower values indicate more focused attention patterns.
    
    Note:
        Uses small epsilon (1e-10) to prevent log(0).
    """
    return -(attention_weights * torch.log(attention_weights + 1e-10)).sum(-1)

def surprisal(logits: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
    """Compute token-level surprisal (-log probability) for target tokens.
    
    Args:
        logits: Raw model output logits of shape (batch_size, seq_length, vocab_size).
        target_ids: Target token IDs of shape (batch_size, seq_length).
    
    Returns:
        Surprisal values of shape (batch_size, seq_length).
        Higher values indicate more surprising (less expected) tokens.
    """
    log_probs = F.log_softmax(logits, dim=-1)
    return -torch.gather(log_probs, -1, target_ids.unsqueeze(-1)).squeeze(-1)

def top_k_predictions(logits: torch.Tensor, tokenizer, k: int = 5) -> List[Tuple[str, float]]:
    """Get top k token predictions with their probabilities.
    
    Args:
        logits: Raw model output logits of shape (vocab_size,).
        tokenizer: Tokenizer for decoding token IDs to strings.
        k: Number of top predictions to return. Defaults to 5.
    
    Returns:
        List of (token_string, probability) tuples for top k predictions,
        sorted by descending probability.
    """
    probs = F.softmax(logits, dim=-1)
    top_probs, top_indices = torch.topk(probs, k)
    return [(tokenizer.decode([idx.item()]), prob.item()) 
            for idx, prob in zip(top_indices, top_probs)]

def multiple_token_probability_trajectories(
    hidden_states: List[torch.Tensor],
    head: torch.nn.Module,
    token_ids: List[int]
) -> Dict[int, List[float]]:
    """Compute probability trajectories for multiple tokens across layers.
    
    Args:
        hidden_states: List of hidden states from each model layer.
        head: Model head for converting hidden states to logits.
        token_ids: List of token IDs to track.
    
    Returns:
        Dictionary mapping token IDs to lists of probabilities,
        where each list contains the token's probability at each layer.
    
    Note:
        Probabilities are computed for the last position in the sequence.
    """
    trajectories = {token_id: [] for token_id in token_ids}
    
    for layer_output in hidden_states:
        logits = head(layer_output)
        probs = F.softmax(logits, dim=-1)
        
        for token_id in token_ids:
            trajectories[token_id].append(probs[0, -1, token_id].item())
    
    return trajectories
