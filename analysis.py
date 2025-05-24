"""Analysis module for computing layer-wise metrics in language models.

This module provides functions for analyzing the internal behavior of GPT-2 and BERT
models across their layers. It computes various information-theoretic metrics
including entropy, perplexity, attention entropy, and more.

The analysis functions support both regular prediction and sampling-based prediction,
with configurable temperature for controlling prediction confidence.

Typical usage example:
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    results = gpt2_entropy_analysis("The cat sat on the", model, tokenizer)
"""

import torch
import torch.nn.functional as F
from transformers import GPT2Tokenizer, GPT2LMHeadModel, BertTokenizer, BertForMaskedLM
from .utils.metrics import (
    compute_entropy,
    top_k_predictions,
    multiple_token_probability_trajectories,
    compute_perplexity_per_token,
    compute_logit_margin,
    compute_kl_divergence_from_uniform,
    compute_attention_entropy,
    surprisal
)
from transformers import PreTrainedModel, PreTrainedTokenizer
from typing import Dict, Any

def gpt2_entropy_analysis(text, model, tokenizer, temperature=1.0, use_sampling=False):
    """Analyzes GPT-2 model's layer-wise behavior for the given text.

    Computes various metrics across all layers of the model including entropy,
    perplexity, logit margins, KL divergence, and attention entropy. Also generates
    token predictions and probability trajectories.

    Args:
        text (str): Input text to analyze.
        model (GPT2LMHeadModel): Pre-trained GPT-2 model.
        tokenizer (GPT2Tokenizer): Associated GPT-2 tokenizer.
        temperature (float, optional): Softmax temperature for predictions. 
            Higher values produce more uniform distributions. Defaults to 1.0.
        use_sampling (bool, optional): Whether to use sampling for predictions 
            instead of greedy selection. Defaults to False.

    Returns:
        dict: Dictionary containing:
            - layer_entropies (list): Entropy values for each layer
            - avg_layer_entropies (list): Average entropy across tokens per layer
            - layer_perplexities (list): Perplexity values for each layer
            - layer_margins (list): Logit margins for each layer
            - layer_kl_divs (list): KL divergences from uniform for each layer
            - layer_attn_entropies (list): Attention entropy values if available
            - layer_surprisals (list): Surprisal values for each layer
            - predictions (list): Top token predictions with probabilities
            - token_trajectories (dict): Probability trajectories for top tokens

    Note:
        All metrics are computed for the last token in the sequence, as this
        represents the model's next-token prediction task.
    """
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
        hidden_states = outputs.hidden_states
        attentions = outputs.attentions if hasattr(outputs, 'attentions') else None
        final_logits = outputs.logits / temperature

    # Initialize lists for all metrics
    layer_entropies = []
    avg_layer_entropies = []
    layer_perplexities = []
    layer_margins = []
    layer_kl_divs = []
    layer_attn_entropies = []
    layer_surprisals = []

    target_ids = inputs['input_ids']
    
    for idx, layer_output in enumerate(hidden_states):
        # Calculate logits for current layer
        logits = model.lm_head(layer_output) / temperature
        
        # Compute all metrics
        entropy = compute_entropy(logits)
        perplexity = compute_perplexity_per_token(logits)
        margin = compute_logit_margin(logits)
        kl_div = compute_kl_divergence_from_uniform(logits)
        surp = surprisal(logits, target_ids)
        
        # Store last token metrics
        layer_entropies.append(entropy[0, -1].item())
        layer_perplexities.append(perplexity[0, -1].item())
        layer_margins.append(margin[0, -1].item())
        layer_kl_divs.append(kl_div[0, -1].item())
        layer_surprisals.append(surp[0, -1].item())
        
        # Safely compute attention entropy if available
        try:
            if attentions and idx < len(attentions):
                attn_weights = attentions[idx][0]  # [num_heads, seq_length, seq_length]
                attn_entropy = compute_attention_entropy(attn_weights[:, -1, :])  # compute for last token
                layer_attn_entropies.append(attn_entropy.mean().item())
            else:
                layer_attn_entropies.append(None)
        except Exception as e:
            print(f"Warning: Could not compute attention entropy for layer {idx}: {e}")
            layer_attn_entropies.append(None)

    # Get predictions for final layer
    if use_sampling:
        probabilities = F.softmax(final_logits[0, -1], dim=-1)
        sampled_token_id = torch.multinomial(probabilities, num_samples=1).item()
        predictions = [(tokenizer.decode([sampled_token_id]), probabilities[sampled_token_id].item())]
    else:
        predictions = top_k_predictions(final_logits[0, -1], tokenizer)

    top_token_ids = [tokenizer.encode(tok, add_special_tokens=False)[0] for tok, _ in predictions[:3]]
    token_probs_by_layer = multiple_token_probability_trajectories(hidden_states, model.lm_head, top_token_ids)

    return {
        'layer_entropies': layer_entropies,
        'avg_layer_entropies': avg_layer_entropies,
        'layer_perplexities': layer_perplexities,
        'layer_margins': layer_margins,
        'layer_kl_divs': layer_kl_divs,
        'layer_attn_entropies': layer_attn_entropies,
        'layer_surprisals': layer_surprisals,
        'predictions': predictions,
        'token_trajectories': token_probs_by_layer
    }

def bert_entropy_analysis(
    text: str,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    mask_index: int,
    temperature: float = 1.0,
    use_sampling: bool = False
) -> Dict[str, Any]:
    """Analyzes BERT model's layer-wise behavior for masked token prediction.

    Similar to gpt2_entropy_analysis but focuses on the masked token position
    instead of the last token. Computes the same set of metrics across all layers.

    Args:
        text (str): Input text containing a [MASK] token.
        model (BertForMaskedLM): Pre-trained BERT model.
        tokenizer (BertTokenizer): Associated BERT tokenizer.
        mask_index (int): Position of the [MASK] token in the sequence.
        temperature (float, optional): Softmax temperature for predictions.
            Higher values produce more uniform distributions. Defaults to 1.0.
        use_sampling (bool, optional): Whether to use sampling for predictions
            instead of greedy selection. Defaults to False.

    Returns:
        dict: Dictionary containing the same metrics as gpt2_entropy_analysis,
            but computed for the masked token position instead of the last token.

    Note:
        All metrics are computed for the masked token position, as this
        represents BERT's masked language modeling task.
    """
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
        hidden_states = outputs.hidden_states
        attentions = outputs.attentions
        final_logits = outputs.logits / temperature

    layer_entropies = []
    avg_layer_entropies = []
    layer_perplexities = []
    layer_margins = []
    layer_kl_divs = []
    layer_attn_entropies = []
    layer_surprisals = []

    target_ids = inputs['input_ids']
    
    for idx, layer_output in enumerate(hidden_states):
        logits = model.cls.predictions.decoder(layer_output) / temperature
        
        # Compute all metrics
        entropy = compute_entropy(logits)
        perplexity = compute_perplexity_per_token(logits)
        margin = compute_logit_margin(logits)
        kl_div = compute_kl_divergence_from_uniform(logits)
        surp = surprisal(logits, target_ids)
        
        # Store masked token metrics
        layer_entropies.append(entropy[0, mask_index].item())
        layer_perplexities.append(perplexity[0, mask_index].item())  # Now should work correctly
        layer_margins.append(margin[0, mask_index].item())
        layer_kl_divs.append(kl_div[0, mask_index].item())
        layer_surprisals.append(surp[0, mask_index].item())
        
        try:
            if attentions and idx < len(attentions):
                attn_weights = attentions[idx][0]  # [num_heads, seq_length, seq_length]
                attn_entropy = compute_attention_entropy(attn_weights[:, mask_index, :])
                layer_attn_entropies.append(attn_entropy.mean().item())
            else:
                layer_attn_entropies.append(None)
        except Exception as e:
            print(f"Warning: Could not compute attention entropy for layer {idx}: {e}")
            layer_attn_entropies.append(None)

    if use_sampling:
        probabilities = F.softmax(final_logits[0, mask_index], dim=-1)
        sampled_token_id = torch.multinomial(probabilities, num_samples=1).item()
        predictions = [(tokenizer.decode([sampled_token_id]), probabilities[sampled_token_id].item())]
    else:
        predictions = top_k_predictions(final_logits[0, mask_index], tokenizer)

    top_token_ids = [tokenizer.encode(tok, add_special_tokens=False)[0] for tok, _ in predictions[:3]]
    token_probs_by_layer = multiple_token_probability_trajectories(hidden_states, model.cls.predictions.decoder, top_token_ids)

    return {
        'layer_entropies': layer_entropies,
        'avg_layer_entropies': avg_layer_entropies,
        'layer_perplexities': layer_perplexities,
        'layer_margins': layer_margins,
        'layer_kl_divs': layer_kl_divs,
        'layer_attn_entropies': layer_attn_entropies,
        'layer_surprisals': layer_surprisals,
        'predictions': predictions,
        'token_trajectories': token_probs_by_layer
    }