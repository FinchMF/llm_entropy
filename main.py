import os
import logging
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from typing import List, Tuple, Dict, Any
from .utils.logging import setup_logging
from .config import Config

from transformers import GPT2Tokenizer, GPT2LMHeadModel, BertTokenizer, BertForMaskedLM
from llm_entropy.utils.sentence_generator import example_sentences
from llm_entropy.utils.visualizations import plot_entropy_curves, plot_token_prob_trajectory, save_predictions_to_csv
from . import (
    gpt2_entropy_analysis, 
    bert_entropy_analysis,
    plot_entropy_curves, 
    plot_token_prob_trajectory, 
    save_predictions_to_csv,
    example_sentences
)

class ModelManager:
    def __init__(self):
        self.models = {}
        self.tokenizers = {}
    
    def get_model(self, model_name: str):
        if model_name not in self.models:
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    if model_name == "gpt2":
                        self.models[model_name] = GPT2LMHeadModel.from_pretrained(model_name, output_hidden_states=True)
                        self.tokenizers[model_name] = GPT2Tokenizer.from_pretrained(model_name)
                    elif model_name == "bert":
                        self.models[model_name] = BertForMaskedLM.from_pretrained("google-bert/bert-base-uncased", output_hidden_states=True)
                        self.tokenizers[model_name] = BertTokenizer.from_pretrained("google-bert/bert-base-uncased")
                    break
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise RuntimeError(f"Failed to load model {model_name} after {max_retries} attempts") from e
                    logger.warning(f"Attempt {attempt + 1} failed, retrying...")
        
        return self.models[model_name], self.tokenizers[model_name]

def process_batch(
    sentences: List[str], 
    model, 
    tokenizer, 
    analysis_func, 
    analysis_type: str,  # Add analysis type parameter
    batch_size: int = 8, 
    **kwargs
) -> List[Tuple[str, Dict]]:
    """Process a batch of sentences with the given analysis function."""
    results = []
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i + batch_size]
        # For BERT, we need to compute mask_index for each sentence
        if analysis_type == 'bert':
            batch_results = []
            for sent in batch:
                # Get mask index before calling analysis function
                encoded = tokenizer(sent, return_tensors="pt")
                mask_index = encoded["input_ids"].tolist()[0].index(tokenizer.mask_token_id)
                # Pass parameters in correct order
                result = analysis_func(
                    text=sent,
                    model=model,
                    tokenizer=tokenizer,
                    mask_index=mask_index,
                    **kwargs
                )
                batch_results.append(result)
        else:
            batch_results = [
                analysis_func(
                    text=sent,
                    model=model,
                    tokenizer=tokenizer,
                    **kwargs
                ) for sent in batch
            ]
        results.extend(zip(batch, batch_results))
    return results

def run_analysis(num_sentences=10, temperature=1.0, use_sampling=False, batch_size=8, max_workers=4):
    """Run analysis on GPT-2 and BERT models."""
    run_id = setup_logging()
    logger = logging.getLogger('llm_entropy.analysis')
    
    logger.info(f"Starting analysis run {run_id}")
    
    try:
        # Initialize model manager
        model_manager = ModelManager()
        
        # Load models
        logger.info("Loading models...")
        gpt2_model, gpt2_tokenizer = model_manager.get_model("gpt2")
        bert_model, bert_tokenizer = model_manager.get_model("bert")
        
        # Generate sentences
        logger.info("Generating example sentences...")
        gpt2_sentences, bert_sentences = example_sentences(num_sentences)
        
        # Prepare analysis functions with partial application
        gpt2_analyze = partial(gpt2_entropy_analysis, 
                             temperature=temperature, 
                             use_sampling=use_sampling)
        bert_analyze = partial(bert_entropy_analysis,
                             temperature=temperature,
                             use_sampling=use_sampling)
        
        # Run analyses in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Process GPT-2 sentences
            gpt2_futures = [
                executor.submit(
                    process_batch, 
                    gpt2_sentences[i:i + batch_size],
                    gpt2_model,
                    gpt2_tokenizer,
                    gpt2_analyze,
                    'gpt2',  # Add analysis type
                    batch_size
                )
                for i in range(0, len(gpt2_sentences), batch_size)
            ]
            
            # Process BERT sentences
            bert_futures = [
                executor.submit(
                    process_batch,
                    bert_sentences[i:i + batch_size],
                    bert_model,
                    bert_tokenizer,
                    bert_analyze,
                    'bert',  # Add analysis type
                    batch_size
                )
                for i in range(0, len(bert_sentences), batch_size)
            ]
            
            # Collect results
            gpt2_results = []
            bert_results = []
            
            for future in as_completed(gpt2_futures):
                gpt2_results.extend(future.result())
            
            for future in as_completed(bert_futures):
                bert_results.extend(future.result())

        # Generate outputs
        logger.info("Generating plots and saving results...")
        
        # Save plots
        os.makedirs("plots", exist_ok=True)
        
        # Plot all metrics for GPT-2
        metrics = ['layer_entropies', 'layer_perplexities', 'layer_margins', 
                   'layer_kl_divs', 'layer_attn_entropies', 'layer_surprisals']
        
        for metric in metrics:
            plot_entropy_curves(
                [res[metric] for sent, res in gpt2_results],
                f"GPT-2 {metric.replace('_', ' ').title()}", 
                f"plots/gpt2_{metric}.png"
            )
            plot_entropy_curves(
                [res[metric] for sent, res in bert_results],
                f"BERT {metric.replace('_', ' ').title()}", 
                f"plots/bert_{metric}.png"
            )

        # Plot token trajectories
        plot_token_prob_trajectory(
            [np.array(list(res['token_trajectories'].values())).T 
             if isinstance(res['token_trajectories'], dict) 
             else np.array(res['token_trajectories']).T 
             for sent, res in gpt2_results],
            "GPT-2 Token Probabilities",
            "plots/gpt2_token_probs.png"
        )
        plot_token_prob_trajectory(
            [np.array(list(res['token_trajectories'].values())).T 
             if isinstance(res['token_trajectories'], dict) 
             else np.array(res['token_trajectories']).T 
             for sent, res in bert_results],
            "BERT Token Probabilities",
            "plots/bert_token_probs.png"
        )

        # Save predictions to CSV
        os.makedirs("results", exist_ok=True)
        save_predictions_to_csv(
            [(sent, res['predictions']) for sent, res in gpt2_results],
            "results/gpt2_predictions.csv"
        )
        save_predictions_to_csv(
            [(sent, res['predictions']) for sent, res in bert_results],
            "results/bert_predictions.csv"
        )

        logger.info(f"Analysis run {run_id} completed successfully")
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}", exc_info=True)
        raise
    finally:
        # Cleanup
        if 'model_manager' in locals():
            del model_manager.models
            del model_manager.tokenizers

    return {
        'run_id': run_id,
        'gpt2_results': gpt2_results,
        'bert_results': bert_results
    }