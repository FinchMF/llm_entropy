import os
import logging
import numpy as np
from .utils.logging import setup_logging

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

def run_analysis(num_sentences=10, temperature=1.0, use_sampling=False):
    """Run analysis on GPT-2 and BERT models."""
    # Initialize logging
    run_id = setup_logging()
    logger = logging.getLogger('llm_entropy.analysis')
    
    logger.info(f"Starting analysis run {run_id}")
    logger.info(f"Parameters: sentences={num_sentences}, temp={temperature}, sampling={use_sampling}")

    try:
        # Load models
        logger.info("Loading models...")
        gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2", output_hidden_states=True)
        gpt2_model.eval()
        logger.debug("GPT-2 model loaded successfully")

        bert_tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased")
        bert_model = BertForMaskedLM.from_pretrained("google-bert/bert-base-uncased", output_hidden_states=True)
        bert_model.eval()
        logger.debug("BERT model loaded successfully")

        # Generate sentences
        logger.info("Generating example sentences...")
        gpt2_sentences, bert_sentences = example_sentences(num_sentences)
        logger.debug(f"Generated {len(gpt2_sentences)} sentences for each model")

        # Run analyses
        logger.info("Running GPT-2 analysis...")
        gpt2_results = []
        for i, sentence in enumerate(gpt2_sentences):
            logger.debug(f"Analyzing GPT-2 sentence {i+1}/{len(gpt2_sentences)}")
            results = gpt2_entropy_analysis(sentence, gpt2_model, gpt2_tokenizer, 
                                         temperature=temperature, use_sampling=use_sampling)
            gpt2_results.append((sentence, results))

        logger.info("Running BERT analysis...")
        bert_results = []
        for i, sentence in enumerate(bert_sentences):
            logger.debug(f"Analyzing BERT sentence {i+1}/{len(bert_sentences)}")
            mask_index = bert_tokenizer(sentence, return_tensors="pt")["input_ids"].tolist()[0].index(bert_tokenizer.mask_token_id)
            results = bert_entropy_analysis(sentence, mask_index, bert_model, bert_tokenizer,
                                         temperature=temperature, use_sampling=use_sampling)
            bert_results.append((sentence, results))

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

    return {
        'run_id': run_id,
        'gpt2_results': gpt2_results,
        'bert_results': bert_results
    }