"""Entry point for running llm_entropy as a module."""

import argparse
from llm_entropy.main import run_analysis
from llm_entropy.config import load_config

def main():
    parser = argparse.ArgumentParser(description='Analyze language model behavior')
    parser.add_argument('--config', type=str,
                      help='Path to config file (optional)')
    parser.add_argument('--num-sentences', type=int,
                      help='Number of example sentences to analyze')
    parser.add_argument('--temperature', type=float,
                      help='Sampling temperature')
    parser.add_argument('--use-sampling', action='store_true',
                      help='Use sampling instead of greedy decoding')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    run_params = {
        'num_sentences': config.analysis.num_sentences,
        'temperature': config.analysis.temperature,
        'use_sampling': config.analysis.use_sampling,
        'compare_control': config.analysis.compare_control
    }
    
    # Add control parameters if enabled
    if hasattr(config, 'control_tensors'):
        run_params.update({
            'control_vector': config.control_tensors['control_vector'],
            'layer_weights': config.control_tensors['layer_weights'],
            'token_weights': config.control_tensors['token_weights']
        })
    
    run_analysis(**run_params)

if __name__ == "__main__":
    main()
