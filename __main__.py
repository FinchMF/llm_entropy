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
    
    # Command line args override config file
    run_params = {
        'num_sentences': args.num_sentences or config['analysis']['num_sentences'],
        'temperature': args.temperature or config['analysis']['temperature'],
        'use_sampling': args.use_sampling or config['analysis']['use_sampling']
    }
    
    run_analysis(**run_params)

if __name__ == "__main__":
    main()
