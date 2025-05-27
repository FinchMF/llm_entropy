import os
import logging
import warnings
import sys
import numpy as np
import matplotlib.pyplot as plt

# Set environment variables to suppress warnings
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTHONWARNINGS"] = "ignore"

# Configure warnings before any other imports
warnings.filterwarnings('ignore', category=Warning)
warnings.filterwarnings('ignore', message=".*NumPy.*")
warnings.filterwarnings('ignore', module='numpy')
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Suppress stderr during imports
_stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')

try:
    import torch
    from transformers import GPT2Tokenizer, GPT2LMHeadModel, BertTokenizer, BertForMaskedLM
    from llm_entropy.utils.sentence_generator import example_sentences
    from .analysis import (
        gpt2_entropy_analysis, 
        bert_entropy_analysis,
        analyze_with_control,
        analyze_bert_with_control
    )
    from .utils.visualizations import (
        plot_entropy_curves, 
        plot_token_prob_trajectory, 
        save_predictions_to_csv
    )
    from .utils.comparative import compute_comparative_metrics
    from .utils.plotting import plot_metric_distribution, plot_comparative_metrics
finally:
    sys.stderr = _stderr

from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from typing import List, Tuple, Dict, Any, Optional
from .utils.logging import setup_logging
from .config import Config
from .utils.validator import validate_file_operation, FileValidator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

@contextmanager
def suppress_warnings():
    """Context manager to suppress specific warnings."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        warnings.filterwarnings("ignore", category=Warning)
        yield

class ModelManager:
    def __init__(self):
        self.models = {}
        self.tokenizers = {}
        self.logger = logging.getLogger('llm_entropy.model_manager')
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def get_model(self, model_name: str):
        if model_name not in self.models:
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    with suppress_warnings():
                        if model_name == "gpt2":
                            model = GPT2LMHeadModel.from_pretrained(
                                model_name, 
                                output_hidden_states=True,
                                torch_dtype=torch.float32,  # Explicitly set dtype
                                trust_remote_code=False     # Disable remote code execution
                            )
                            self.models[model_name] = model.to(self.device)
                            self.tokenizers[model_name] = GPT2Tokenizer.from_pretrained(model_name)
                        elif model_name == "bert":
                            model = BertForMaskedLM.from_pretrained(
                                "google-bert/bert-base-uncased", 
                                output_hidden_states=True,
                                torch_dtype=torch.float32,  # Explicitly set dtype
                                trust_remote_code=False     # Disable remote code execution
                            )
                            self.models[model_name] = model.to(self.device)
                            self.tokenizers[model_name] = BertTokenizer.from_pretrained("google-bert/bert-base-uncased")
                        break
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise RuntimeError(f"Failed to load model {model_name} after {max_retries} attempts") from e
                    self.logger.warning(f"Attempt {attempt + 1} failed: {str(e)}, retrying...")
        
        return self.models[model_name], self.tokenizers[model_name]

def process_batch(
    sentences: List[str], 
    model, 
    tokenizer, 
    analysis_func,
    analysis_type: str,
    batch_size: int = 8,
    control_vector: Optional[torch.Tensor] = None,
    layer_weights: Optional[torch.Tensor] = None,
    token_weights: Optional[torch.Tensor] = None,
    compare_control: bool = False,
    **kwargs
) -> List[Tuple[str, Dict]]:
    """Process a batch of sentences with optional control comparison."""
    logger = logging.getLogger('llm_entropy.analysis')
    results = []
    
    for sent in sentences:
        analysis_kwargs = kwargs.copy()
        if analysis_type == 'bert':
            analysis_kwargs['mask_index'] = tokenizer(sent, return_tensors="pt")["input_ids"].tolist()[0].index(tokenizer.mask_token_id)

        if compare_control and control_vector is not None:
            # Run both analyses for comparison
            logger.info(f"Running comparative analysis for {analysis_type} on: {sent[:50]}...")
            
            # Run base analysis first
            logger.info("Running base analysis...")
            base_result = analysis_func(
                text=sent,
                model=model,
                tokenizer=tokenizer,
                **analysis_kwargs
            )
            
            # Then run control analysis
            logger.info("Running control analysis...")
            control_func = analyze_bert_with_control if analysis_type == 'bert' else analyze_with_control
            control_kwargs = {
                'text': sent,
                'model': model,
                'tokenizer': tokenizer,
                'control_vector': control_vector,
                'layer_weights': layer_weights,
                'token_weights': token_weights,
                **analysis_kwargs
            }
            control_result = control_func(**control_kwargs)
            
            # Combine results
            result = {
                'base_results': base_result,
                'controlled_results': control_result
            }
            logger.info("Comparative analysis complete.")
        else:
            if control_vector is not None:
                logger.info(f"Running control analysis for {analysis_type} on: {sent[:50]}...")
                control_func = analyze_bert_with_control if analysis_type == 'bert' else analyze_with_control
                control_kwargs = {
                    'text': sent,
                    'model': model,
                    'tokenizer': tokenizer,
                    'control_vector': control_vector,
                    'layer_weights': layer_weights,
                    'token_weights': token_weights,
                    **analysis_kwargs
                }
                result = control_func(**control_kwargs)
                logger.info("Control analysis complete.")
            else:
                logger.info(f"Running standard analysis for {analysis_type} on: {sent[:50]}...")
                result = analysis_func(
                    text=sent,
                    model=model,
                    tokenizer=tokenizer,
                    **analysis_kwargs
                )
                logger.info("Standard analysis complete.")
            
        results.append((sent, result))
    
    return results

def run_analysis(
    num_sentences=10, 
    temperature=1.0, 
    use_sampling=False, 
    batch_size=8, 
    max_workers=4,
    control_vector: Optional[torch.Tensor] = None,
    layer_weights: Optional[torch.Tensor] = None,
    token_weights: Optional[torch.Tensor] = None,
    compare_control: bool = False
):
    """Run analysis on GPT-2 and BERT models with optional control vector modification."""
    run_id = setup_logging()
    logger = logging.getLogger('llm_entropy.analysis')
    result_manager = ResultManager(run_id)
    
    # Determine mode early
    if compare_control and control_vector is not None:
        mode = 'comparative'
    elif control_vector is not None:
        mode = 'control'
    else:
        mode = 'standard'
        
    logger.info(f"Starting analysis run {run_id} in {mode} mode")
    
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
        
        # Prepare analysis functions with only base parameters
        base_params = {
            'temperature': temperature,
            'use_sampling': use_sampling
        }
        
        gpt2_analyze = partial(gpt2_entropy_analysis, **base_params)
        bert_analyze = partial(bert_entropy_analysis, **base_params)
        
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
                    'gpt2',
                    batch_size,
                    control_vector,
                    layer_weights,
                    token_weights,
                    compare_control,  # Pass compare_control flag
                    **base_params
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
                    'bert',
                    batch_size,
                    control_vector,
                    layer_weights,
                    token_weights,
                    compare_control,  # Pass compare_control flag
                    **base_params
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
        
        # Process results through ResultManager
        logger.info("Saving analysis results...")
        result_manager.save_results(gpt2_results, "gpt2", mode)
        result_manager.save_results(bert_results, "bert", mode)
        
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
        'mode': mode,
        'output_dir': result_manager.base_dir,
        'gpt2_results': gpt2_results,
        'bert_results': bert_results
    }

class ResultManager:
    """Unified manager for analysis results and storage."""
    def __init__(self, run_id: str):
        self.run_id = run_id
        self.base_dir = f"analysis_outputs/run_{run_id}"
        self.logger = logging.getLogger('llm_entropy.results')
        # Configure matplotlib for non-interactive backend
        plt.switch_backend('agg')
        self.METRIC_NAMES = {
            'layer_entropies': 'Layer Entropy',
            'layer_perplexities': 'Layer Perplexity',
            'layer_margins': 'Layer Margin',
            'layer_kl_divs': 'KL Divergence',
            'layer_attn_entropies': 'Attention Entropy',
            'layer_surprisals': 'Surprisal'
        }
        self._setup_directories()
        
    def _setup_directories(self):
        """Create isolated directory structure for each analysis type."""
        self.logger.info(f"Setting up directory structure in: {self.base_dir}")
        for analysis_type in ['standard', 'control', 'comparative']:
            for subdir in ['data', 'plots', 'predictions']:
                path = os.path.join(self.base_dir, analysis_type, subdir)
                os.makedirs(path, exist_ok=True)
                self.logger.info(f"Created directory: {path}")

    def _get_metrics_from_nested(self, results_dict: Dict) -> Dict:
        """Extract metrics from nested result structure."""
        metrics = {}
        for key, value in results_dict.items():
            if key in self.METRIC_NAMES:
                metrics[key] = value
            elif isinstance(value, dict):
                # Recursively extract from nested dicts
                nested_metrics = self._get_metrics_from_nested(value)
                metrics.update(nested_metrics)
        return metrics

    def _get_predictions_from_results(self, results_dict: Dict) -> Optional[List]:
        """Extract predictions from a results dictionary."""
        if not isinstance(results_dict, dict):
            return None
            
        # Check for direct predictions
        if 'predictions' in results_dict:
            return results_dict['predictions']
            
        # Check nested structures
        for key in ['controlled_results', 'base_results']:
            if key in results_dict and isinstance(results_dict[key], dict):
                nested_preds = results_dict[key].get('predictions')
                if nested_preds:
                    return nested_preds
        return None

    def _extract_control_results(self, results: Dict) -> Optional[Dict]:
        """Extract results from control analysis."""
        if not isinstance(results, dict):
            return None
            
        # For control mode, results are in controlled_results
        if 'controlled_results' in results:
            return results['controlled_results']
            
        # For direct control results (when run in control mode)
        metrics = {}
        for key, value in results.items():
            if key in self.METRIC_NAMES or key == 'predictions':
                metrics[key] = value
        return metrics if metrics else None

    def save_results(self, results: List[Tuple], model_name: str, mode: str):
        """Save results with proper mode handling."""
        self.logger.info(f"\nProcessing {model_name} results in {mode} mode")
        self.logger.info(f"Base directory: {self.base_dir}")
        
        try:
            if mode == 'comparative':
                # Extract standard and control results
                standard_results = []
                control_results = []
                
                for sent, res in results:
                    if isinstance(res, dict):
                        if 'base_results' in res:
                            standard_results.append((sent, res['base_results']))
                        ctrl_res = self._extract_control_results(res)
                        if ctrl_res:
                            control_results.append((sent, ctrl_res))

                # First save individual analyses
                if standard_results:
                    self._save_analysis(standard_results, model_name, 'standard')
                if control_results:
                    self._save_analysis(control_results, model_name, 'control')

                # Then generate and save comparisons
                if standard_results and control_results:
                    self._generate_comparisons(standard_results, control_results, model_name)
            elif mode == 'control':
                # For control mode, process control results directly
                processed_results = []
                for sent, res in results:
                    ctrl_res = self._extract_control_results(res)
                    if ctrl_res:
                        processed_results.append((sent, ctrl_res))
                self._save_analysis(processed_results, model_name, 'control')
                
            else:
                # Standard mode
                self._save_analysis(results, model_name, mode)

        except Exception as e:
            self.logger.error(f"Failed to save {mode} results for {model_name}: {str(e)}")
            raise

    def _generate_comparisons(self, standard_results: List[Tuple], 
                            control_results: List[Tuple], 
                            model_name: str):
        """Generate comparative analysis from standard and control results."""
        base_path = os.path.join(self.base_dir, 'comparative')
        data_dir = os.path.join(base_path, 'data')
        plots_dir = os.path.join(base_path, 'plots')
        
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(plots_dir, exist_ok=True)

        # Extract metrics from both sets of results
        std_metrics = self._get_metrics(standard_results)
        ctrl_metrics = self._get_metrics(control_results)

        # For each metric, compute and store comparisons
        for metric_name in self.METRIC_NAMES:
            if metric_name not in std_metrics or metric_name not in ctrl_metrics:
                continue

            # Get values from both analyses and validate
            std_values = []
            ctrl_values = []
            valid_pairs = []
            
            for (std_sent, std_val), (ctrl_sent, ctrl_val) in zip(
                std_metrics[metric_name], ctrl_metrics[metric_name]
            ):
                # Skip if sentences don't match or values are None
                if std_sent != ctrl_sent or std_val is None or ctrl_val is None:
                    continue
                    
                try:
                    # Convert to numpy arrays and ensure numeric
                    std_array = np.array(std_val, dtype=np.float32)
                    ctrl_array = np.array(ctrl_val, dtype=np.float32)
                    
                    # Skip if arrays contain NaN or inf
                    if np.isnan(std_array).any() or np.isnan(ctrl_array).any() or \
                       np.isinf(std_array).any() or np.isinf(ctrl_array).any():
                        continue
                        
                    std_values.append(std_array)
                    ctrl_values.append(ctrl_array)
                    valid_pairs.append((std_sent, std_array, ctrl_array))
                except (ValueError, TypeError) as e:
                    self.logger.warning(f"Skipping invalid values for {metric_name}: {e}")
                    continue

            # Skip if no valid pairs found
            if not valid_pairs:
                self.logger.warning(f"No valid comparison pairs for {metric_name}")
                continue

            # Save comparative data
            metric_file = os.path.join(data_dir, f"{model_name}_{metric_name}_comparative.csv")
            comparative_data = []
            
            for sent, std_array, ctrl_array in valid_pairs:
                # Compute differences safely
                raw_diff = ctrl_array - std_array
                # Handle division by zero in percent change
                with np.errstate(divide='ignore', invalid='ignore'):
                    pct_change = np.where(
                        std_array != 0,
                        ((ctrl_array - std_array) / std_array) * 100,
                        0
                    )
                abs_diff = np.abs(raw_diff)
                
                comparative_data.extend([
                    (f"{sent} (standard)", std_array),
                    (f"{sent} (control)", ctrl_array),
                    (f"{sent} (difference)", raw_diff),
                    (f"{sent} (percent_change)", pct_change)
                ])

            if comparative_data:
                self._write_metric_data(comparative_data, metric_file)
                self.logger.info(f"Saved comparative data for {metric_name}")

                # Generate comparison plots
                try:
                    std_stack = np.stack(std_values)
                    ctrl_stack = np.stack(ctrl_values)
                    
                    plot_path = os.path.join(plots_dir, f"{model_name}_{metric_name}_comparison.png")
                    plot_comparative_metrics(
                        std_stack,
                        ctrl_stack,
                        f"{model_name} {self.METRIC_NAMES[metric_name]} Comparison",
                        filepath=plot_path
                    )
                    self.logger.info(f"Generated comparison plot for {metric_name}")

                    # Plot differences
                    diff_stack = np.stack([pair[2] - pair[1] for pair in valid_pairs])
                    diff_path = os.path.join(plots_dir, f"{model_name}_{metric_name}_differences.png")
                    plot_metric_distribution(
                        diff_stack,
                        f"{model_name} {self.METRIC_NAMES[metric_name]} Differences",
                        filepath=diff_path
                    )
                    self.logger.info(f"Generated difference plot for {metric_name}")
                except Exception as e:
                    self.logger.error(f"Failed to generate plots for {metric_name}: {str(e)}")
                    continue
        
        # Save labeled predictions
        pred_dir = os.path.join(base_path, 'predictions')
        os.makedirs(pred_dir, exist_ok=True)
        
        # Combine predictions from both analyses
        combined_predictions = []
        for (std_sent, std_res), (ctrl_sent, ctrl_res) in zip(standard_results, control_results):
            if std_sent == ctrl_sent:
                std_preds = std_res.get('predictions', [])
                ctrl_preds = ctrl_res.get('predictions', [])
                if std_preds:
                    combined_predictions.append((f"{std_sent} (standard)", std_preds))
                if ctrl_preds:
                    combined_predictions.append((f"{std_sent} (control)", ctrl_preds))

        if combined_predictions:
            pred_file = os.path.join(pred_dir, f"{model_name}_comparative_predictions.csv")
            success = save_predictions_to_csv(combined_predictions, pred_file)
            if success:
                self.logger.info(f"Saved comparative predictions to {pred_file}")
            else:
                self.logger.warning("Failed to save comparative predictions")

    def _get_metrics(self, results: List[Tuple]) -> Dict:
        """Get metrics from results with better extraction."""
        metrics = {}
        for sent, res in results:
            if isinstance(res, dict):
                # Extract metrics from direct or nested results
                extracted = self._get_metrics_from_nested(res)
                for key, value in extracted.items():
                    if key not in metrics:
                        metrics[key] = []
                    metrics[key].append((sent, value))
        return metrics

    def _save_analysis(self, results: List[Tuple], model_name: str, analysis_type: str):
        """Save analysis results with detailed logging and visualization."""
        base_path = os.path.join(self.base_dir, analysis_type)
        self.logger.info(f"\nSaving {analysis_type} analysis for {model_name}")
        
        try:
            # Extract predictions first
            predictions = self._extract_predictions(results)
            
            # Save predictions if found
            if predictions:
                pred_file = os.path.join(base_path, 'predictions', f"{model_name}_predictions.csv")
                self.logger.info(f"Saving {len(predictions)} predictions to {pred_file}")
                os.makedirs(os.path.dirname(pred_file), exist_ok=True)
                
                success = save_predictions_to_csv(predictions, pred_file)
                if success:
                    self.logger.info(f"Successfully saved predictions to {pred_file}")
                else:
                    self.logger.warning("Failed to save predictions - invalid data format")
            else:
                self.logger.warning("No predictions found to save")

            # Get and save metrics
            metrics = self._get_metrics(results)
            if metrics:
                self.logger.info(f"Found metrics: {list(metrics.keys())}")
                data_dir = os.path.join(base_path, 'data')
                os.makedirs(data_dir, exist_ok=True)
                
                for metric_name, metric_data in metrics.items():
                    if metric_name in self.METRIC_NAMES:
                        metric_file = os.path.join(data_dir, f"{model_name}_{metric_name}.csv")
                        self._write_metric_data(metric_data, metric_file)
                        self.logger.info(f"Saved {metric_name} data to {metric_file}")

            # Generate plots
            plots_dir = os.path.join(base_path, 'plots')
            os.makedirs(plots_dir, exist_ok=True)
            
            for metric_name in self.METRIC_NAMES:
                if metric_name in metrics:
                    metric_data = metrics[metric_name]
                    if metric_data:
                        plot_path = os.path.join(plots_dir, f"{model_name}_{metric_name}.png")
                        self.logger.info(f"Generating plot for {metric_name}")
                        
                        # Extract and validate values for plotting
                        values = [val for _, val in metric_data if val is not None]
                        if values:
                            try:
                                plot_metric_distribution(
                                    values, 
                                    title=f"{model_name} {self.METRIC_NAMES[metric_name]}", 
                                    filepath=plot_path
                                )
                                self.logger.info(f"Successfully generated plot for {metric_name}")
                            except ValueError as e:
                                self.logger.warning(f"Could not generate plot for {metric_name}: {e}")
                            except Exception as e:
                                self.logger.error(f"Error generating plot for {metric_name}: {e}")
                        else:
                            self.logger.warning(f"No valid data for {metric_name} plot")
            
        except Exception as e:
            self.logger.error(f"Failed to save {analysis_type} analysis: {str(e)}")
            raise

    @validate_file_operation("saving metric data")
    def _write_metric_data(self, metric_data: List[Tuple], metric_file: str) -> str:
        """Write metric data to CSV with validation.
        
        Args:
            metric_data: List of (sentence, values) tuples
            metric_file: Output file path
            
        Returns:
            Path to saved file
        """
        FileValidator.ensure_dir_exists(os.path.dirname(metric_file))
        
        try:
            with open(metric_file, 'w') as f:
                # Write header
                f.write("sentence,values\n")
                
                # Write data
                for sent, values in metric_data:
                    if isinstance(values, (list, np.ndarray, torch.Tensor)):
                        # Handle array-like values
                        values_str = ','.join(map(str, values))
                    else:
                        # Handle scalar values
                        values_str = str(values)
                    
                    # Clean sentence for CSV
                    sent = sent.replace('"', '""')  # Escape quotes
                    if ',' in sent:
                        sent = f'"{sent}"'  # Quote if contains comma
                        
                    f.write(f"{sent},{values_str}\n")
                    
            return metric_file
            
        except Exception as e:
            self.logger.error(f"Failed to write metric data: {str(e)}")
            raise

    def _extract_predictions(self, results: List[Tuple]) -> List[Tuple]:
        """Extract predictions with improved nested handling."""
        predictions = []
        
        for sent, res in results:
            # Try direct predictions first
            pred_data = None
            
            if isinstance(res, dict):
                # For comparative results, check both base and control
                if 'base_results' in res or 'controlled_results' in res:
                    # Get base predictions
                    base_results = res.get('base_results', {})
                    if 'predictions' in base_results:
                        predictions.append((f"{sent} (base)", base_results['predictions']))
                        self.logger.debug(f"Found base predictions for: {sent[:30]}...")
                    
                    # Get control predictions
                    ctrl_results = res.get('controlled_results', {})
                    if 'predictions' in ctrl_results:
                        predictions.append((f"{sent} (control)", ctrl_results['predictions']))
                        self.logger.debug(f"Found control predictions for: {sent[:30]}...")
                else:
                    # Try direct predictions
                    if 'predictions' in res:
                        pred_data = res['predictions']
                        predictions.append((sent, pred_data))
                        self.logger.debug(f"Found direct predictions for: {sent[:30]}...")
            
            if not pred_data and len(predictions) == 0:
                self.logger.warning(f"No predictions found for sentence: {sent[:30]}...")
                
        if predictions:
            self.logger.info(f"Extracted {len(predictions)} prediction sets")
        else:
            self.logger.warning(f"No predictions found in any results")
            
        return predictions if predictions else None