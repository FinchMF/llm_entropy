import torch
from typing import List, Tuple

def compute_comparative_metrics(standard_results: List[Tuple], control_results: List[Tuple], metrics: List[str]) -> List[Tuple]:
    """Compute comparative metrics between standard and control results.
    
    Args:
        standard_results: List of (sentence, result) tuples from standard analysis
        control_results: List of (sentence, result) tuples from control analysis
        metrics: List of metric names to compare
        
    Returns:
        List of (sentence, comparative_metrics) tuples where comparative_metrics
        contains the differences between control and standard for each metric
    """
    comparative_results = []
    
    for (sent1, std_res), (sent2, ctrl_res) in zip(standard_results, control_results):
        if sent1 == sent2:
            # Store original metrics for reference
            comp_metrics = {
                'standard': {k: std_res[k] for k in metrics if k in std_res},
                'control': {k: ctrl_res[k] for k in metrics if k in ctrl_res}
            }
            
            # Compute differences
            for metric in metrics:
                if metric in std_res and metric in ctrl_res:
                    # Raw difference
                    comp_metrics[f"{metric}_diff"] = [
                        c - s for c, s in zip(ctrl_res[metric], std_res[metric])
                    ]
                    # Percent change
                    comp_metrics[f"{metric}_pct_change"] = [
                        ((c - s) / s) * 100 if s != 0 else 0
                        for c, s in zip(ctrl_res[metric], std_res[metric])
                    ]
                    # Add absolute differences
                    comp_metrics[f"{metric}_abs_diff"] = [
                        abs(c - s) for c, s in zip(ctrl_res[metric], std_res[metric])
                    ]
            
            comparative_results.append((sent1, comp_metrics))
    
    return comparative_results
