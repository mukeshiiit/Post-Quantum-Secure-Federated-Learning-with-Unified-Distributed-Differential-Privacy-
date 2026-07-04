


import numpy as np
from scipy import stats
from typing import Dict, List

class StatisticalAnalyzer:
    """ statistical validation of simulation results."""
    
    def __init__(self, metrics_data: Dict):
        self.data = metrics_data
        
    def compare_algorithms(self, algo_a: str, algo_b: str, last_n_rounds: int = 50) -> Dict:
        """
        Compares two algorithms using T-test and Cohen's d on the final N rounds.
        """
        if algo_a not in self.data or algo_b not in self.data:
            raise ValueError(f"Algorithms {algo_a} or {algo_b} not found in data.")
            
        acc_a = np.array(self.data[algo_a]['accuracy'][-last_n_rounds:])
        acc_b = np.array(self.data[algo_b]['accuracy'][-last_n_rounds:])
        
        # 1. T-Test (Independent samples assumed for different runs, 
        # but here we treat rounds as samples of converged distribution)
        t_stat, p_val = stats.ttest_ind(acc_a, acc_b)
        
        # 2. Cohen's d
        n1, n2 = len(acc_a), len(acc_b)
        s1, s2 = np.std(acc_a, ddof=1), np.std(acc_b, ddof=1)
        pooled_std = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
        cohens_d = (np.mean(acc_a) - np.mean(acc_b)) / pooled_std
        
        return {
            "mean_a": np.mean(acc_a),
            "mean_b": np.mean(acc_b),
            "t_statistic": t_stat,
            "p_value": p_val,
            "cohens_d": cohens_d,
            "significant": p_val < 0.05
        }

    def generate_report(self, comparisons: List[tuple]) -> str:
        """Generates a formatted text report and a summary table for multiple comparisons."""
        lines = []
        lines.append("Statistical Significance Report")
        lines.append("=============================\n")
        
        table_rows = []
        table_header = "| Comparison | Mean Diff | T-Statistic | P-Value | Effect Size (d) | Result |"
        table_sep    = "|------------|-----------|-------------|---------|-----------------|--------|"
        
        for algo_a, algo_b in comparisons:
            try:
                res = self.compare_algorithms(algo_a, algo_b)
                
                # Detailed Section
                lines.append(f"Comparison: {algo_a} vs {algo_b}")
                lines.append(f"------------------------------------------------")
                lines.append(f"Mean Accuracy ({algo_a}): {res['mean_a']:.2f}%")
                lines.append(f"Mean Accuracy ({algo_b}): {res['mean_b']:.2f}%")
                lines.append(f"T-statistic: {res['t_statistic']:.4f}")
                lines.append(f"P-value:     {res['p_value']:.4e} {'(Significant)' if res['significant'] else '(Not Significant)'}")
                
                effect_size = "Large" if abs(res['cohens_d']) > 0.8 else "Small"
                lines.append(f"Cohen's d:   {res['cohens_d']:.4f} ({effect_size} Effect)")
                lines.append("\n")
                
                # Table Row
                mean_diff = res['mean_a'] - res['mean_b']
                significance = "**Significant**" if res['significant'] else "Not Significant"
                row = f"| {algo_a} vs {algo_b} | +{mean_diff:.2f}% | {res['t_statistic']:.2f} | {res['p_value']:.2e} | {res['cohens_d']:.2f} | {significance} |"
                table_rows.append(row)
                
            except Exception as e:
                lines.append(f"Error comparing {algo_a} vs {algo_b}: {str(e)}\n")
        
        # Prepend Table to report
        lines.insert(2, "\nSummary Table")
        lines.insert(3, table_header)
        lines.insert(4, table_sep)
        for i, row in enumerate(table_rows):
            lines.insert(5+i, row)
        lines.insert(5+len(table_rows), "\n")
                
        return "\n".join(lines)
