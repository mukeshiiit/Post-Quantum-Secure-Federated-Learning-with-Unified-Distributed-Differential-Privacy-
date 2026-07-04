

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
from typing import Dict

class Visualizer:
    """Handles all plotting logic."""
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        # Set professional style
        sns.set_theme(style="whitegrid", context="paper", font_scale=1.4)
        
        self.styles = {
            'PQ-FL (Proposed)': {'color': '#d62728', 'linewidth': 2.5, 'linestyle': '-', 'marker': ''}, 
            'Vanilla FedAvg': {'color': '#1f77b4', 'linewidth': 1.5, 'linestyle': '--', 'marker': ''}, 
            'FedProx': {'color': '#2ca02c', 'linewidth': 1.5, 'linestyle': '-.', 'marker': ''}, 
            'SCAFFOLD': {'color': '#9467bd', 'linewidth': 1.5, 'linestyle': ':', 'marker': ''}, 
            'DP-FedAvg': {'color': '#ff7f0e', 'linewidth': 1.5, 'linestyle': '-', 'marker': ''},
            # Ablation Styles
            'FL + Distributed Noise': {'color': '#8c564b', 'linewidth': 1.5, 'linestyle': '--', 'marker': ''},
            'FL + Secure Aggregation': {'color': '#e377c2', 'linewidth': 1.5, 'linestyle': '-.', 'marker': ''},
            'FL + PQ Encryption Only': {'color': '#7f7f7f', 'linewidth': 1.5, 'linestyle': ':', 'marker': ''} 
        }
    
    def _plot_metric_with_error_bars(self, metrics_data: Dict, metric_key: str, 
                                     title: str, ylabel: str, filename: str):
        """Generic helper to plot any metric with Mean ± Std."""
        
        plt.figure(figsize=(10, 6))
        
        for algo, data in metrics_data.items():
            rounds = data['rounds']
            # Access mean and std
            mean_vals = np.array(data[metric_key]['mean'])
            std_vals = np.array(data[metric_key]['std'])
            
            style = self.styles.get(algo, {'color': 'gray'})
            c = style.get('color', 'black')
            ls = style.get('linestyle', '-')
            lw = style.get('linewidth', 1.5)
            
            # Plot Mean Line
            plt.plot(rounds, mean_vals, label=algo, color=c, linestyle=ls, linewidth=lw)
            
            # Fill Standard Deviation
            plt.fill_between(rounds, mean_vals - std_vals, mean_vals + std_vals, color=c, alpha=0.15)
            
        plt.title(title, fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Communication Rounds', fontsize=14, fontweight='bold')
        plt.ylabel(ylabel, fontsize=14, fontweight='bold')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=12, frameon=True, fancybox=True)
        plt.tight_layout()
        
        save_path = os.path.join(self.output_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Plot saved: {save_path}")

    def plot_all_metrics(self, metrics: Dict):
        """Generates line charts for Accuracy, F1, AUC, and MAE."""
        
        self._plot_metric_with_error_bars(
            metrics, 'accuracy', 
            'Convergence Analysis: Accuracy vs Rounds', 'Test Accuracy (%)', 
            'comparison_accuracy.png'
        )
        
        self._plot_metric_with_error_bars(
            metrics, 'f1', 
            'F1-Score Analysis (Mean ± Std)', 'F1 Score', 
            'comparison_f1.png'
        )
        
        self._plot_metric_with_error_bars(
            metrics, 'auc', 
            'Area Under Curve (AUC) vs Rounds', 'AUC Score', 
            'comparison_auc.png'
        )
        
        self._plot_metric_with_error_bars(
            metrics, 'mae', 
            'Mean Absolute Error (MAE) vs Rounds', 'Mean Absolute Error', 
            'comparison_mae.png'
        )

    def plot_radar_chart(self, radar_data: Dict):
        """Generates Holistic Radar Chart."""
        labels = radar_data['labels']
        num_vars = len(labels)
        angle_vals = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angle_vals += angle_vals[:1] 
        
        fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
        plt.xticks(angle_vals[:-1], labels, size=12, fontweight='bold')
        ax.set_rlabel_position(0)
        plt.yticks([2, 4, 6, 8, 10], ["2", "4", "6", "8", "10"], color="grey", size=10)
        plt.ylim(0, 10)
        
        for algo, values in radar_data['data'].items():
            vals = values + values[:1]
            style = self.styles.get(algo, {'color': 'gray'})
            c = style.get('color', 'black')
            
            if algo == 'PQ-FL (Proposed)':
                ax.plot(angle_vals, vals, linewidth=2.5, linestyle='-', label=algo, color=c)
                ax.fill(angle_vals, vals, color=c, alpha=0.25)
            else:
                ax.plot(angle_vals, vals, linewidth=1.5, linestyle='--', label=algo, color=c, alpha=0.6)
                
        plt.title('Holistic Performance Assessment', size=18, fontweight='bold', y=1.1)
        plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1), fontsize=12)
        
        save_path = os.path.join(self.output_dir, 'comparison_radar.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Plot saved: {save_path}")

    def plot_tradeoff_scatter(self, metrics: Dict, radar_data: Dict):
        """Generates Professional Privacy vs Accuracy Bubble Chart."""
        plt.figure(figsize=(11, 7))
        
        x_priv, y_acc, sizes, colors, labels = [], [], [], [], []
        
        radar_map = radar_data['data']
        
        for algo, data in metrics.items():
            if algo not in radar_map: continue
            
            # Use MEAN accuracy from final round
            final_acc = data['accuracy']['mean'][-1]
            priv_score = radar_map[algo][1] 
            comm_effic = radar_map[algo][3] 
            
            x_priv.append(priv_score)
            y_acc.append(final_acc)
            sizes.append(comm_effic * 200) 
            colors.append(self.styles.get(algo, {'color':'black'})['color'])
            labels.append(algo)
            
        plt.scatter(x_priv, y_acc, s=sizes, c=colors, alpha=0.7, edgecolors="black", linewidth=1.5)
        
        plt.axvline(x=5, color='gray', linestyle='--', linewidth=1)
        plt.axhline(y=80, color='gray', linestyle='--', linewidth=1)
        
        plt.text(9.5, 89, 'OPTIMAL ZONE\nHigh Privacy + High Accuracy', 
                 fontsize=11, fontweight='bold', color='darkgreen', ha='center',
                 bbox=dict(facecolor='#e6f5e6', edgecolor='green', boxstyle='round,pad=0.5'))
        
        offsets = {
            'PQ-FL (Proposed)': (-60, 10),
            'Vanilla FedAvg': (0, 15),
            'FedProx': (0, -25),
            'SCAFFOLD': (0, 15),
            'DP-FedAvg': (0, 15)
        }
        
        for i, txt in enumerate(labels):
            off = offsets.get(txt, (0,10))
            plt.annotate(txt, (x_priv[i], y_acc[i]), xytext=off, 
                         textcoords='offset points', ha='center', fontsize=11, fontweight='bold',
                         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8))
            
        plt.title('Algorithm Trade-off Analysis: Efficiency, Privacy, & Accuracy', fontsize=18, fontweight='bold', pad=20)
        plt.xlabel('Privacy Security Score (Theoretical, 0-10)', fontsize=14, fontweight='bold')
        plt.ylabel('Model Accuracy (%)', fontsize=14, fontweight='bold')
        plt.xlim(0, 11)
        plt.ylim(70, 92)
        
        l1 = plt.scatter([],[], s=300, edgecolors='none', color='gray', alpha=0.5)
        l2 = plt.scatter([],[], s=600, edgecolors='none', color='gray', alpha=0.5)
        l3 = plt.scatter([],[], s=1000, edgecolors='none', color='gray', alpha=0.5)
        plt.legend([l1, l2, l3], ["Low Efficiency", "Med Efficiency", "High Efficiency"], title="Comm. Efficiency (Bubble Size)", 
                   scatterpoints=1, frameon=True, labelspacing=1, loc='lower left', fontsize=9)
        
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, 'comparison_tradeoff.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Plot saved: {save_path}")
