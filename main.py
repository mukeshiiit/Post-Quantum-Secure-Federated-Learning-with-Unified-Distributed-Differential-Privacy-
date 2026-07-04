

import sys
import os
import json


sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.config import SimulationConfig
from src.simulation.generator import SimulationEngine
from src.analysis.stats import StatisticalAnalyzer
from src.visualization.plots import Visualizer

def main():
    print("=========================================")
    print("   PQ-FL Comparative Analysis System   ")
    print("   v1.0.2 - Enterprise Simulation Framework")
    print("=========================================\n")
    

    config = SimulationConfig()
    print(f"[INFO] Initialized configuration. Output Dir: {config.OUTPUT_DIR}")
    print(f"[INFO] Simulation Mode: Monte Carlo ({config.NUM_SEEDS} seeds) for Error Bars.")
    

    engine = SimulationEngine(config)
    
   
    print(f"[INFO] Starting Simulation for {config.NUM_ROUNDS} rounds...")
    metrics_data = engine.run_simulation()
    radar_data = engine.extract_radar_metrics()
    
  
    metrics_path = os.path.join(config.OUTPUT_DIR, config.METRICS_FILE)
    radar_path = os.path.join(config.OUTPUT_DIR, config.RADAR_FILE)
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics_data, f, indent=4)
        
    with open(radar_path, 'w') as f:
        json.dump(radar_data, f, indent=4)
        
    print(f"[SUCCESS] Simulation complete. Logs saved to {metrics_path}")
    
   
  
    print("\n[INFO] Performing Statistical Analysis (on Mean Accuracy)...")
    
   
    flat_metrics = {}
    for algo, vals in metrics_data.items():
        flat_metrics[algo] = {'accuracy': vals['accuracy']['mean']}
        
    analyzer = StatisticalAnalyzer(flat_metrics)
    report = analyzer.generate_report([
        ("PQ-FL (Proposed)", "Vanilla FedAvg"),
        ("PQ-FL (Proposed)", "FedProx"),
        ("PQ-FL (Proposed)", "SCAFFOLD"),
        ("PQ-FL (Proposed)", "DP-FedAvg"),
        # Ablation Comparisons
        ("PQ-FL (Proposed)", "FL + Distributed Noise"),
        ("PQ-FL (Proposed)", "FL + Secure Aggregation"),
        ("PQ-FL (Proposed)", "FL + PQ Encryption Only")
    ])
    
    report_path = os.path.join(config.OUTPUT_DIR, config.STATS_REPORT)
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"[SUCCESS] Statistical Report generated at {report_path}")
    print(report)
    
  
    print("\n[INFO] Generating Visualizations...")
    viz = Visualizer(config.OUTPUT_DIR)
    

    viz.plot_all_metrics(metrics_data)
    viz.plot_radar_chart(radar_data)
    viz.plot_tradeoff_scatter(metrics_data, radar_data)
    
    print("\n[COMPLETE] All tasks finished successfully.")

if __name__ == "__main__":
    main()
