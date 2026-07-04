

import numpy as np
import pandas as pd
import os
import json
from typing import Dict, List
from src.config import SimulationConfig
from src.algorithms.base import GenericAlgorithm
from src.simulation.scenarios import ScenarioRegistry

class SimulationEngine:
    """Engine to simulate Federated Learning training processes with Error Bars."""
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.scenarios = ScenarioRegistry.get_scenarios()
        
    def _init_algorithms(self) -> Dict:
        """Dynamically initializes algorithms from Scenarios."""
        algos = {}
        for name, scenario in self.scenarios.items():
            algos[name] = GenericAlgorithm(self.config, scenario)
        return algos
        
    def run_simulation(self, active_algos: List[str] = None) -> Dict:
        
        
        rounds = list(range(1, self.config.NUM_ROUNDS + 1))
        metrics_to_track = ['accuracy', 'f1', 'auc', 'mae']
        
        
        # raw_storage[algo][metric][round_idx] = [val_seed1, val_seed2...]
        temp_algos = self._init_algorithms()
        target_names = active_algos if active_algos else list(temp_algos.keys())
        
        raw_storage = {
            algo: {
                metric: [[] for _ in rounds] for metric in metrics_to_track
            } for algo in target_names
        }
        
        print(f"Running simulation with {self.config.NUM_SEEDS} seeds for Error Bars...")
        
       
        supplementary_data = []

        for seed in range(self.config.NUM_SEEDS):
            print(f"  > Executing Seed {seed+1}/{self.config.NUM_SEEDS}...")
            np.random.seed(self.config.seed + seed)
            
            current_algos = self._init_algorithms()
            
            for name in target_names:
                algo = current_algos[name]
                for r_idx, r in enumerate(rounds):
                    metrics = algo.run_round(r)
                    
                   
                    for m in metrics_to_track:
                        raw_storage[name][m][r_idx].append(metrics[m])
                    
                   
                    supplementary_data.append({
                        "Algorithm": name,
                        "Seed": seed + 1,
                        "Round": r,
                        "Accuracy": metrics['accuracy'],
                        "F1_Score": metrics['f1'],
                        "AUC": metrics['auc'],
                        "MAE": metrics['mae']
                    })

        
        self._export_supplementary_data(supplementary_data)
        self._generate_fairness_table(target_names)
        self._generate_overhead_table(target_names)

        
        final_results = {}
        print("Aggregating statistics...")
        for name in target_names:
            final_results[name] = {"rounds": rounds}
            for m in metrics_to_track:
                data_matrix = np.array(raw_storage[name][m]) # Shape: (rounds, seeds)
                mean_curve = np.mean(data_matrix, axis=1).tolist()
                std_curve = np.std(data_matrix, axis=1).tolist()
                
                final_results[name][m] = {
                    "mean": mean_curve,
                    "std": std_curve
                }
                
        return final_results

    def _export_supplementary_data(self, data):
        
        df = pd.DataFrame(data)
        out_dir = "Supplementary_Data_Sheet"
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
            
        path = os.path.join(out_dir, "Simulation_Raw_Metrics.csv")
        df.to_csv(path, index=False)
        print(f"[SUCCESS] Supplementary data saved to {path}")

    def _generate_fairness_table(self, algos):
        
        # Fairness = Accuracy / max(Accuracy) * Privacy_Score
        # This is a synthetic metric for the table
        results = []
        for name in algos:
            scenario = self.scenarios[name]
            results.append({
                "Algorithm": name,
                "Privacy Guarantee": "High" if scenario.privacy_score > 8 else ("Medium" if scenario.privacy_score > 4 else "Low/None"),
                "Privacy Score (1-10)": scenario.privacy_score,
                "Fairness Index": round(scenario.privacy_score * (scenario.target_accuracy/100.0), 2)
            })
        
        df = pd.DataFrame(results)
        df.to_csv(os.path.join(self.config.OUTPUT_DIR, "Privacy_Fairness_Table.csv"), index=False)
        print("[SUCCESS] Privacy Fairness Table generated.")

    def _generate_overhead_table(self, algos):
       
        # Simulating overhead based on comp_efficiency
        # Lower efficiency = Higher overhead (ms per round)
        results = []
        base_overhead = 50 # ms
        
        for name in algos:
            eff = self.scenarios[name].comp_efficiency
            # heuristic: overhead = base + (10 - eff) * 50
            overhead = base_overhead + (10 - eff) * 50
            results.append({
                "Algorithm": name,
                "Comp. Efficiency (1-10)": eff,
                "Est. Overhead (ms/round)": overhead,
                "Comm. Cost (MB/round)": 10.0 + (10 - self.scenarios[name].comm_efficiency) * 2.5
            })
            
        df = pd.DataFrame(results)
        df.to_csv(os.path.join(self.config.OUTPUT_DIR, "Cryptographic_Overhead_Metrics.csv"), index=False)
        print("[SUCCESS] Overhead Metrics generated.")

    def extract_radar_metrics(self) -> Dict:
        """Extracts static metrics from algorithm instances."""
        algos = self._init_algorithms()
        
        labels = ['Test Accuracy', 'Privacy Security', 'Robustness', 'Comm. Efficiency', 'Client Comp. Speed']
        data = {}
        
        for name, algo in algos.items():
            metrics = algo.get_metrics()
            
          
            acc_score = self.scenarios[name].target_accuracy / 10.0
            
            data[name] = [
                round(acc_score, 1),
                metrics['privacy'],
                metrics['robustness'],
                metrics['comm_efficiency'],
                metrics['comp_efficiency']
            ]
            
        return {
            "labels": labels,
            "data": data
        }
