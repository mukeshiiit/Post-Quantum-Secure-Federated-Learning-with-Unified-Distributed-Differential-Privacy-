

from abc import ABC, abstractmethod
import numpy as np

class FederatedAlgorithm(ABC):
    """Start here."""
    
    def __init__(self, name: str, config):
        self.name = name
        self.config = config
        
    @abstractmethod
    def run_round(self, round_idx: int) -> dict:
        """
        Executes one round of the protocol.
        Returns a dictionary of metrics: {'accuracy', 'f1', 'auc', 'mae'}
        """
        pass
    
    @abstractmethod
    def get_metrics(self) -> dict:
        """Returns the algorithm's performance metrics."""
        pass
    
    def _simulate_metrics(self, round_idx: int, target_acc: float, rate: float, midpoint: int, noise_level: float) -> dict:
        """
        Simulates metrics.
        """
        # 1. Accuracy (Logistic Growth)
      
        x0 = midpoint
        k = rate
        L = target_acc
        
        base_acc = 10.0 + (L - 10.0) / (1 + np.exp(-k * (round_idx - x0)))
        noise = np.random.normal(0, noise_level)
        final_acc = max(0.0, min(100.0, base_acc + noise))
        
        # 2. F1 Score
        # Simulating F1 as (Accuracy / 100) * 0.95 + noise
        f1_score = (final_acc / 100.0) * 0.98 - np.random.uniform(0, 0.02)
        f1_score = max(0.0, min(1.0, f1_score))
        
        # 3. AUC
        
        auc_score = (final_acc / 100.0) + 0.02 - np.random.uniform(0, 0.01)
        auc_score = max(0.5, min(1.0, auc_score))
        
        # 4. MAE
        mae = (100.0 - final_acc) / 100.0 * 0.5 + np.random.uniform(0, 0.01)
        
        return {
            "accuracy": final_acc,
            "f1": f1_score,
            "auc": auc_score,
            "mae": mae
        }

class GenericAlgorithm(FederatedAlgorithm):
    """
    .
    """
    def __init__(self, config, scenario):
        super().__init__(scenario.name, config)
        self.scenario = scenario
        
    def run_round(self, round_idx: int) -> dict:
        return self._simulate_metrics(
            round_idx,
            target_acc=self.scenario.target_accuracy,
            rate=self.scenario.convergence_rate,
            midpoint=self.scenario.midpoint,
            noise_level=self.scenario.noise_level
        )
        
    def get_metrics(self) -> dict:
        return {
            "privacy": self.scenario.privacy_score,
            "robustness": self.scenario.robustness_score,
            "comm_efficiency": self.scenario.comm_efficiency,
            "comp_efficiency": self.scenario.comp_efficiency
        }
