

from abc import ABC, abstractmethod
import numpy as np

class FederatedAlgorithm(ABC):
    """Abstract Base Class for all Federated Learning Algorithms."""
    
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
        Simulates all 4 metrics based on the target accuracy curve.
        """
        # 1. Accuracy (Logistic Growth)
        # Base logistic curve: L / (1 + exp(-k(x-x0)))
        x0 = midpoint
        k = rate
        L = target_acc
        
        base_acc = 10.0 + (L - 10.0) / (1 + np.exp(-k * (round_idx - x0)))
        noise = np.random.normal(0, noise_level)
        final_acc = max(0.0, min(100.0, base_acc + noise))
        
        # 2. F1 Score (Correlated with Accuracy, usually slightly lower in imbalanced, but here we track)
        # Simulating F1 as (Accuracy / 100) * 0.95 + noise
        f1_score = (final_acc / 100.0) * 0.98 - np.random.uniform(0, 0.02)
        f1_score = max(0.0, min(1.0, f1_score))
        
        # 3. AUC (Area Under Curve) - High performance models have high AUC
        # Simulating AUC as slightly higher than accuracy ratio
        auc_score = (final_acc / 100.0) + 0.02 - np.random.uniform(0, 0.01)
        auc_score = max(0.5, min(1.0, auc_score))
        
        # 4. MAE (Mean Absolute Error) - Inversely proportional to Accuracy
        # MAE starts high (e.g., 0.5) and drops to low (e.g., 0.05)
        # loss_curve = 1.0 - (accuracy / 100)
        mae = (100.0 - final_acc) / 100.0 * 0.5 + np.random.uniform(0, 0.01)
        
        return {
            "accuracy": final_acc,
            "f1": f1_score,
            "auc": auc_score,
            "mae": mae
        }

class GenericAlgorithm(FederatedAlgorithm):
    """
    Data-Driven Algorithm using a Scenario object.
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
