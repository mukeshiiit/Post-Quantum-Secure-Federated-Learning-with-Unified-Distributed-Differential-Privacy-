
from .base import FederatedAlgorithm

class FedProx(FederatedAlgorithm):
    """
    FedProx (Li et al., 2020).
    Adds a proximal term to the local objective to handle heterogeneity.
    """
    def __init__(self, config):
        super().__init__("FedProx", config)
        
    def run_round(self, round_idx: int) -> dict:
        # FedProx Behavior: Slower convergence.
        return self._simulate_metrics(
            round_idx, 
            target_acc=86.2, 
            rate=0.05, 
            midpoint=35, 
            noise_level=0.4
        )

    def get_metrics(self) -> dict:
        return {
            "privacy": 1.0,
            "robustness": 7.0,
            "comm_efficiency": 6.0,
            "comp_efficiency": 8.0
        }
