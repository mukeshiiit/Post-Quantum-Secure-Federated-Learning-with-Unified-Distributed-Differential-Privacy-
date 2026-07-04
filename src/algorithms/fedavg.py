
from .base import FederatedAlgorithm

class FedAvg(FederatedAlgorithm):
    """
    Vanilla Federated Averaging (McMahan et al., 2017).
    Standard aggregation of model updates. 
    No privacy mechanism.
    """
    def __init__(self, config):
        super().__init__("Vanilla FedAvg", config)
        
    def run_round(self, round_idx: int) -> dict:
        # FedAvg Behavior: Good convergence, assumes IID-ish data.
        return self._simulate_metrics(
            round_idx, 
            target_acc=84.5, 
            rate=0.06, 
            midpoint=30, 
            noise_level=0.5
        )
        
    def get_metrics(self) -> dict:
        return {
            "privacy": 1.0,
            "robustness": 4.0,
            "comm_efficiency": 7.0,
            "comp_efficiency": 10.0
        }
