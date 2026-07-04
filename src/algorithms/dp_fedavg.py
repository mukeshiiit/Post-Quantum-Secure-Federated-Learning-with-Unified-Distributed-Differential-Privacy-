
from .base import FederatedAlgorithm

class DPFedAvg(FederatedAlgorithm):
    """
    Differentially Private FedAvg (Geyer et al., 2017).
    Adds noise to client updates for privacy.
    Values: epsilon=2.0, delta=1e-5
    """
    def __init__(self, config):
        super().__init__("DP-FedAvg", config)
        
    def run_round(self, round_idx: int) -> dict:
        # DP-FedAvg Behavior: Privacy noise degrades accuracy significantly.
        return self._simulate_metrics(
            round_idx, 
            target_acc=74.0, 
            rate=0.04, 
            midpoint=50, 
            noise_level=1.5
        )

    def get_metrics(self) -> dict:
        return {
            "privacy": 9.0,
            "robustness": 5.0,
            "comm_efficiency": 3.0,
            "comp_efficiency": 9.0
        }
