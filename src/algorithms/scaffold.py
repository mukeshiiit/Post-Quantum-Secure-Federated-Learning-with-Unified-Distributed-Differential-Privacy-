
from .base import FederatedAlgorithm

class SCAFFOLD(FederatedAlgorithm):
    """

    """
    def __init__(self, config):
        super().__init__("SCAFFOLD", config)
        
    def run_round(self, round_idx: int) -> dict:

        return self._simulate_metrics(
            round_idx, 
            target_acc=87.5, 
            rate=0.07, 
            midpoint=28, 
            noise_level=0.4
        )

    def get_metrics(self) -> dict:
        return {
            "privacy": 1.0,
            "robustness": 6.0,
            "comm_efficiency": 8.0,
            "comp_efficiency": 7.0
        }
