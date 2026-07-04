
import os
from dataclasses import dataclass, field
from typing import List, Dict

@dataclass
class SimulationConfig:
    """Configuration for the Federated Learning Simulation."""
    
    # General diverse settings
    NUM_ROUNDS: int = 200
    NUM_SEEDS: int = 5  # For generating Error Bars
    seed: int = 42
    
    # Output paths
    OUTPUT_DIR: str = "results"
    METRICS_FILE: str = "comprehensive_metrics.json"
    RADAR_FILE: str = "radar_metrics.json"
    STATS_REPORT: str = "statistical_report.txt"
    
    # Visualization settings
    PLOT_STYLE: str = "seaborn-v0_8-whitegrid"
    DPI: int = 300
    
    # Algorithm Registry
    ALGORITHMS: List[str] = field(default_factory=lambda: [
        "Vanilla FedAvg",
        "FedProx",
        "SCAFFOLD",
        "DP-FedAvg",
        "PQ-FL"
    ])

    def __post_init__(self):
        if not os.path.exists(self.OUTPUT_DIR):
            os.makedirs(self.OUTPUT_DIR)
