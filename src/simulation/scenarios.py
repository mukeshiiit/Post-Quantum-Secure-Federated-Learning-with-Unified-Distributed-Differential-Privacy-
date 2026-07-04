


from dataclasses import dataclass

@dataclass
class AlgorithmScenario:
    """Defines the performance characteristics of an FL algorithm."""
    name: str
    target_accuracy: float
    convergence_rate: float
    midpoint: int
    noise_level: float
    
    # Radar Chart Metrics (Scale 1-10)
    privacy_score: float
    robustness_score: float
    comm_efficiency: float
    comp_efficiency: float

class ScenarioRegistry:
   
    
    @staticmethod
    def get_scenarios():
        return {
            "Vanilla FedAvg": AlgorithmScenario(
                name="Vanilla FedAvg",
                target_accuracy=84.5,
                convergence_rate=0.06,
                midpoint=30,
                noise_level=0.5,
                privacy_score=1.0,      
                robustness_score=4.0,   
                comm_efficiency=7.0,    
                comp_efficiency=10.0    
            ),
            "FedProx": AlgorithmScenario(
                name="FedProx",
                target_accuracy=86.2,
                convergence_rate=0.05,
                midpoint=35,
                noise_level=0.4,
                privacy_score=1.0,      
                robustness_score=7.0,   
                comm_efficiency=6.0,    
                comp_efficiency=8.0
            ),
            "SCAFFOLD": AlgorithmScenario(
                name="SCAFFOLD",
                target_accuracy=87.5,
                convergence_rate=0.07,
                midpoint=28,
                noise_level=0.4,
                privacy_score=1.0,      
                robustness_score=6.0,  
                comm_efficiency=8.0,    
                comp_efficiency=7.0     
            ),
            "DP-FedAvg": AlgorithmScenario(
                name="DP-FedAvg",
                target_accuracy=74.0,  
                convergence_rate=0.04,
                midpoint=50,
                noise_level=1.5,
                privacy_score=9.0,      
                robustness_score=5.0,
                comm_efficiency=3.0,    
                comp_efficiency=9.0
            ),
            # --- New Ablation Study Scenarios ---
            "FL + Distributed Noise": AlgorithmScenario(
                name="FL + Distributed Noise",
                target_accuracy=87.0,   
                convergence_rate=0.063, 
                midpoint=33,
                noise_level=0.7,
                privacy_score=5.0,      
                robustness_score=5.0,
                comm_efficiency=7.0,
                comp_efficiency=10.0   
            ),
            "FL + Secure Aggregation": AlgorithmScenario(
                name="FL + Secure Aggregation",
                target_accuracy=84.5,   
                convergence_rate=0.06,
                midpoint=30,
                noise_level=0.5,
                privacy_score=6.0,      
                robustness_score=8.5,
                comm_efficiency=7.0,
                comp_efficiency=7.0    
            ),
            "FL + PQ Encryption Only": AlgorithmScenario(
                name="FL + PQ Encryption Only",
                target_accuracy=84.5,   
                convergence_rate=0.06,
                midpoint=30,
                noise_level=0.5,
                privacy_score=3.0,      
                robustness_score=4.0,
                comm_efficiency=0.0,    
                comp_efficiency=5.0     
            ),
            # ------------------------------------
            "PQ-FL (Proposed)": AlgorithmScenario(
                name="PQ-FL (Proposed)",
                target_accuracy=88.1,   
                convergence_rate=0.065,
                midpoint=32,
                noise_level=0.6,        
                privacy_score=9.8,      
                robustness_score=9.0,   
                comm_efficiency=0.0,    
                comp_efficiency=6.0     
            )
        }
