import os
import pandas as pd
import numpy as np
import time

def generate_dataset(output_dir="dataset", num_records=10000, num_clients=100, seed=42):
    """
    Generates synthetic 6G IoT dataset and partitions it among clients using 
    Dirichlet distribution to create non-IID (heterogeneous) data silos.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print(f"Generating {num_records} records for 6G IoT Dataset across {num_clients} clients...")
    
    # Set seed for reproducibility as requested in R5-C1 / R3-C2
    np.random.seed(seed)
           
    timestamps = pd.date_range(start="2025-01-01", periods=num_records, freq="T")
    
    # Generate labels (4 Device Types)
    device_types = ["SmartSensor", "Wearable", "AutonomousVehicle", "IndustrialController"]
    labels = np.random.choice(device_types, num_records)
    
    # Dirichlet Allocation Logic for Non-IID distribution (alpha=0.5 for moderate heterogeneity)
    # Each client gets a different proportion of each class
    client_ids = []
    alpha = 0.5
    class_priors = np.random.dirichlet([alpha] * num_clients, size=len(device_types))
    
    # Assign each record to a client based on its class's Dirichlet distribution
    for label in labels:
        class_idx = device_types.index(label)
        client_assigned = np.random.choice(range(num_clients), p=class_priors[class_idx])
        client_ids.append(f"Client_{client_assigned:03d}")
    
    data = {
        "Record_ID": [f"REC_{i:05d}" for i in range(num_records)],
        "Client_ID": client_ids,
        "Timestamp": timestamps,
        "Device_Type": labels,
        "Battery_Level": np.random.uniform(10.0, 100.0, num_records).round(2),
        "Signal_Strength_dBm": np.random.normal(-60, 10, num_records).round(1), # 6G Signal
        "Latency_ms": np.random.gamma(2, 2, num_records).round(3), # Ultra-low latency
        "Data_Volume_MB": np.random.exponential(50, num_records).round(2),
        "Local_Model_Accuracy": np.random.uniform(0.70, 0.95, num_records).round(4),
        "Privacy_Budget_Used": np.random.uniform(0.1, 5.0, num_records).round(2),
        "User_Heart_Rate": np.random.normal(75, 12, num_records).astype(int),
        "Location_X": np.random.uniform(0, 1, num_records).round(4), # scaled [0,1]^2
        "Location_Y": np.random.uniform(0, 1, num_records).round(4),
    }
    
    df = pd.DataFrame(data)
    
    # Save Main Dataset
    csv_path = os.path.join(output_dir, "6G_IoT_Federated_Data.csv")
    df.to_csv(csv_path, index=False)
    print(f"[SUCCESS] Non-IID Dataset partitioned and saved to {csv_path}")
    
    desc = """
# 6G IoT Federated Learning Dataset - Attribute Description

1. **Record_ID**: Unique identifier for the data point.
2. **Client_ID**: Edge device owning the data (Allocated via Dirichlet distribution seed=42 for non-IID).
3. **Timestamp**: Time of data recording.
4. **Device_Type**: Category of the IoT device (Wearable, Vehicle, etc.).
5. **Battery_Level**: Remaining energy percentage (Critical for FL participation).
6. **Signal_Strength_dBm**: 6G channel quality indicator [-120 to -50 dBm].
7. **Latency_ms**: Network latency [5-200 ms], critical for 6G URLLC.
8. **Data_Volume_MB**: Amount of local data available for training.
9. **Local_Model_Accuracy**: Pre-aggregation accuracy of the local model.
10. **Privacy_Budget_Used**: Cumulative epsilon consumed by the device.
11. **User_Heart_Rate**: Sensitive health metric (Protected attribute) [40-180 bpm].
12. **Location_X/Y**: Geo-spatial coordinates (Sensitive attribute) [0,1]^2.
    """
    
    desc_path = os.path.join(output_dir, "Dataset_Description.txt")
    with open(desc_path, "w") as f:
        f.write(desc.strip())
    print(f"[SUCCESS] Descriptions saved to {desc_path}")

if __name__ == "__main__":
    generate_dataset()
