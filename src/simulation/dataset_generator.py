import os
import pandas as pd
import numpy as np
import time

def generate_dataset(output_dir='dataset', num_records=10000, num_clients=100, seed=42):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print(f'Generating {num_records} records for 6G IoT Dataset across {num_clients} clients...')
    np.random.seed(seed)
    timestamps = pd.date_range(start='2025-01-01', periods=num_records, freq='T')
    device_types = ['SmartSensor', 'Wearable', 'AutonomousVehicle', 'IndustrialController']
    labels = np.random.choice(device_types, num_records)
    client_ids = []
    alpha = 0.5
    class_priors = np.random.dirichlet([alpha] * num_clients, size=len(device_types))
    for label in labels:
        class_idx = device_types.index(label)
        client_assigned = np.random.choice(range(num_clients), p=class_priors[class_idx])
        client_ids.append(f'Client_{client_assigned:03d}')
    data = {'Record_ID': [f'REC_{i:05d}' for i in range(num_records)], 'Client_ID': client_ids, 'Timestamp': timestamps, 'Device_Type': labels, 'Battery_Level': np.random.uniform(10.0, 100.0, num_records).round(2), 'Signal_Strength_dBm': np.random.normal(-60, 10, num_records).round(1), 'Latency_ms': np.random.gamma(2, 2, num_records).round(3), 'Data_Volume_MB': np.random.exponential(50, num_records).round(2), 'Local_Model_Accuracy': np.random.uniform(0.7, 0.95, num_records).round(4), 'Privacy_Budget_Used': np.random.uniform(0.1, 5.0, num_records).round(2), 'User_Heart_Rate': np.random.normal(75, 12, num_records).astype(int), 'Location_X': np.random.uniform(0, 1, num_records).round(4), 'Location_Y': np.random.uniform(0, 1, num_records).round(4)}
    df = pd.DataFrame(data)
    csv_path = os.path.join(output_dir, '6G_IoT_Federated_Data.csv')
    df.to_csv(csv_path, index=False)
    print(f'[SUCCESS] Non-IID Dataset partitioned and saved to {csv_path}')
    desc = '\n# 6G IoT Federated Learning Dataset - Attribute Description\n\n1. **Record_ID**: Unique identifier for the data point.\n2. **Client_ID**: Edge device owning the data (Allocated via Dirichlet distribution seed=42 for non-IID).\n3. **Timestamp**: Time of data recording.\n4. **Device_Type**: Category of the IoT device (Wearable, Vehicle, etc.).\n5. **Battery_Level**: Remaining energy percentage (Critical for FL participation).\n6. **Signal_Strength_dBm**: 6G channel quality indicator [-120 to -50 dBm].\n7. **Latency_ms**: Network latency [5-200 ms], critical for 6G URLLC.\n8. **Data_Volume_MB**: Amount of local data available for training.\n9. **Local_Model_Accuracy**: Pre-aggregation accuracy of the local model.\n10. **Privacy_Budget_Used**: Cumulative epsilon consumed by the device.\n11. **User_Heart_Rate**: Sensitive health metric (Protected attribute) [40-180 bpm].\n12. **Location_X/Y**: Geo-spatial coordinates (Sensitive attribute) [0,1]^2.\n    '
    desc_path = os.path.join(output_dir, 'Dataset_Description.txt')
    with open(desc_path, 'w') as f:
        f.write(desc.strip())
    print(f'[SUCCESS] Descriptions saved to {desc_path}')
if __name__ == '__main__':
    generate_dataset()
