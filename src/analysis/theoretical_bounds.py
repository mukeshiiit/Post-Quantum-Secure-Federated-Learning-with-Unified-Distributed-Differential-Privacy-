import numpy as np

def analyze_key_setup_scalability():
    print("=========================================")
    print("   Scalability checking    ")
    print("=========================================")
    for N in [100, 1000, 10000]:
        exchanges = (N * (N - 1)) // 2
        print(f"Clients (N={N}): {exchanges:,} pairwise.")
    print("************")
    print("-----------------------------------------\n")

def analyze_communication_overhead():
    print("=========================================")
    print("   Overhead       ")
    print("=========================================")
   
    plaintext_mb = 17.5
 
    expansion_factor = 2.0
    pq_fl_mb = plaintext_mb * expansion_factor
    
    print(f"Vanilla FedAvg Baseline Cost: {plaintext_mb} MB")
    print(f"Ring-LWE Ciphertext Expansion Factor: {expansion_factor}x")
    print(f"PQ-FL Total Communication Cost: {pq_fl_mb} MB")
    print("*********")
    print("-----------------------------------------\n")

def analyze_dropout_variance(n_clients=10, dropouts=2, sigma=1.0, C=1.0):
    print("=========================================")
    print("   Gap Analysis       ")
    print("=========================================")
    target_variance = (sigma * C)**2
    
    active_clients = n_clients - dropouts
    post_dropout_variance = (sigma * C)**2 * (active_clients / n_clients)
    
    print(f"Target DP Noise Variance: {target_variance:.2f}")
    print(f"Number of Selected Clients (n): {n_clients}")
    print(f"Number of Dropouts (|D|): {dropouts}")
    print(f"Post-Dropout Aggregated Variance: {post_dropout_variance:.2f}")
    print(f"Variance Gap (Loss): {target_variance - post_dropout_variance:.2f}")
    print("***********")
    print("-----------------------------------------\n")

def analyze_decryption_failure():
    print("=========================================")
    print("   Failure Bound              ")
    print("=========================================")
  
    q = 12289
    print(f"Ring modulus (q): {q}")
    print(f"Safety margin (q/4): {q/4:.2f}")
    
    print("-----------------------------------------\n")

if __name__ == "__main__":
    analyze_key_setup_scalability()
    analyze_communication_overhead()
    analyze_dropout_variance()
    analyze_decryption_failure()
