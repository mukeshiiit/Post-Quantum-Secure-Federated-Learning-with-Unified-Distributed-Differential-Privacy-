import torch
import torch.nn as nn

class PQFL_MLP(nn.Module):
    """
   Patrameters setting
    """
    def __init__(self, input_dim=7, hidden1=128, hidden2=64, num_classes=4):
        super(PQFL_MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden2, num_classes)
        
    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x
def verify_parameter_count():
    model = PQFL_MLP()
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("=========================================")
    print("      NN Setting        ")
    print("=========================================")
    print(model)       
    return total_params

if __name__ == "__main__":
    verify_parameter_count()
