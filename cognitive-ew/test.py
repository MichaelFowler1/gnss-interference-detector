import torch
from torch.utils.data import DataLoader
from train_pipeline import CognitiveEWDataset, CognitiveEWNet

def quick_train():
    # Detect your RTX 3080
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training on device: {device}")
    
    # Generate a small dataset
    dataset = CognitiveEWDataset(num_samples_per_class=100)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    model = CognitiveEWNet(num_classes=3).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    for epoch in range(5): # Quick 5 epochs just to get functional weights
        for specs, labels in dataloader:
            specs, labels = specs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(specs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f"[+] Epoch {epoch+1}/5 complete.")
        
    torch.save(model.state_dict(), "cognitive_ew_model.pth")
    print("[+] Model weights saved to 'cognitive_ew_model.pth'")

if __name__ == "__main__":
    quick_train()