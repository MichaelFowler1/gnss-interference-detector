# train_and_save.py
import torch
from torch.utils.data import DataLoader
from train_pipeline import CognitiveEWDataset, CognitiveEWNet

def quick_train():
    # Detect your RTX 3080 to utilize GPU acceleration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training on device: {device}")
    
    # 1. Generate a small synthetic dataset using our Phase 2 pipeline
    # 100 samples per class = 300 total signal environments (Clear, Barrage, Tone)
    dataset = CognitiveEWDataset(num_samples_per_class=100)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    # 2. Instantiate our lightweight edge CNN and push it to the GPU
    model = CognitiveEWNet(num_classes=3).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 3. Core Neural Network Training Loop
    model.train()
    print("[*] Starting optimization loop...")
    for epoch in range(5): # 5 quick epochs to learn basic patterns
        total_loss = 0
        for specs, labels in dataloader:
            # Move the spectrogram tensors to your RTX 3080
            specs, labels = specs.to(device), labels.to(device)
            
            # Forward pass & error calculation
            optimizer.zero_grad()
            outputs = model(specs)
            loss = criterion(outputs, labels)
            
            # Backpropagation (adjusting the weights)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        print(f"[+] Epoch {epoch+1}/5 complete. Average Loss: {avg_loss:.4f}")
        
    # 4. Save the trained neural weights to disk
    torch.save(model.state_dict(), "cognitive_ew_model.pth")
    print("[+] Optimization complete. Model weights saved to 'cognitive_ew_model.pth'")

if __name__ == "__main__":
    quick_train()