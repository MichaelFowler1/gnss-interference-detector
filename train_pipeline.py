import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Bring over our physics engine from Phase 1
from realistic_rf_env import RealisticRFEnvironment 

class CognitiveEWDataset(Dataset):
    """
    On-the-fly PyTorch Dataset that generates realistic RF environments.
    Converts 1D complex I/Q vectors into 2D Spectrogram Tensors.
    """
    def __init__(self, num_samples_per_class=200, sample_rate=1e6, num_symbols=1024):
        self.env = RealisticRFEnvironment(sample_rate=sample_rate, num_symbols=num_symbols)
        self.num_samples_per_class = num_samples_per_class
        
        # Class Map: 0 = CLEAR, 1 = BARRAGE_JAMMED, 2 = TONE_JAMMED
        self.labels = np.array([0, 1, 2] * num_samples_per_class)
        np.random.shuffle(self.labels) # Ensure mixed batches

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        
        # 1. Generate pristine signal and apply atmospheric/hardware distortions
        base_signal = self.env.generate_base_qpsk()
        iq_signal = self.env.apply_channel_effects(
            base_signal, 
            cfo_hz=np.random.uniform(-15000, 15000), # Random Doppler shift
            snr_db=np.random.uniform(10, 25)         # Varied signal strength
        )
        
        # 2. Inject Electronic Attack profile based on label
        if label == 1:
            iq_signal = self.env.apply_barrage_jamming(iq_signal, jsr_db=np.random.uniform(10, 22))
            iq_signal = self.env.apply_hardware_clipping(iq_signal, clip_percentile=93)
        elif label == 2:
            # Random tone jammer offset anywhere inside our bandwidth
            random_offset = np.random.uniform(-300e3, 300e3)
            iq_signal = self.env.apply_tone_jamming(iq_signal, jsr_db=np.random.uniform(12, 25), offset_hz=random_offset)
            iq_signal = self.env.apply_hardware_clipping(iq_signal, clip_percentile=96)
            
        # 3. Convert 1D complex numpy array to 1D complex PyTorch Tensor
        # PyTorch processes complex numbers as complex64
        iq_tensor = torch.from_numpy(iq_signal).to(torch.complex64)
        
        # 4. Compute Short-Time Fourier Transform (STFT) to make it a 2D "image"
        # n_fft is our frequency resolution; hop_length controls time overlapping
        stft_matrix = torch.stft(
            iq_tensor, 
            n_fft=128, 
            hop_length=32, 
            return_complex=True
        )
        
        # Take the absolute magnitude of the complex matrix to get raw power
        spectrogram = torch.abs(stft_matrix)
        
        # Convert to logarithmic decibel scale (dB) so the CNN can see low-power features
        spectrogram_db = 20 * torch.log10(spectrogram + 1e-6)
        
        # Add a channel dimension [Channels=1, Height=Frequency, Width=Time]
        spectrogram_db = spectrogram_db.unsqueeze(0)
        
        return spectrogram_db, torch.tensor(label, dtype=torch.long)


class CognitiveEWNet(nn.Module):
    """
    An ultra-lightweight CNN architecture designed for edge deployment.
    Minimizes parameters to reduce latency on tactical SWaP hardware.
    """
    def __init__(self, num_classes=3):
        super(CognitiveEWNet, self).__init__()
        
        # Input shape: [Batch, 1, 65, 257] (Based on n_fft=128 and 1024 symbols * 8 sps)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Global Average Pooling ensures the network handles variable transmission window lengths
        self.global_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        # Tiny linear layer for low memory footprint
        self.fc = nn.Linear(16 * 4 * 4, num_classes)

    def forward(self, x):
        # Layer 1: Feature Extraction
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        # Layer 2: Deeper texturing
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        
        # Reduce dimensions to a fixed size before classification
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        
        # Output Logits
        x = self.fc(x)
        return x

# ==========================================
# Functional Test Verification
# ==========================================
if __name__ == "__main__":
    print("[*] Instantiating Synthetic EW Dataset...")
    dataset = CognitiveEWDataset(num_samples_per_class=10)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # Grab a single batch to test shapes
    specs, labels = next(iter(dataloader))
    print(f"[+] Batch Loaded! Tensors Shape: {specs.shape} (Batch Size, Channels, Freq Bins, Time Steps)")
    print(f"[+] Labels in Batch: {labels}")
    
    print("[*] Initializing CognitiveEWNet Neural Network...")
    model = CognitiveEWNet(num_classes=3)
    
    # Run a test forward pass
    outputs = model(specs)
    print(f"[+] Forward pass successful! Output Shape: {outputs.shape}")