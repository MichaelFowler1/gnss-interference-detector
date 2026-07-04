import torch
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from train_pipeline import CognitiveEWNet

app = FastAPI(
    title="Cognitive EW Inference Node",
    description="Edge AI service for real-time Electronic Attack (EA) detection.",
    version="1.0.0"
)

# Define the incoming data schema
class IQPayload(BaseModel):
    # Streaming I/Q data arrives as a flattened list of pairs: [I0, Q0, I1, Q1, ...]
    iq_data: List[float] 

# Map labels back to military alert statuses
CLASS_MAP = {
    0: {"status": "CLEAR", "type": "NONE"},
    1: {"status": "JAMMED", "type": "BARRAGE"},
    2: {"status": "JAMMED", "type": "TONE"}
}

# Global variables to hold the model and device status
model = None
device = None

@app.on_event("startup")
def load_edge_model():
    """Executes on startup. Loads the model directly onto the GPU."""
    global model, device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Initializing Cognitive EW Node. Target Hardware: {device}")
    
    model = CognitiveEWNet(num_classes=3)
    try:
        model.load_state_dict(torch.load("cognitive_ew_model.pth", map_location=device))
        model.to(device)
        model.eval() # Freeze layers for inference optimization
        print("[+] Defense-grade model successfully loaded into GPU memory.")
    except FileNotFoundError:
        print("[-] Critical Error: 'cognitive_ew_model.pth' not found. Run training script first.")

@app.post("/predict")
async def predict_rf_environment(payload: IQPayload):
    """
    Ingests live 1D baseband I/Q data, transforms it to a 2D spectrogram 
    via GPU-accelerated STFT, and determines spectrum state.
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model not initialized.")
        
    try:
        # 1. Convert flat incoming list back into a complex numpy array
        raw_array = np.array(payload.iq_data, dtype=np.float32)
        if len(raw_array) % 2 != 0:
            raise ValueError("Payload must contain paired In-Phase and Quadrature samples.")
            
        # Reconstruct complex numbers: I = even indices, Q = odd indices
        iq_complex = raw_array[0::2] + 1j * raw_array[1::2]
        
        # 2. Push raw data to your RTX 3080
        iq_tensor = torch.from_numpy(iq_complex).to(torch.complex64).to(device)
        
        # 3. Compute STFT inside the GPU (Ultra low-latency DSP)
        stft_matrix = torch.stft(
            iq_tensor, 
            n_fft=128, 
            hop_length=32, 
            return_complex=True
        )
        spectrogram = torch.abs(stft_matrix)
        spectrogram_db = 20 * torch.log10(spectrogram + 1e-6)
        
        # Add Batch and Channel dimensions: [Batch=1, Channel=1, Freq, Time]
        input_tensor = spectrogram_db.unsqueeze(0).unsqueeze(0)
        
        # 4. Forward Pass through the CNN
        with torch.no_grad(): # Disable gradient tracking to maximize inference speeds
            logits = model(input_tensor)
            probabilities = torch.softmax(logits, dim=1)
            confidence, predicted_class = torch.max(probabilities, dim=1)
            
        # 5. Formulate Telemetry Alert Response
        classification = CLASS_MAP[predicted_class.item()]
        
        return {
            "status": classification["status"],
            "type": classification["type"],
            "confidence": round(confidence.item(), 4),
            "hardware_accelerated": "cuda" in str(device)
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"DSP Pipeline Error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    # Spin up the local microservice on port 8000
    uvicorn.run(app, host="127.0.0.1", port=8000)