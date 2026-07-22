import os 
import numpy as np
import tarfile
import huggingface_hub as hf
import dotenv
import string 
import cv2
import torch
from torch.utils.data import Dataset, DataLoader 
from torch.nn.utils.rnn import pad_sequence 
from torch.utils.data import random_split
from torchvision import transforms
import torch.nn.functional as F
import random
from torch.optim.lr_scheduler import ReduceLROnPlateau

import modal


"""
This is a custom training code for training on modal.com 

"""

image = modal.Image.debian_slim().pip_install(
    "torch", "torchvision", "numpy", "opencv-python-headless", 
    "huggingface_hub", "python-dotenv", "matplotlib"
)
app = modal.App(name="lip-reading-training", image=image)

# Create a cloud drive to hold your video dataset and save your checkpoints
vol = modal.Volume.from_name("lip-reading-data", create_if_missing=True)
# -------------------------

chars = list(string.ascii_lowercase) + [' ']
blank_token = '-'
vocab = [blank_token] + chars
char2idx = {char: idx for idx, char in enumerate(vocab)}    
idx2char = {idx:char for char,idx in char2idx.items()}


    

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def ctc_decode(predictions, blank_idx=0):
    """
    Collapses repeated characters and removes CTC blank tokens.
    """
    decoded_text = []
    previous_token = None
    
    for token_id in predictions:
        if token_id != blank_idx:
            if token_id != previous_token:
                decoded_text.append(idx2char[token_id])
        previous_token = token_id
        
    return "".join(decoded_text)

class LipReadingDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir 
        self.transform = transform
        self.video_files = []
        self.align_files = []
        for speaker in sorted(os.listdir(data_dir)):
            
            speaker_dir = os.path.join(data_dir, speaker)
            if not os.path.isdir(speaker_dir):
                continue
            for file in sorted(os.listdir(speaker_dir)):
                if file.endswith(".mp4"):
                    self.video_files.append(os.path.join(speaker_dir, file))
                    alignment_file = os.path.join(speaker_dir,'align',file.replace("_lip_roi.mp4", ".align"))
                    self.align_files.append(alignment_file)
                    
    def __len__(self):
            return len(self.video_files)

    def __getitem__(self, idx):
        try:
            roi_path = self.video_files[idx]
            align_path = self.align_files[idx]
            
            roi_data = []
            cap = cv2.VideoCapture(roi_path)
            while True:
                ret, frame = cap.read()
                if not ret:
                    break 
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                roi_data.append(frame)
            cap.release()
            roi_data = np.array(roi_data)

            if self.transform:            
                roi_data = torch.tensor(roi_data, dtype=torch.float32)
                roi_data = self.transform(roi_data)
                roi_data = roi_data.unsqueeze(0) 
                
            with open(align_path, 'r') as f:
                lines = f.readlines()
                ids = []
                words = []
                for line in lines:
                    start, end, word = line.strip().split()
                    if word.lower() != 'sil':  # optional: handle silences
                        words.append(word.lower())
                
               
                full_text = " ".join(words)
                ids = [char2idx[char] for char in full_text if char in char2idx]
                ids = np.array(ids, dtype=np.int32)
                input_length = roi_data.shape[1]
                return roi_data, torch.tensor(ids, dtype=torch.int32), input_length, len(ids)
        except Exception as e:
            print(f"Skipping bad data at index {idx}. Error: {e}")
            new_idx = random.randint(0, len(self.video_files) - 1)
            return self.__getitem__(new_idx) #keep trying until a valid sample is found

inp_transform=transforms.Compose([
     transforms.Resize((100, 50))
])

def collate_fn(batch):
    roi_data, ids, input_lengths, output_lengths = zip(*batch)
    
    ids = pad_sequence(ids, batch_first=True, padding_value=0)
    
    max_len = max(t.shape[1] for t in roi_data)

    padded_roi = []
    for t in roi_data:
        pad_len = max_len - t.shape[1]
        padded = F.pad(t, (0, 0, 0, 0, 0, pad_len))
        padded_roi.append(padded)

    roi_data = torch.stack(padded_roi)

    return roi_data, ids, input_lengths, output_lengths

class LipReadingModel(torch.nn.Module):
    def __init__(self,  hidden_size=256):
        super(LipReadingModel, self).__init__()
        #input shape is (batch_size, 75, 100, 50). 
        self.conv1 = torch.nn.Conv3d(in_channels=1, out_channels=16, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1))
        self.bn1 = torch.nn.BatchNorm3d(16)
        self.pool1 = torch.nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2))
        
        self.conv2 = torch.nn.Conv3d(in_channels=16, out_channels=32, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1))
        self.bn2 = torch.nn.BatchNorm3d(32)
        self.pool2 = torch.nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2))

        self.conv3 = torch.nn.Conv3d(in_channels=32, out_channels=64, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1))
        self.bn3 = torch.nn.BatchNorm3d(64)
        self.pool3 = torch.nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2))

        self.gru = torch.nn.GRU(input_size=64*12*6, hidden_size=hidden_size, num_layers=2, batch_first=True, bidirectional=True,dropout=0.2)
        self.drop = torch.nn.Dropout(0.3)
        self.linear = torch.nn.Linear(hidden_size*2 , len(vocab))
            
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)

        x = x.view(x.size(0), x.size(2),-1)
        x,_ = self.gru(x)
        x = self.drop(x)
        x = self.linear(x)
        return x 

@app.function(
    gpu="T4",               
    cpu=4.0,                 
    memory=8192,             
    timeout=86400,          
    volumes={"/data": vol}  
)
def evaluate():
    data_dir = "/data/dataset"
    
    dataset = LipReadingDataset(data_dir, transform=inp_transform) 
    train_size = int(0.95 * len(dataset)) 
    val_size = len(dataset) - train_size 

    _, val_dataset = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    batch_size = 64
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4)      

    model = LipReadingModel()
    model.to(device)
    
    ckpt_path = "/data/lip_best.pt" if os.path.exists("/data/lip_best.pt") else "/data/lip_latest.pt"

    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded model from checkpoint at epoch {checkpoint.get('epoch', 0)} with validation loss {checkpoint.get('val_loss', 0.0):.4f}")
    else:
        print("No checkpoint found! Decoding with uninitialized model.")

    model.eval()
    
    print("\n" + "="*80)
    print("RUNNING FULL VALIDATION BATCH DECODING")
    print("="*80)

    with torch.no_grad():
        for roi_data, ids, input_lengths, output_lengths in val_dataloader:
            roi_data = roi_data.to(device)
            outputs = model(roi_data) # Shape: (batch_size, seq_len, num_classes)
            
            pred_tokens = torch.argmax(outputs, dim=-1) # Shape: (batch_size, seq_len)
            
            for i in range(len(roi_data)):
                # 1. Ground Truth
                target_length = output_lengths[i]
                gt_ids = ids[i][:target_length].tolist()
                gt_text = "".join(idx2char[idx] for idx in gt_ids if idx in idx2char)
                
                # 2. Raw CTC Output
                sample_preds = pred_tokens[i].tolist()
                raw_decoded = "".join(idx2char[idx] for idx in sample_preds)
                
                # 3. Clean Collapsed CTC Output
                clean_decoded = ctc_decode(sample_preds, blank_idx=0)
                
                print(f"Sample {i+1:02d}:")
                print(f"  TARGET: {gt_text}")
                print(f"  RAW   : {raw_decoded}")
                print(f"  CLEAN : {clean_decoded}")
                print("-" * 80)
            
            # Stop after 1 batch for inspection
            break

# --- MAIN ORCHESTRATION SHIFTED INTO THE CLOUD FUNCTION ---
@app.function(
    gpu="T4",                
    cpu=4.0,                  # 4 CPU cores for video decoding
    memory=8192,              #Request 8192 MiB (8 GiB) of RAM
    timeout=86400,          
    volumes={"/data": vol}  
)

def train():  
    data_dir = "/data/dataset"
    
    dataset = LipReadingDataset(data_dir, transform=inp_transform) 
    train_size = int(0.95 * len(dataset)) 
    val_size = len(dataset) - train_size 

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size],generator=torch.Generator().manual_seed(42))

    batch_size = 100
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn,num_workers=4)  
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn,num_workers=4)      
    epochs = 30
    log_every = 10

    model = LipReadingModel()
    criterion = torch.nn.CTCLoss()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-2)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    start_epoch = 0
    start_batch_idx = 0
    best_val_loss = float('inf')
    
    ckpt_path = "/data/lip_best.pt" if os.path.exists("/data/lip_best.pt") else "/data/lip_latest.pt"

    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint.get("epoch", 0)
        start_batch_idx = checkpoint.get("batch_idx", 0)
        best_val_loss = checkpoint.get("val_loss", float('inf'))
        print(f"Loaded model from checkpoint at epoch {checkpoint.get('epoch', 0)} with validation loss {checkpoint.get('val_loss', 0.0):.4f}")
    params = []
   
    
    print(f'training ....{sum(p.numel() for p in model.parameters())}    |  Batch size: {batch_size}')

    def validate():
        model.eval()
        val_loss = 0 
       
        with torch.no_grad():
            for batch_idx, (roi_data, ids,input_lengths, output_lengths) in enumerate(val_dataloader):
                
                roi_data = roi_data.to(device)
                ids = ids.to(device)
                outputs = model(roi_data)
                log_probs = torch.nn.functional.log_softmax(outputs, dim=2)
                log_probs = log_probs.permute(1, 0 , 2) #(T,N,C)
                loss = criterion(log_probs, ids, torch.tensor(input_lengths, dtype=torch.int32, device=device), torch.tensor(output_lengths, dtype=torch.int32, device=device))
                val_loss += loss.item()

        model.train()
        return val_loss / len(val_dataloader)

    def save_model(epoch, batch_idx, val_loss, best=False):
        state = {
            'epoch': epoch,
            'batch_idx': batch_idx,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "model_state_dict": model.state_dict(),
            "val_loss": val_loss
        }
        # Save to the mounted /data volume so it persists
        if best:
            torch.save(state, f"/data/lip_best.pt")
        else:
            torch.save(state, f"/data/lip_latest.pt")
        
        # Tell Modal to persist the new files in the volume immediately
        vol.commit()

    for epoch in range(start_epoch, epochs):
        model.train()
        qual=True
        for batch_idx, (roi_data, ids,input_lengths, output_lengths) in enumerate(train_dataloader):
            if epoch == start_epoch and batch_idx < start_batch_idx:
                continue
            
            optimizer.zero_grad()
            roi_data = roi_data.to(device)
            ids = ids.to(device)
            outputs = model(roi_data)
            if qual:
                pred_tokens = torch.argmax(outputs, dim=-1)
                sample_ids = pred_tokens[0].tolist()
                decoded= ''.join(idx2char[idx] for idx in sample_ids)
                print(decoded)
                
                qual=False
            log_probs = torch.nn.functional.log_softmax(outputs, dim=2)
            
            log_probs = log_probs.permute(1, 0, 2)  #(T,N, C)
            
            loss = criterion(log_probs, ids, torch.tensor(input_lengths, dtype=torch.int32, device=device),torch.tensor(output_lengths, dtype=torch.int32, device=device)) 
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            
            optimizer.step()
            
            if batch_idx % log_every == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Batch [{batch_idx+1}/{len(train_dataloader)}], Loss: {loss.item():.4f}")

        
        start_batch_idx = 0

        print("Running validation")
        val_loss = validate()
        print(f"Validation Loss : {val_loss}")
        scheduler.step(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss 
            save_model(epoch + 1, 0, val_loss, best=True)
        else:
            save_model(epoch + 1, 0, val_loss, best=False)


# --- TRIGGER THE CLOUD EXECUTION ---
@app.local_entrypoint()
def main():
    print("Deploying evaluation job to Modal T4...")
    train.remote()
    print("Evaluation complete!")