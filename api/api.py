
import string 
import torch 
import random 
from torch.utils.data import Dataset
import os
import cv2
import numpy as np
import torch.nn.functional as F
from fastapi import FastAPI , HTTPException
from pydantic import BaseModel, Field



chars = list(string.ascii_lowercase) + [' ']
blank_token = '-'
vocab = [blank_token] + chars
char2idx = {char: idx for idx, char in enumerate(vocab)}    
idx2char = {idx:char for char,idx in char2idx.items()}

    

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


def ctc_decode(predictions, blank_idx=0):
    """
    Collapses repeated characters and removes CTC blank tokens.
    """
    decoded_text = []
    previous_token = None
    
    for token_id in predictions:
        token_id = token_id.item()
        if token_id != blank_idx:
            if token_id != previous_token:
                decoded_text.append(idx2char[token_id])
        previous_token = token_id
        
    return "".join(decoded_text)
def prepare_data(video):
        video_path  = os.path.join('sample_data','videos', video)  
        align_path = os.path.join('sample_data','labels', video.replace("_lip_roi.mp4", ".align"))     
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break 
            frame = cv2.cvtColor(frame , cv2.COLOR_BGR2GRAY)
            frame = cv2.resize(frame, (100, 50)) 
            frames.append(frame)
        cap.release()
       
        return np.array(frames, dtype=np.float32) , align_path










class RequestModel(BaseModel):
    video_path: str = Field(..., description="Path to the video file for lip reading prediction")

class ResponseModel(BaseModel):
    prediction: str 
    label: str

class SampleVideosResponseModel(BaseModel):
    sample_videos: list[str] = Field(..., description="List of sample video filenames available for testing")


model = None
# modal volume get lip-reading-data lip_latest.pt ./lip_latest.pt to get model weights from modal
App = FastAPI(title="Lip Reading API", description="API for lip reading from video", version="1.0.0")
@App.on_event("startup")
def load_model():
    global model
    model = LipReadingModel().to(device)
    chkpt = torch.load("lip_latest.pt", map_location=device)
    weights = chkpt['model_state_dict']
    model.load_state_dict(weights)

@App.get("/")
def health_check():
    return {"message": "Health check successful. The API is running."}

@App.get("/samples", response_model=SampleVideosResponseModel)
def get_sample_videos():
    sample_dir = os.path.join('sample_data', 'videos')
    if not os.path.exists(sample_dir):
         return {"error": "Sample videos directory not found."}
    
    sample_videos = [f for f in os.listdir(sample_dir) if f.endswith('.mp4')]
    return SampleVideosResponseModel(sample_videos=sample_videos)


@App.get("/predict",response_model=ResponseModel)
def predict(request: RequestModel):
    try:
        rois, align_path = prepare_data(request.video_path)
        
        print(align_path)
        with open(align_path, 'r') as f:
                        lines = f.readlines()
                        
                        words = []
                        
                        for line in lines:
                            start, end, word = line.strip().split()
                            if word.lower() != 'sil':  # optional: handle silences
                                words.append(word.lower())
                        
                        
                        full_text = " ".join(words)
                    
        rois = torch.tensor(rois, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        rois = rois.permute(0,1,2,4,3)
        
        with torch.no_grad():
            output = model(rois)
            probs = torch.nn.functional.softmax(output, dim=-1)
            idxs = torch.argmax(probs , dim=-1)
            
        return ResponseModel(prediction = ctc_decode(idxs[0]), label = full_text)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


