import requests 

url = "http://127.0.0.1:8000"


#sample requests 
sample_path = "/teamspace/studios/this_studio/dataset/1/data/s5_processed/bbal2n_lip_roi.mp4"


response = requests.get(f"{url}/predict", json={"video_path": sample_path})
if response.status_code == 200:
    print("Prediction:", response.json())
else:
    print('error')
