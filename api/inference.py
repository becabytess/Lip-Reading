import requests 

url = "http://127.0.0.1:8001"


#sample requests 
sample_path = "bbbv6s_lip_roi.mp4"


response = requests.get(f"{url}/predict", json={"video_path": sample_path})
if response.status_code == 200:
    print("Prediction:", response.json())
else:
    print('error')
