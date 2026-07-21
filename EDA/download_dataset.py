import kagglehub
import shutil

# Download latest version
path = kagglehub.dataset_download("jedidiahangekouakou/grid-corpus-dataset-for-training-lipnet")

print("Path to dataset files:", path)

# Copy the dataset to a local directory
shutil.move(path, "../raw_dataset")




