import modal
import tarfile

app = modal.App(name="unzip-dataset")
vol = modal.Volume.from_name("lip-reading-data")

@app.function(volumes={"/data": vol}) 
def extract():
    print("Extracting files in the cloud...")
    # Open the uploaded archive from the cloud volume
    with tarfile.open("/data/dataset.tar.gz", "r:gz") as tar:
        tar.extractall("/data/")
    
    # Commit the changes to permanently save the extracted files
    vol.commit()
    print("Extraction complete!")

@app.local_entrypoint()
def main():
    extract.remote()