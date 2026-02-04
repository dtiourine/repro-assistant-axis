import gdown
from pathlib import Path 
import os

def download_rollouts():
    """Download rollout data from Google Drive"""
    
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    
    files = {
        "gemma-2-2b-it": {
            "file_id": "1tBjcTY0weFzI9ynwHxa9z_ODjPwgMjPe",  # From shareable link
            "output": os.path.join(data_dir, "gemma_2b_it_responses.parquet")
        },
        "Qwen2.5-3B-Instruct": {
            "file_id": "1kPZp_CWBdCOZwwlUe42S-z3eAx9PINE-", 
            "output": os.path.join(data_dir, "responses_Qwen2.5-3B-Instruct.parquet")
        },
        "Llama-3.2-3B-Instruct": {
            "file_id": "1A7M9IE216njQ3kX3KbKqv0qODKPbBk3e",
            "output": os.path.join(data_dir, "responses_Llama-3.2-3B-Instruct.parquet")
        }
        
        
    }
    
    for name, info in files.items():
        print(f"Downloading {name}...")
        os.makedirs(os.path.dirname(info['output']), exist_ok=True)
        
        url = f"https://drive.google.com/uc?id={info['file_id']}"
        gdown.download(url, info['output'], quiet=False)
        
    print("Download complete!")

if __name__ == "__main__":
    download_rollouts()