import os
from huggingface_hub import HfApi, hf_hub_download
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.environ.get("HF_TOKEN")
HF_REPO_ID = os.environ.get("HF_REPO_ID")

def is_hf_configured():
    return bool(HF_TOKEN and HF_REPO_ID)

def download_from_hf(filename, local_path):
    """Downloads a file from the Hugging Face Dataset repository if configured."""
    if not is_hf_configured():
        return False
        
    try:
        print(f"[HF Sync] Downloading {filename} from {HF_REPO_ID}...")
        downloaded_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=filename,
            repo_type="dataset",
            token=HF_TOKEN,
            local_dir=os.path.dirname(local_path) or ".",
            local_dir_use_symlinks=False
        )
        print(f"[HF Sync] Successfully downloaded {filename}")
        return True
    except Exception as e:
        print(f"[HF Sync] Error downloading {filename}: {e}")
        return False

def upload_to_hf(local_path, filename_in_repo):
    """Uploads a file to the Hugging Face Dataset repository if configured."""
    if not is_hf_configured():
        return False
        
    if not os.path.exists(local_path):
        print(f"[HF Sync] File {local_path} does not exist. Skipping upload.")
        return False
        
    try:
        print(f"[HF Sync] Uploading {filename_in_repo} to {HF_REPO_ID}...")
        api = HfApi(token=HF_TOKEN)
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=filename_in_repo,
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            commit_message=f"Auto-update {filename_in_repo}"
        )
        print(f"[HF Sync] Successfully uploaded {filename_in_repo}")
        return True
    except Exception as e:
        print(f"[HF Sync] Error uploading {filename_in_repo}: {e}")
        return False
