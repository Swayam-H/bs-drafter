import os
import sys
from huggingface_hub import HfApi, login
from dotenv import load_dotenv

print("=== BRAWL STARS DRAFTER AUTOMATED HOSTING ===")
print("This script will automatically create and host your app on Hugging Face for FREE.")
print("You need a Hugging Face account and an Access Token (Write permission).")
print("Get your token here: https://huggingface.co/settings/tokens\n")

# Load existing .env if any
load_dotenv()
existing_token = os.environ.get("HF_TOKEN")

hf_token = input(f"Paste your Hugging Face Write Token {'(Press Enter to use existing)' if existing_token else ''}: ").strip()
if not hf_token and existing_token:
    hf_token = existing_token
elif not hf_token:
    print("Error: Token is required.")
    sys.exit(1)

try:
    login(token=hf_token)
    api = HfApi(token=hf_token)
    user = api.whoami()["name"]
    print(f"\nSuccessfully logged in as: {user}")
except Exception as e:
    print(f"Error logging in: {e}")
    sys.exit(1)

app_name = input("Enter a name for your app (e.g. bs-drafter-bot): ").strip()
if not app_name:
    app_name = "bs-drafter-bot"

space_repo_id = f"{user}/{app_name}"
dataset_repo_id = f"{user}/{app_name}-data"

print("\n--- STEP 1: Creating Data Storage (Dataset) ---")
try:
    api.create_repo(repo_id=dataset_repo_id, repo_type="dataset", exist_ok=True)
    print(f"Dataset created: https://huggingface.co/datasets/{dataset_repo_id}")
except Exception as e:
    print(f"Error creating dataset: {e}")

print("\n--- STEP 2: Creating Web Server (Space) ---")
try:
    api.create_repo(repo_id=space_repo_id, repo_type="space", space_sdk="docker", exist_ok=True)
    print(f"Space created: https://huggingface.co/spaces/{space_repo_id}")
except Exception as e:
    print(f"Error creating space: {e}")

print("\n--- STEP 3: Setting Cloud Secrets ---")
brawl_api_key = os.environ.get("BRAWL_STARS_API_KEY", "")
if not brawl_api_key:
    brawl_api_key = input("Enter your Brawl Stars API Key: ").strip()

try:
    api.add_space_secret(repo_id=space_repo_id, key="HF_TOKEN", value=hf_token)
    api.add_space_secret(repo_id=space_repo_id, key="HF_REPO_ID", value=dataset_repo_id)
    if brawl_api_key:
        api.add_space_secret(repo_id=space_repo_id, key="BRAWL_STARS_API_KEY", value=brawl_api_key)
    print("Secrets successfully configured in the cloud.")
except Exception as e:
    print(f"Warning setting secrets (might already exist): {e}")

print("\n--- STEP 4: Uploading Files to Cloud (This may take a minute) ---")
# Files to exclude from upload
ignore_patterns = ["*.pkl", "*.csv", ".git*", "__pycache__", "venv", ".env", ".venv", "deploy.py"]

try:
    api.upload_folder(
        folder_path=".",
        repo_id=space_repo_id,
        repo_type="space",
        ignore_patterns=ignore_patterns,
        commit_message="Initial automated deployment"
    )
    print("\n✅ Code uploaded successfully to the Space!")
except Exception as e:
    print(f"Error uploading to Space: {e}")

print("\n--- STEP 5: Uploading Initial Database ---")
try:
    if os.path.exists("matches.csv"):
        api.upload_file(
            path_or_fileobj="matches.csv",
            path_in_repo="matches.csv",
            repo_id=dataset_repo_id,
            repo_type="dataset"
        )
    if os.path.exists("draft_model.pkl"):
        api.upload_file(
            path_or_fileobj="draft_model.pkl",
            path_in_repo="draft_model.pkl",
            repo_id=dataset_repo_id,
            repo_type="dataset"
        )
    print("✅ Initial data uploaded successfully!")
except Exception as e:
    print(f"Error uploading data: {e}")

print(f"\n🎉 DEPLOYMENT COMPLETE! 🎉")
print(f"Your app is now booting up at: https://huggingface.co/spaces/{space_repo_id}")
print("It will take a few minutes for the Docker container to build and start.")
