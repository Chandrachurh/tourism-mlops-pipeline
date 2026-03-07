from huggingface_hub.utils import RepositoryNotFoundError
from huggingface_hub import HfApi, create_repo
import os

repo_id = "chandrachurhghosh/tourism-mlops-dataset"
repo_type = "dataset"

api = HfApi(token=os.getenv("HF_TOKEN"))

try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Dataset '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Dataset '{repo_id}' not found. Creating it...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Dataset '{repo_id}' created.")

folder_path = "visit_with_us_mlops/data/rawdata"

print("Current working directory:", os.getcwd())
print("Folder path:", folder_path)
print("Folder exists:", os.path.isdir(folder_path))

if not os.path.isdir(folder_path):
    raise FileNotFoundError(f"Provided path does not exist: {folder_path}")

api.upload_folder(
    folder_path=folder_path,
    repo_id=repo_id,
    repo_type=repo_type,
    path_in_repo="rawdata"
)

print(f"Uploaded contents of '{folder_path}' to '{repo_id}/rawdata'")
