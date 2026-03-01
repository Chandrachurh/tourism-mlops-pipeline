import os
import argparse
from huggingface_hub import login, upload_folder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", required=True, help="HF model repo id (e.g., user/model-repo)")
    parser.add_argument("--artifact_dir", required=True, help="Local folder containing model + logs")
    args = parser.parse_args()

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN env var not set. Add it to GitHub Secrets as HF_TOKEN.")

    login(token=hf_token)

    upload_folder(
        folder_path=args.artifact_dir,
        repo_id=args.repo_id,
        repo_type="model"
    )

    print(f"✅ Uploaded model + artifacts to HF Model Hub: {args.repo_id}")


if __name__ == "__main__":
    main()
