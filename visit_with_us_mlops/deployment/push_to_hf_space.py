import os
import argparse
from huggingface_hub import login, upload_folder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--space_repo", required=True, help="HF space repo id (e.g., user/space)")
    parser.add_argument("--deploy_dir", required=True, help="Local deployment folder path")
    args = parser.parse_args()

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN env var not set. Add it to GitHub Secrets as HF_TOKEN.")

    login(token=hf_token)

    upload_folder(
        folder_path=args.deploy_dir,
        repo_id=args.space_repo,
        repo_type="space"
    )

    print(f"✅ Deployment pushed to HF Space: {args.space_repo}")


if __name__ == "__main__":
    main()
