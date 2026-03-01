import os
import argparse
import pandas as pd

from huggingface_hub import login, upload_folder
from datasets import load_dataset
from sklearn.model_selection import train_test_split


def clean_and_prepare(df: pd.DataFrame) -> pd.DataFrame:
    # Drop common unnecessary columns if present
    for col in ["CustomerID"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Basic missing value handling
    # Numeric -> median
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())

    # Categorical -> mode
    cat_cols = df.select_dtypes(include=["object"]).columns
    for col in cat_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].mode(dropna=True)[0])

    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_repo", required=True, help="HF dataset repo id (e.g., user/repo)")
    parser.add_argument("--out_dir", required=True, help="Local output dir for processed data")
    parser.add_argument("--push_processed", default="true", help="Upload processed data to HF dataset repo (true/false)")
    args = parser.parse_args()

    dataset_repo = args.dataset_repo
    out_dir = args.out_dir
    push_processed = args.push_processed.lower() == "true"

    # Load dataset directly from HF
    ds = load_dataset(dataset_repo)
    df = ds["train"].to_pandas()

    if "ProdTaken" not in df.columns:
        raise ValueError("Target column 'ProdTaken' not found in dataset.")

    # Clean
    df = clean_and_prepare(df)

    # One-hot encode categoricals (keeping ProdTaken intact)
    y = df["ProdTaken"].astype(int)
    X = df.drop(columns=["ProdTaken"])
    X = pd.get_dummies(X)  # one-hot encoding

    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    os.makedirs(out_dir, exist_ok=True)

    train_path = os.path.join(out_dir, "train.csv")
    test_path = os.path.join(out_dir, "test.csv")

    pd.concat([X_train, y_train.rename("ProdTaken")], axis=1).to_csv(train_path, index=False)
    pd.concat([X_test, y_test.rename("ProdTaken")], axis=1).to_csv(test_path, index=False)

    print(f"✅ Saved: {train_path}")
    print(f"✅ Saved: {test_path}")

    # Upload processed datasets back to HF dataset space
    if push_processed:
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            raise ValueError("HF_TOKEN env var not set. Add it to GitHub Secrets as HF_TOKEN.")
        login(token=hf_token)

        upload_folder(
            folder_path=out_dir,
            repo_id=dataset_repo,
            repo_type="dataset",
            path_in_repo="processed"
        )
        print(f"✅ Uploaded processed train/test to HF dataset repo: {dataset_repo} under /processed")


if __name__ == "__main__":
    main()
