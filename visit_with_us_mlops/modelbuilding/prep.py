# for data manipulation
import pandas as pd
import sklearn
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for converting text data in to numerical representation
from sklearn.preprocessing import LabelEncoder
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi

# Define constants for the dataset and output paths
api = HfApi(token=os.getenv("HF_TOKEN"))
DATASET_PATH = "hf://datasets/chandrachurhghosh/tourism-package-prediction/data/raw/tourism.csv"

# Load the dataset directly from the Hugging Face data space
df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

# Perform data cleaning and remove any unnecessary columns
# Drop unique identifier and 'CustomerID' columns (not useful for modeling)
df = df.drop(columns=['Unnamed: 0', 'CustomerID'])

# Convert appropriate float columns to int
for col in ['DurationOfPitch', 'NumberOfFollowups', 'PreferredPropertyStar', 'NumberOfTrips', 'NumberOfChildrenVisiting', 'Age', 'MonthlyIncome']:
    if col in df.columns:
        df[col] = df[col].astype(int)

# Encode categorical columns
df = pd.get_dummies(df, columns=['TypeofContact', 'Occupation', 'Gender', 'ProductPitched', 'MaritalStatus', 'Designation'], drop_first=True)

# Define features (X) and target (y)
X = df.drop('ProdTaken', axis=1)
y = df['ProdTaken']

# Stratified split to preserve class balance
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Save processed train and test datasets locally
processed_path = "visit_with_us_mlops/data/processed"
os.makedirs(processed_path, exist_ok=True)

X_train.to_csv("X_train.csv",index=False)
X_test.to_csv("X_test.csv",index=False)
y_train.to_csv("y_train.csv",index=False)
y_test.to_csv("y_test.csv",index=False)

files = ["X_train.csv","X_test.csv","y_train.csv","y_test.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=processed_path,
        path_in_repo=processed_path.split("/")[-1],  # just the filename
        repo_id="chandrachurhghosh/tourism-mlops-dataset",
        repo_type="dataset",
    )
