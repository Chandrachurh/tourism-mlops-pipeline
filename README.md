
# Visit With Us – Wellness Tourism MLOps Pipeline

## Business Context
"Visit with Us" is a leading travel company introducing a new Wellness Tourism Package.
The objective of this project is to build a scalable and automated MLOps pipeline that
predicts whether a customer is likely to purchase the package before being contacted.

---

## Project Objective
- Predict customer purchase behavior for the Wellness Tourism Package
- Automate the complete ML lifecycle using MLOps best practices
- Enable continuous training, deployment, and CI/CD automation

---

## Tech Stack
- Programming Language: Python
- Machine Learning: Scikit-learn
- Data Handling: Pandas
- Model & Data Hosting: Hugging Face (Datasets, Models, Spaces)
- Web Application: Streamlit
- Containerization: Docker
- CI/CD: GitHub Actions
- Development Environment: Google Colab

---

## Folder Structure

visit_with_us_mlops/
├── data/
│   ├── raw/                Raw datasets from Hugging Face
│   └── processed/          Cleaned train and test datasets
│
├── src/
│   ├── data_preparation.py Data cleaning and preprocessing
│   ├── train.py            Model training and tuning
│   └── evaluate.py         Model evaluation
│
├── deployment/
│   ├── app.py              Streamlit application
│   ├── Dockerfile          Docker configuration
│   └── requirements.txt   Deployment dependencies
│
├── .gitignore
├── README.md
└── pipeline.yml            GitHub Actions workflow

---

## End-to-End Workflow
1. Register dataset on Hugging Face Dataset Hub
2. Load, clean, and preprocess customer data
3. Split data into training and testing sets
4. Train and tune ML models
5. Register the best model to Hugging Face Model Hub
6. Deploy the model using Streamlit on Hugging Face Spaces
7. Automate the workflow using GitHub Actions

---

## Model Deployment
- Model loaded directly from Hugging Face Model Hub
- User inputs collected via Streamlit UI
- Predictions generated in real time
- Application containerized using Docker

---

## CI/CD Automation
- GitHub Actions triggers on every push to main branch
- Automates training, evaluation, and deployment updates

---

## Expected Outcomes
- Improved customer targeting
- Automated and scalable ML workflow
- Production-ready deployment

---

## Author
Chandrachurh Ghosh
