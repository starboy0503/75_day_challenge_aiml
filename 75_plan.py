from datetime import datetime, timedelta

# === Define daily tasks ===

# Phase 1: ML Fundamentals (Days 1–25)
phase1_tasks = [
    "Python for ML: NumPy, pandas, matplotlib hands-on",
    "Exploratory Data Analysis: Titanic Dataset",
    "Data Preprocessing: Diabetes Dataset",
    "Supervised vs Unsupervised Learning Comparison",
    "Linear Regression: Predict House Prices",
    "Evaluation Metrics (Regression): MAE, MSE, R²",
    "Logistic Regression: Predict Heart Disease",
    "Evaluation Metrics (Classification): Confusion Matrix, ROC",
    "Decision Trees: Classify Iris Dataset",
    "Random Forest: Improve on Titanic or Iris",
    "SVM: Binary Classification on MNIST",
    "KNN: Wine Quality Classifier",
    "Naive Bayes: Spam Detection",
    "k-Means Clustering: Customer Segmentation",
    "PCA: Visualize MNIST in 2D",
    "Cross-Validation & GridSearch: Random Forest Tuning",
    "Feature Engineering: Titanic Dataset",
    "Handling Imbalanced Data: SMOTE with Fraud Data",
    "Time Series Basics: Stock Price Plotting",
    "Time Series Forecasting: ARIMA on Sales",
    "Intro to NLP: Bag of Words on Text",
    "Sentiment Analysis: Movie Review Classifier",
    "Text Vectorization: TF-IDF Application",
    "Mini Project 1: Any Kaggle ML Dataset",
    "Recap + GitHub Upload of ML Phase"
]

# Phase 2: Deep Learning (Days 26–50)
phase2_tasks = [
    "Neural Network from Scratch (NumPy)",
    "TensorFlow Basics: First DL Model",
    "Keras Model API + Fashion MNIST",
    "Activation Functions: ReLU, Sigmoid, Tanh",
    "Loss Functions: MSE, CrossEntropy Implementation",
    "Optimizers: SGD vs Adam Comparison",
    "CNNs Basics: MNIST Digit Classifier",
    "CNNs Advanced: CIFAR-10 Classifier",
    "Data Augmentation: CIFAR or Fashion MNIST",
    "Transfer Learning: Use VGG16 or ResNet",
    "Image Classification Project: Custom Dataset",
    "RNNs Intro: Character-Level RNN",
    "LSTM: Text Generation Task",
    "GRU vs LSTM: Compare Performance",
    "Word Embeddings: Use GloVe for Sentiment",
    "Sequence Classification: Spam with RNN",
    "Transformer Basics: Attention Mechanism",
    "Hugging Face: Use BERT for QA or Classification",
    "NLP Project: Named Entity Recognition",
    "GANs Intro: Generate Handwritten Digits",
    "GANs Advanced: DCGAN with Face Images",
    "Image Generation Project: GAN/Diffusion",
    "Text Classification Project: Custom Text Dataset",
    "Model Evaluation + Metrics Recap",
    "Recap + GitHub Upload of DL Phase"
]

# Phase 3: Advanced + Deployment (Days 51–75)
phase3_tasks = [
    "MLOps Introduction: ML Pipelines Overview",
    "Flask for ML Deployment: REST API",
    "Streamlit: ML Dashboard for EDA",
    "FastAPI: Deploy DL Model as API",
    "Docker Basics: Containerize ML Model",
    "Model Monitoring: Track Metrics with MLflow",
    "Google Colab vs AWS: Cloud Training Basics",
    "AutoML: H2O.ai or Google AutoML Use",
    "Model Explainability: SHAP",
    "Model Explainability: LIME",
    "Real-World Project 1: Image Classifier End-to-End",
    "Real-World Project 2: Fraud Detection + Deployment",
    "Real-World Project 3: NLP Chatbot",
    "AI Ethics: Bias, Fairness, Safety",
    "Papers with Code: Reproduce a Paper",
    "Model Compression: Pruning or Quantization",
    "Edge AI: TinyML Introduction",
    "Resume + GitHub Polishing Day",
    "LinkedIn Post: Share Your Journey",
    "Kaggle Competition Submission",
    "Capstone Planning: Choose a Final Project",
    "Capstone Build Part 1: Data & Prep",
    "Capstone Build Part 2: Model + Tune",
    "Capstone Deployment: Streamlit or Flask",
    "Final Showcase + Blog + GitHub Upload"
]

# === Combine and Generate Plan ===

all_tasks = phase1_tasks + phase2_tasks + phase3_tasks
start_date = datetime.today()
plan = []

for i, task in enumerate(all_tasks):
    day = f"Day {i + 1}"
    date = (start_date + timedelta(days=i)).strftime("%Y-%m-%d")
    plan.append(f"{day} - {date} ➤ {task}")

# Preview first 10 days
for entry in plan[:10]:
    print(entry)
