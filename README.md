# Employee Attrition Prediction

A Streamlit web app that predicts whether an employee will leave the company using a **TensorFlow Neural Network** (Binary Classification).

## What This App Does

| Section | What it shows |
|---|---|
| Data Overview | Raw data, attrition distribution, correlation heatmap |
| Build & Train Model | Customize and train your neural network, see training curves |
| Learning Rate Finder | Find the best learning rate for your model |
| Predict Single Employee | Enter employee details → get prediction |

## How to Run Locally

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/employee-attrition-tf.git
cd employee-attrition-tf
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the app
```bash
streamlit run app.py
```

## Project Structure

```
tf_attrition_app/
├── app.py                   ← Streamlit web app (main file)
├── model.py                 ← TensorFlow model logic
├── utils.py                 ← Charts and visualization helpers
├── employee_attrition.csv   ← Dataset
├── requirements.txt         ← Python dependencies
└── README.md                ← This file
```

## Model Architecture

```
Input Layer  → N features (from CSV)
Hidden Layer → Dense neurons (configurable)
Output Layer → 1 neuron, Sigmoid → 0 (Stayed) or 1 (Left)
```

## Key Concepts Used

- **Binary Classification** — predict 0 (stay) or 1 (leave)
- **StandardScaler** — normalize features before training
- **SGD Optimizer** — Stochastic Gradient Descent
- **Learning Rate Scheduler** — find optimal learning rate
- **Loss & Accuracy Curves** — monitor training progress
- **Confusion Matrix** — evaluate predictions

## Dependencies

- Python 3.8+
- tensorflow, streamlit, pandas, numpy
- matplotlib, seaborn, scikit-learn
