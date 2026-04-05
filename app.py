"""
app.py - Streamlit Web App for Employee Attrition Prediction
Using TensorFlow Neural Network (Binary Classification)
Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from model import (
    load_and_prepare_data,
    scale_and_split,
    build_model,
    train_model,
    evaluate_model,
    find_best_learning_rate,
)
from utils import (
    plot_attrition_distribution,
    plot_training_curves,
    plot_confusion_matrix,
    plot_learning_rate_search,
    plot_feature_correlation,
)

# ── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Employee Attrition Predictor",
    layout="wide"
)

st.title("Employee Attrition Prediction")
st.markdown("A **TensorFlow Neural Network** that predicts whether an employee will leave the company.")

# ── Load Data ─────────────────────────────────────────────────
DATA_PATH = "employee_attrition.csv"

@st.cache_data
def load_data():
    return load_and_prepare_data(DATA_PATH)

try:
    df, X, Y = load_data()
except Exception as e:
    st.error(f"Could not load employee_attrition.csv. Make sure it's in the same folder.\nError: {e}")
    st.stop()

# ── Sidebar Navigation ────────────────────────────────────────
st.sidebar.title("Navigation")
section = st.sidebar.radio("Choose a section:", [
    "Data Overview",
    "Build & Train Model",
    "Learning Rate Finder",
    "Predict Single Employee",
])

# ════════════════════════════════════════════════════════════
# SECTION 1 - Data Overview
# ════════════════════════════════════════════════════════════
if section == "Data Overview":
    st.header("Data Overview")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Employees", df.shape[0])
    col2.metric("Features", X.shape[1])
    col3.metric("Left Company", int(Y.sum()))

    st.subheader("First 5 Rows")
    st.dataframe(df.head())

    st.subheader("Attrition Distribution")
    fig = plot_attrition_distribution(df)
    st.pyplot(fig)

    st.subheader("Feature Correlation Heatmap")
    st.markdown("Shows how strongly each feature is related to others. Brighter = stronger relationship.")
    fig2 = plot_feature_correlation(df)
    st.pyplot(fig2)

    st.subheader("Dataset Statistics")
    st.dataframe(df.describe())

# ════════════════════════════════════════════════════════════
# SECTION 2 - Build & Train Model
# ════════════════════════════════════════════════════════════
elif section == "Build & Train Model":
    st.header("Build & Train Neural Network")
    st.markdown("Adjust the hyperparameters below and train your model!")

    st.sidebar.subheader("Model Settings")
    neurons       = st.sidebar.slider("Neurons in hidden layer", 1, 20, 3)
    extra_layer   = st.sidebar.checkbox("Add extra hidden layer", value=False)
    learning_rate = st.sidebar.select_slider(
        "Learning Rate",
        options=[0.0001, 0.001, 0.005, 0.01, 0.05, 0.1],
        value=0.01
    )
    epochs      = st.sidebar.slider("Epochs", 10, 300, 100, step=10)
    batch_size  = st.sidebar.slider("Batch Size", 8, 128, 32, step=8)
    activation  = st.sidebar.selectbox("Output Activation", ['sigmoid', 'tanh', 'relu'])
    test_size   = st.sidebar.slider("Test Set Size", 0.1, 0.4, 0.2, step=0.05)

    # Show model architecture preview
    st.subheader("Model Architecture Preview")
    arch_text = f"""
    Input Layer  → {X.shape[1]} features
    Hidden Layer → {neurons} neurons (Linear activation)
    {"Hidden Layer → " + str(neurons) + " neurons (Linear activation)" if extra_layer else ""}
    Output Layer → 1 neuron ({activation} activation) → 0 (Stayed) or 1 (Left)
    """
    st.code(arch_text)

    if st.button("🚀 Train Model"):
        with st.spinner("Preparing data and training neural network... "):
            try:
                x_train, x_test, y_train, y_test, sc = scale_and_split(X, Y, test_size=test_size)

                model = build_model(
                    neurons=neurons,
                    extra_layer=extra_layer,
                    learning_rate=learning_rate,
                    activation=activation
                )

                history = train_model(model, x_train, y_train, epochs=epochs, batch_size=batch_size)
                acc, report, cm, y_preds, y_probs = evaluate_model(model, x_test, y_test)

                # Save to session state for use in predict section
                st.session_state['model'] = model
                st.session_state['sc'] = sc
                st.session_state['columns'] = X.columns.tolist()

                # Results
                st.success(f"Training complete!")

                col1, col2, col3 = st.columns(3)
                col1.metric("Test Accuracy", f"{acc:.2%}")
                col2.metric("Epochs Trained", epochs)
                col3.metric("Learning Rate", learning_rate)

                st.subheader("Training Curves")
                st.markdown("Loss should go **down** ↓ and Accuracy should go **up** ↑ — that means the model is learning!")
                fig = plot_training_curves(history)
                st.pyplot(fig)

                st.subheader("Confusion Matrix")
                st.markdown("""
                - **Top-left:** Correctly predicted STAYED
                - **Bottom-right:** Correctly predicted LEFT 
                - **Top-right:** Predicted LEFT but actually STAYED
                - **Bottom-left:** Predicted STAYED but actually LEFT
                """)
                fig2 = plot_confusion_matrix(cm)
                st.pyplot(fig2)

                st.subheader("Classification Report")
                report_df = pd.DataFrame(report).T.round(3)
                st.dataframe(report_df)

            except Exception as e:
                st.error(f"Training failed: {e}")

# ════════════════════════════════════════════════════════════
# SECTION 3 - Learning Rate Finder
# ════════════════════════════════════════════════════════════
elif section == "Learning Rate Finder":
    st.header("Learning Rate Finder")
    st.markdown("""
    **What is learning rate?**
    Think of it like the step size when climbing down a hill (finding the lowest loss).
    - Too **small** → takes forever to learn
    - Too **big** → jumps around and never settles
    - **Just right** → learns efficiently!

    This tool tests many learning rates and shows you which one gives the lowest loss.
    """)

    neurons_lr = st.slider("Neurons in hidden layer", 1, 10, 3)

    if st.button("🔍 Find Best Learning Rate"):
        with st.spinner("Testing 100 different learning rates..."):
            try:
                x_train, x_test, y_train, y_test, _ = scale_and_split(X, Y)
                history = find_best_learning_rate(x_train, y_train, neurons=neurons_lr)

                fig = plot_learning_rate_search(history)
                st.pyplot(fig)

                st.info("The best learning rate is usually just **before** the loss starts shooting upward — typically around **0.001**.")

            except Exception as e:
                st.error(f"Learning rate search failed: {e}")

# ════════════════════════════════════════════════════════════
# SECTION 4 - Predict Single Employee
# ════════════════════════════════════════════════════════════
elif section == "Predict Single Employee":
    st.header("Predict: Will This Employee Leave?")

    if 'model' not in st.session_state:
        st.warning("Please go to **Build & Train Model** first and train a model!")
    else:
        st.markdown("Fill in the employee details below and click **Predict**.")

        cols = st.session_state['columns']
        df_sample = df[cols]

        user_input = {}
        col1, col2 = st.columns(2)

        for i, col in enumerate(cols):
            min_val = float(df_sample[col].min())
            max_val = float(df_sample[col].max())
            mean_val = float(df_sample[col].mean())

            if i % 2 == 0:
                user_input[col] = col1.number_input(col, min_value=min_val, max_value=max_val, value=mean_val)
            else:
                user_input[col] = col2.number_input(col, min_value=min_val, max_value=max_val, value=mean_val)

        if st.button("Predict Attrition"):
            try:
                import pandas as pd
                input_df = pd.DataFrame([user_input])
                sc = st.session_state['sc']
                model = st.session_state['model']

                input_scaled = sc.transform(input_df)
                prob = model.predict(input_scaled, verbose=0)[0][0]
                prediction = int(round(prob))

                if prediction == 1:
                    st.error(f"**This employee is likely to LEAVE** (Probability: {prob:.2%})")
                else:
                    st.success(f"**This employee is likely to STAY** (Probability of leaving: {prob:.2%})")

            except Exception as e:
                st.error(f"Prediction failed: {e}")

# ── Footer ────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.markdown("**Individual Project**")
st.sidebar.markdown("Employee Attrition with TensorFlow Neural Network")
