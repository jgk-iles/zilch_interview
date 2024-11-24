from pathlib import Path

import shap
import streamlit as st
import pickle as pkl
import pandas as pd
import matplotlib.pyplot as plt

# Add models directory to path
root_dir = Path(__file__).resolve().parent.parent
model_dir = root_dir / 'models/'
data_dir = root_dir / 'data/'


def load_model():
    """Load the CatBoost model from a pickle file."""
    model_path = model_dir / 'catboost_model.pkl'
    with open(model_path, 'rb') as file:
        loaded_model = pkl.load(file)
    return loaded_model


def load_customers(dataset="train"):
    """Load the customers dataset."""
    if dataset == "train":
        data_path = data_dir / 'scored' / 'train.pkl'
    elif dataset == "test":
        data_path = data_dir / 'scored' / 'test.pkl'
    elif dataset == "validation":
        data_path = data_dir / 'scored' / 'validation.pkl'
    else:
        raise ValueError("Invalid dataset. Must be one of 'train', 'test', or 'validation'.")
    return pd.read_pickle(data_path)


def compute_shap_values(model, X):
    """Compute SHAP values for the given data using the given model."""
    explainer = shap.TreeExplainer(model)
    explanation = explainer(X)
    return explanation


def plot_prediction_explanation(explanation, customer_data, customer_id):
    """Plot the SHAP values for the prediction."""
    # Get index of the customer
    customer_index = customer_data[customer_data['customer_id'] == customer_id].index[0]
    base_value = explanation[customer_index].base_values

    fig, axs = plt.subplots(1, 1, figsize=(15, 20))
    shap.plots.bar(explanation[customer_index], show=False, ax=axs, max_display=30)
    plt.axvline(
        x=base_value,
        color='red',
        linestyle='--',
        label=f"Base Value: {base_value:.2f}"
    )
    plt.legend()
    st.pyplot(fig=fig)


if __name__ == '__main__':
    model = load_model()

    st.title('Credit Risk Prediction Explainer')

    st.write('This is a simple web app to predict the credit risk of a customer and explain the prediction using SHAP values.')

    # Load the customer data
    dataset = st.sidebar.selectbox('Select a dataset:', ['train', 'test', 'validation'])
    customer_data = load_customers(dataset=dataset)
    customer = st.sidebar.selectbox('Select a customer:', customer_data['customer_id'])

    st.write(f'Customer ID: {customer}')
    st.write(customer_data[customer_data['customer_id'] == customer])
    
    # Get actual and predicted credit scores and compute % error
    predicted_score = customer_data[customer_data['customer_id'] == customer]['credit_score_prediction'].values[0]
    if dataset in ("train", "test"):
        actual_score = customer_data[customer_data['customer_id'] == customer]['credit_score_target'].values[0].astype(int)
        error = abs(predicted_score - actual_score) / actual_score * 100

    # Display metrics
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    with metric_col1:
        st.metric(label='Predicted Credit Score', value=predicted_score)
    if dataset in ("train", "test"):
        with metric_col2:
            st.metric(label='Actual Credit Score', value=actual_score)
        with metric_col3:
            st.metric(label='% Error', value=f'{error:.2f}%')

    # Compute SHAP values
    X = customer_data.drop(columns=['customer_id'])
    explanation = compute_shap_values(model, X)


    st.write('''The following chart explains why the customer received the predicted credit score.
                The base value is the average prediction for the dataset. The bars show how much
                each feature contributed to the prediction. Positive values increase the prediction
                while negative values decrease the prediction.''')
    # Plot SHAP values
    plot_prediction_explanation(explanation, customer_data, customer)
