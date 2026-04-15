# dashboard.py
import streamlit as st
import pandas as pd
import requests
import plotly.express as px

st.set_page_config(page_title="Churn Dashboard", layout="wide")

# Аутентификация
api_key = st.sidebar.text_input("API Key", type="password")

if api_key:
    headers = {"X-API-Key": api_key}
    
    # Заголовок
    st.title("📊 Customer Churn Prediction Dashboard")
    
    # Колонки с метриками
    col1, col2, col3, col4 = st.columns(4)
    
    # Получаем статистику
    response = requests.get("http://localhost:8000/label/stats", headers=headers)
    stats = response.json()
    
    with col1:
        st.metric("Total Customers", stats["total_records"])
    with col2:
        st.metric("Labeled", stats["labeled_records"])
    with col3:
        st.metric("Unlabeled", stats["unlabeled_records"])
    with col4:
        st.metric("Churn Rate", f"{stats['churn_yes']/stats['labeled_records']*100:.1f}%" if stats['labeled_records'] > 0 else "0%")
    
    # График распределения
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Churn Distribution")
        churn_data = pd.DataFrame({
            'Status': ['Churned', 'Not Churned'],
            'Count': [stats['churn_yes'], stats['churn_no']]
        })
        fig = px.pie(churn_data, values='Count', names='Status', title="Churn vs No Churn")
        st.plotly_chart(fig)
    
    with col2:
        st.subheader("Labeling Progress")
        progress_data = pd.DataFrame({
            'Type': ['Labeled', 'Unlabeled'],
            'Count': [stats['labeled_records'], stats['unlabeled_records']]
        })
        fig = px.bar(progress_data, x='Type', y='Count', title="Labeling Progress")
        st.plotly_chart(fig)
    
    # Список моделей
    st.subheader("🤖 Available Models")
    models_response = requests.get("http://localhost:8000/models", headers=headers)
    if models_response.status_code == 200:
        models = models_response.json()
        for model in models['models']:
            with st.expander(f"{model['name']} {'⭐ ACTIVE' if model['is_active'] else ''}"):
                st.write(f"Size: {model['size_mb']} MB")
                st.write(f"Created: {model['created_at']}")
                if model['metrics']:
                    st.write(f"Accuracy: {model['metrics'].get('accuracy', 'N/A')}")
    
    # Форма для добавления лейбла
    st.subheader("🏷️ Add Churn Label")
    col1, col2 = st.columns(2)
    with col1:
        customer_id = st.text_input("Customer ID")
    with col2:
        churn = st.selectbox("Churn", ["Yes", "No"])
    
    if st.button("Update Label"):
        response = requests.post(
            "http://localhost:8000/label/update",
            json={"customerID": customer_id, "Churn": churn},
            headers=headers
        )
        if response.status_code == 200:
            st.success(f"Label updated for {customer_id}")
        else:
            st.error(f"Error: {response.json().get('detail', 'Unknown error')}")
    
    # Форма для предсказания
    st.subheader("🔮 Predict Single Customer")
    with st.form("prediction_form"):
        customer_id = st.text_input("Customer ID")
        tenure = st.number_input("Tenure (months)", min_value=0, max_value=100)
        monthly_charges = st.number_input("Monthly Charges", min_value=0.0, max_value=200.0)
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        
        if st.form_submit_button("Predict"):
            # Здесь нужно собрать полные данные для предсказания
            # Упрощенно для примера
            st.info(f"Prediction for {customer_id}: Will churn with 73% probability")
else:
    st.warning("Please enter your API Key to continue")