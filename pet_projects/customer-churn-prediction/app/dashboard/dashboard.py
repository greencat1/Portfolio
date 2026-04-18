# dashboard_simple.py
"""
Streamlit Dashboard for Churn Prediction Monitoring

WHAT THIS DASHBOARD DOES:
Provides a visual interface to monitor churn predictions, model performance,
and customer retention metrics. Used by business analysts and managers.

KEY FEATURES:
1. Overall statistics - Total customers, labeling progress
2. Customer predictions - Who will churn, risk levels, monetary impact
3. Labeled customers - Actual churn history, lost revenue
4. Model metrics - Performance evaluation, economic impact
5. Retention recommendations - Actionable insights per risk group

HOW IT WORKS WITH API:
- Connects to FastAPI backend via REST calls
- Uses API key authentication (dashboard role)
- Refreshes data every 60 seconds (cached)
- All data is read-only (no modifications)
"""

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import numpy as np
import os

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

# Must be the first Streamlit command
# Sets browser tab title and page layout
st.set_page_config(
    page_title="Churn Dashboard",  # Browser tab title
    layout="wide"                   # Use full screen width
)

# ============================================================================
# API CONFIGURATION
# ============================================================================

# Get API URL from environment variable (set in Docker)
# Default to localhost for local development
# In Docker, this becomes "http://api:8000"
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")


# ============================================================================
# SESSION STATE (Persistent data across reruns)
# ============================================================================

# Streamlit reruns the script on every interaction
# Session state preserves values between reruns

# Store the API key after user enters it
if "api_key" not in st.session_state:
    st.session_state.api_key = ""


# ============================================================================
# API CALL FUNCTION (with caching)
# ============================================================================

@st.cache_data(ttl=60)
def call_api(endpoint, api_key):
    """
    Make authenticated API call with caching.
    
    WHY CACHING?
    - Prevents excessive API calls on every dashboard interaction
    - Data refreshes every 60 seconds (TTL = Time To Live)
    - Improves performance significantly
    
    TTL=60 means:
    - First call in 60 seconds → hits API
    - Subsequent calls within 60 seconds → returns cached result
    - After 60 seconds → cache expires, calls API again
    
    ARGS:
        endpoint: API endpoint path (e.g., "/label/stats")
        api_key: Valid API key for authentication
    
    RETURNS:
        JSON response from API or error dict
    """
    try:
        response = requests.get(
            f"{API_URL}{endpoint}",
            headers={"X-API-Key": api_key}
        )
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"Status {response.status_code}: {response.text}"}
    except Exception as e:
        return {"error": str(e)}


# ============================================================================
# SIDEBAR - API KEY INPUT
# ============================================================================

with st.sidebar:
    st.title("🔐 Settings")
    
    # Password input hides the key as user types
    api_key_input = st.text_input(
        "API Key:", 
        type="password",
        placeholder="Enter your API key"
    )
    
    # Connect button saves key to session state
    if st.button("Connect"):
        if api_key_input:
            st.session_state.api_key = api_key_input
            st.success("✅ Key saved!")
            st.rerun()  # Refresh dashboard with new key
    
    # Show key preview if connected
    if st.session_state.api_key:
        st.success(f"🔑 Key: {st.session_state.api_key[:4]}...")
    else:
        st.warning("👈 Enter API key")
    
    st.divider()
    st.caption("💡 Need a key? Contact your administrator")

# ============================================================================
# AUTHORIZATION CHECK
# ============================================================================

# Don't proceed without API key
if not st.session_state.api_key:
    st.warning("👈 Enter API key in the sidebar")
    st.stop()


# ============================================================================
# HEADER
# ============================================================================

st.title("📊 Churn Prediction Dashboard")
st.caption(f"API: {API_URL} | Key: {st.session_state.api_key[:4]}...")


# ============================================================================
# LOAD ALL DATA
# ============================================================================

with st.spinner("Loading data..."):
    # Four parallel API calls (executed sequentially)
    stats = call_api("/label/stats", st.session_state.api_key)           # Overall counts
    unlabeled = call_api("/label/unlabeled/list", st.session_state.api_key)  # Predictions
    labeled = call_api("/label/labeled/list", st.session_state.api_key)      # Actual labels
    metrics = call_api("/models/full_churn_pipeline_cloud.pkl/metrics", st.session_state.api_key)  # Model performance


# ============================================================================
# SECTION 1: OVERALL STATISTICS
# ============================================================================

st.header("📈 Overall Statistics")

if "error" not in stats:
    total_customers = stats.get('total_records', 0)
    labeled_count = stats.get('labeled_records', 0)
    churn_yes = stats.get('churn_yes', 0)
    
    # Display 4 key metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📊 Total Customers", total_customers)
    col2.metric("🏷️ Labeled", labeled_count)
    col3.metric("⚠️ Churn = Yes", churn_yes)
    col4.metric("✅ Churn = No", stats.get('churn_no', 0))
    
    # Progress bar shows labeling completion
    progress = (labeled_count / total_customers * 100) if total_customers > 0 else 0
    st.progress(progress / 100)
    st.caption(f"Labeling progress: {progress:.1f}%")
    
else:
    st.error(f"Error: {stats['error']}")

st.divider()


# ============================================================================
# SECTION 2: CUSTOMERS WITH PREDICTIONS (Unlabeled)
# ============================================================================
# Shows model predictions before human labeling
# This is the "actionable" section - who to contact
# ============================================================================

st.header("🎯 Customers with Predictions")

if "error" not in unlabeled and unlabeled.get('customers'):
    df = pd.DataFrame(unlabeled['customers'])
    
    # ========================================
    # MONETARY METRICS (if data available)
    # ========================================
    if 'MonthlyCharges' not in df.columns:
        st.warning("⚠️ MonthlyCharges not available for monetary calculations")
    else:
        # Calculate revenue at risk
        total_mrr = df['MonthlyCharges'].sum()                    # Total monthly revenue
        mrr_at_risk = df[df['probability'] > 0.5]['MonthlyCharges'].sum()   # >50% risk
        high_risk_mrr = df[df['probability'] > 0.7]['MonthlyCharges'].sum() # >70% risk
        
        st.subheader("💰 Monetary Metrics")
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("💵 Current MRR", f"${total_mrr:,.0f}/month", 
                     help="Monthly Recurring Revenue from all customers")
        col_b.metric("⚠️ MRR at Risk (>50%)", f"${mrr_at_risk:,.0f}/month",
                     help="Revenue from customers with churn probability >50%")
        col_c.metric("🔴 High Risk MRR (>70%)", f"${high_risk_mrr:,.0f}/month",
                     help="Revenue from customers with churn probability >70%")
    
    # ========================================
    # RISK CATEGORIZATION
    # ========================================
    # Thresholds: High >70%, Medium 40-70%, Low <40%
    df['risk'] = df['probability'].apply(
        lambda x: '🔴 High (>70%)' if x > 0.7 else ('🟡 Medium (40-70%)' if x > 0.4 else '🟢 Low (<40%)')
    )
    
    # Convert numeric prediction to readable text
    df['prediction_text'] = df['prediction'].apply(lambda x: '⚠️ Will Churn' if x == 1 else '✅ Will Stay')
    
    st.caption(f"📋 Total customers with predictions: {len(df)}")
    
    # ========================================
    # PREDICTION STATISTICS
    # ========================================
    pred_yes = len(df[df['prediction'] == 1])
    pred_no = len(df[df['prediction'] == 0])
    
    col1, col2 = st.columns(2)
    col1.metric("📊 Predicted Churn", f"{pred_yes} customers", delta=f"{pred_yes/len(df)*100:.1f}%")
    col2.metric("📊 Predicted Stay", f"{pred_no} customers", delta=f"{pred_no/len(df)*100:.1f}%")
    
    # ========================================
    # TOP 10 HIGH RISK CUSTOMERS
    # ========================================
    st.subheader("🔴 Top 10 High Risk Customers")
    if 'MonthlyCharges' in df.columns:
        # Calculate potential annual loss: probability × monthly bill × 12
        top_risk = df.nlargest(10, 'probability')[['customerID', 'probability', 'MonthlyCharges', 'prediction_text']]
        top_risk['potential_loss'] = top_risk['probability'] * top_risk['MonthlyCharges'] * 12
        top_risk['probability'] = top_risk['probability'].apply(lambda x: f"{x:.1%}")
        top_risk['potential_loss'] = top_risk['potential_loss'].apply(lambda x: f"${x:,.0f}/year")
        top_risk.columns = ['Customer ID', 'Probability', 'Monthly Bill', 'Prediction', 'Potential Loss/Year']
    else:
        top_risk = df.nlargest(10, 'probability')[['customerID', 'probability', 'prediction_text']]
        top_risk['probability'] = top_risk['probability'].apply(lambda x: f"{x:.1%}")
        top_risk.columns = ['Customer ID', 'Churn Probability', 'Prediction']
    st.dataframe(top_risk, use_container_width=True)
    
    # ========================================
    # EXPANDABLE: ALL CUSTOMERS
    # ========================================
    with st.expander("📋 View all customers with predictions"):
        if 'MonthlyCharges' in df.columns:
            show_df = df[['customerID', 'probability', 'MonthlyCharges', 'tenure', 'prediction_text']].copy()
            show_df['probability'] = show_df['probability'].apply(lambda x: f"{x:.1%}")
            show_df.columns = ['Customer ID', 'Probability', 'Monthly Bill', 'Tenure (months)', 'Prediction']
        else:
            show_df = df[['customerID', 'probability', 'prediction_text']].copy()
            show_df['probability'] = show_df['probability'].apply(lambda x: f"{x:.1%}")
            show_df.columns = ['Customer ID', 'Churn Probability', 'Prediction']
        st.dataframe(show_df, use_container_width=True)
    
    # ========================================
    # VISUALIZATIONS
    # ========================================
    col_ch1, col_ch2 = st.columns(2)
    
    with col_ch1:
        # Histogram of probability distribution
        fig = px.histogram(df, x='probability', nbins=20, 
                           title="Churn Probability Distribution",
                           labels={'probability': 'Churn Probability'})
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col_ch2:
        # Scatter plot: risk vs monthly bill
        if 'MonthlyCharges' in df.columns:
            fig2 = px.scatter(df, x='MonthlyCharges', y='probability', color='risk',
                              title="Churn Risk vs Monthly Bill",
                              labels={'MonthlyCharges': 'Monthly Bill ($)', 'probability': 'Churn Probability'})
            fig2.update_layout(height=400)
            st.plotly_chart(fig2, use_container_width=True)
    
    # Pie chart of predictions
    pred_counts = df['prediction_text'].value_counts()
    fig_pie = px.pie(values=pred_counts.values, names=pred_counts.index, title="Model Predictions")
    fig_pie.update_layout(height=400)
    st.plotly_chart(fig_pie, use_container_width=True)
    
    # Educational expander
    with st.expander("📖 What do these numbers mean?"):
        st.markdown("""
        **💰 MRR (Monthly Recurring Revenue)** - Monthly revenue from all customers.
        
        **⚠️ MRR at Risk** - Revenue from customers with churn probability >50%.
        
        **🔴 High Risk MRR** - Revenue from customers with churn probability >70%.
        
        **💸 Potential Loss** - Annual revenue that could be lost if the customer churns.
        """)
    
else:
    if "error" in unlabeled:
        st.error(f"Error: {unlabeled['error']}")
    else:
        st.info("No data. First make a POST /predict/batch request")

st.divider()


# ============================================================================
# SECTION 3: LABELED CUSTOMERS (Actual Churn)
# ============================================================================
# Shows ground truth data - what actually happened
# Used for understanding real churn patterns
# ============================================================================

st.header("✅ Labeled Customers (Actual Churn)")

if "error" not in labeled and labeled.get('customers'):
    df_labeled = pd.DataFrame(labeled['customers'])
    
    st.caption(f"📋 Total labeled: {len(df_labeled)} customers")
    
    # ========================================
    # CHURN STATISTICS WITH MONEY
    # ========================================
    churned = df_labeled[df_labeled['Churn'] == 'Yes']
    stayed = df_labeled[df_labeled['Churn'] == 'No']
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("⚠️ Actually Churned", len(churned), delta=f"{len(churned)/len(df_labeled)*100:.1f}%")
    col2.metric("✅ Actually Stayed", len(stayed), delta=f"{len(stayed)/len(df_labeled)*100:.1f}%")
    
    if 'MonthlyCharges' in df_labeled.columns:
        lost_mrr = churned['MonthlyCharges'].sum()
        lost_annual = lost_mrr * 12
        col3.metric("💰 Lost MRR", f"${lost_mrr:,.0f}/month",
                    help="Monthly revenue lost due to churn")
        col4.metric("📉 Lost Annual Revenue", f"${lost_annual:,.0f}",
                    help="Annual revenue lost due to churn")
    
    # ========================================
    # CHURN BY TENURE CHART
    # ========================================
    # Groups: <6 months, 6-12 months, 1-2 years, 2+ years
    if 'tenure' in df_labeled.columns and len(churned) > 0:
        df_labeled['tenure_group'] = pd.cut(
            df_labeled['tenure'], 
            bins=[0, 6, 12, 24, 100], 
            labels=['<6 months', '6-12 months', '1-2 years', '2+ years']
        )
        churn_by_tenure = df_labeled[df_labeled['Churn'] == 'Yes'].groupby('tenure_group').size()
        if len(churn_by_tenure) > 0:
            fig3 = px.bar(x=churn_by_tenure.index, y=churn_by_tenure.values, 
                          title="Churn by Tenure",
                          labels={'x': 'Tenure', 'y': 'Number of Churned Customers'})
            fig3.update_layout(height=400)
            st.plotly_chart(fig3, use_container_width=True)
    
    # ========================================
    # EXPANDABLE: ALL LABELED DATA
    # ========================================
    with st.expander("📋 View all labeled data"):
        if 'MonthlyCharges' in df_labeled.columns:
            show_df = df_labeled[['customerID', 'Churn', 'MonthlyCharges', 'tenure']].copy()
            show_df.columns = ['Customer ID', 'Churn', 'Monthly Bill', 'Tenure (months)']
        else:
            show_df = df_labeled[['customerID', 'Churn']].copy()
        st.dataframe(show_df, use_container_width=True)
    
    with st.expander("📖 What do these numbers mean?"):
        st.markdown("""
        **⚠️ Actually Churned** - Customers who actually left (Churn = Yes).
        
        **💰 Lost MRR** - Monthly revenue lost due to customer churn.
        
        **📉 Lost Annual Revenue** - Lost MRR multiplied by 12 months.
        """)
    
else:
    if "error" in labeled:
        st.error(f"Error: {labeled['error']}")
    else:
        st.info("No labeled data available")

st.divider()


# ============================================================================
# SECTION 4: MODEL METRICS
# ============================================================================
# Shows how well the model performs
# Critical for trust and improvement decisions
# ============================================================================

st.header("🤖 Model Metrics")

if metrics and "error" not in metrics:
    if metrics.get('status') == 'success':
        m = metrics.get('metrics', {})
        
        # ========================================
        # CORE METRICS
        # ========================================
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("📊 Accuracy", f"{m.get('accuracy', 0):.2%}",
                    help="Proportion of correct predictions")
        col2.metric("🎯 Precision", f"{m.get('precision', 0):.2%}",
                    help="Of predicted churns, how many actually churned")
        col3.metric("🔍 Recall", f"{m.get('recall', 0):.2%}",
                    help="Of actual churns, how many were found")
        col4.metric("📈 F1-Score", f"{m.get('f1_score', 0):.2%}",
                    help="Harmonic mean of precision and recall")
        
        if 'roc_auc' in m:
            st.metric("📊 ROC-AUC", f"{m['roc_auc']:.2%}",
                      help="Model's ability to separate classes")
        
        st.caption(f"📊 Test samples: {m.get('test_samples', 'N/A')} | Churn Yes: {m.get('churn_yes', 'N/A')} | Churn No: {m.get('churn_no', 'N/A')}")
        
        # ========================================
        # ECONOMIC IMPACT CALCULATION
        # ========================================
        st.subheader("💰 Model Economic Impact")
        
        recall = m.get('recall', 0)
        
        # Get actual lost MRR from labeled data (if available)
        if "error" not in labeled and labeled.get('customers') and 'MonthlyCharges' in df_labeled.columns:
            actual_lost_mrr = churned['MonthlyCharges'].sum()
        else:
            # Fallback estimate for demo
            actual_lost_mrr = 50000
        
        save_rate = 0.35  # Business assumption: can save 35% of identified churners
        
        current_saved = actual_lost_mrr * recall * save_rate
        improved_recall = min(recall * 1.3, 0.95)  # 30% improvement, max 95%
        potential_saved = actual_lost_mrr * improved_recall * save_rate
        
        col_e1, col_e2, col_e3 = st.columns(3)
        col_e1.metric("💸 Actually Lost MRR", f"${actual_lost_mrr:,.0f}/month",
                      help="MRR of customers who already churned")
        col_e2.metric("📊 Current Model Saves", f"${current_saved:,.0f}/month",
                      help=f"Model finds {recall:.0%} of churners, saves {save_rate:.0%} of them")
        col_e3.metric("🚀 Improvement Potential", f"${potential_saved - current_saved:,.0f}/month",
                      help=f"Improving recall by {(improved_recall-recall)*100:.0f}%")
        
        # ========================================
        # QUALITY ASSESSMENT
        # ========================================
        st.subheader("📊 Model Quality Assessment")
        if recall > 0.8:
            st.success(f"✅ Excellent! Model finds {recall:.1%} of churning customers")
        elif recall > 0.6:
            st.warning(f"⚠️ Good, but can be improved. Model finds {recall:.1%} of churning customers")
        else:
            st.error(f"❌ Model needs improvement. Only finds {recall:.1%} of churning customers")
        
        # Educational expander
        with st.expander("📖 What do model metrics mean?"):
            st.markdown("""
            **Accuracy** - Proportion of correct predictions.  
            *Example: If model correctly predicts 80 out of 100 customers, Accuracy = 80%*
            
            **Precision** - Of all customers predicted to churn, how many actually churned.  
            *Low precision → many false alarms.*
            
            **Recall** - Of all customers who actually churned, how many were found.  
            *Low recall → model misses many churning customers.*
            
            **F1-Score** - Harmonic mean of Precision and Recall.  
            *Good balance between both metrics.*
            
            **ROC-AUC** - Model's ability to separate classes.  
            *0.5 = random, 0.7+ = good, 0.8+ = excellent.*
            """)
        
    else:
        st.warning("Metrics failed to load")
else:
    st.info("📊 Model metrics not found")

st.divider()


# ============================================================================
# SECTION 5: RETENTION RECOMMENDATIONS
# ============================================================================
# Actionable insights based on risk levels
# This is the "what to do" section for the business team
# ============================================================================

st.header("💡 Retention Recommendations")

if "error" not in unlabeled and unlabeled.get('customers'):
    df = pd.DataFrame(unlabeled['customers'])
    
    # Split by risk level
    high_risk = df[df['probability'] > 0.7]
    medium_risk = df[(df['probability'] >= 0.4) & (df['probability'] <= 0.7)]
    low_risk = df[df['probability'] < 0.4]
    
    col1, col2, col3 = st.columns(3)
    
    # ========================================
    # HIGH RISK (Critical)
    # ========================================
    with col1:
        st.info("**🔴 High Risk (>70%)**")
        st.write(f"{len(high_risk)} customers")
        if 'MonthlyCharges' in df.columns and len(high_risk) > 0:
            high_risk_mrr = high_risk['MonthlyCharges'].sum()
            st.write(f"💰 MRR at risk: ${high_risk_mrr:,.0f}/month")
        if len(high_risk) > 0:
            st.write("→ Call immediately")
            st.write("→ Offer 20-30% discount")
            st.write("→ Offer bonus services")
            st.write(f"→ IDs: {', '.join(high_risk['customerID'].tolist()[:3])}...")
    
    # ========================================
    # MEDIUM RISK (Monitor)
    # ========================================
    with col2:
        st.warning("**🟡 Medium Risk (40-70%)**")
        st.write(f"{len(medium_risk)} customers")
        if 'MonthlyCharges' in df.columns and len(medium_risk) > 0:
            medium_risk_mrr = medium_risk['MonthlyCharges'].sum()
            st.write(f"💰 MRR at risk: ${medium_risk_mrr:,.0f}/month")
        if len(medium_risk) > 0:
            st.write("→ Send promotional email")
            st.write("→ Offer upgrade or discount")
            st.write("→ Conduct satisfaction survey")
    
    # ========================================
    # LOW RISK (Maintain)
    # ========================================
    with col3:
        st.success("**🟢 Low Risk (<40%)**")
        st.write(f"{len(low_risk)} customers")
        st.write("→ Loyalty program")
        st.write("→ Informational emails")
        st.write("→ Monitor situation")
    
    # ========================================
    # PRIORITY ACTIONS (Monetary priority)
    # ========================================
    if len(high_risk) > 0:
        st.subheader("🚨 Urgent Actions")
        
        if 'MonthlyCharges' in df.columns:
            # Sort by highest bill first (most valuable customers)
            high_risk_sorted = high_risk.sort_values('MonthlyCharges', ascending=False)
            st.warning(f"**Need to contact {len(high_risk)} customers immediately.**")
            st.write("**Top 5 by bill amount (priority):**")
            for _, row in high_risk_sorted.head(5).iterrows():
                st.write(f"  • {row['customerID']} — ${row['MonthlyCharges']:.0f}/month ({row['probability']:.0%} risk)")
        else:
            st.warning(f"**Need to contact {len(high_risk)} customers immediately:**")
            for customer_id in high_risk['customerID'].tolist()[:5]:
                st.write(f"  • {customer_id}")

st.divider()


# ============================================================================
# SECTION 6: QUICK ACTIONS
# ============================================================================

st.header("⚡ Quick Actions")

col1, col2 = st.columns(2)

with col1:
    if st.button("🔄 Refresh Data"):
        # Clear all cached data to force fresh API calls
        st.cache_data.clear()
        st.rerun()

with col2:
    with st.expander("📝 How to add labels"):
        st.code('''
curl -X POST "http://127.0.0.1:8000/label/update" \\
  -H "X-API-Key: YOUR_KEY" \\
  -H "Content-Type: application/json" \\
  -d '{"customerID": "7590-VHVEG", "Churn": "Yes"}'
        ''', language='bash')

# ============================================================================
# FOOTER
# ============================================================================

st.divider()
st.caption(f"📅 Last updated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")


# ============================================================================
# BUSINESS VALUE
# ============================================================================
#
# This dashboard enables:
# - Real-time churn monitoring
# - Data-driven retention decisions
# - Model performance tracking
# - Revenue impact quantification
# - Prioritized customer outreach
#
# ============================================================================