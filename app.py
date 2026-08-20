import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import tensorflow as tf
import shap
import plotly.graph_objects as go
import plotly.express as px

# 1. Page Configuration
st.set_page_config(
    page_title="Salary Intelligence AI",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Advanced Custom CSS Styling (Glassmorphism & High-Tech Theme)
st.markdown("""
<style>
    /* Background Image with Dark Overlay */
    .stApp {
        background: linear-gradient(rgba(10, 15, 30, 0.85), rgba(10, 15, 30, 0.92)), 
                    url("https://images.unsplash.com/photo-1486406146926-c627a92ad1ab?q=80&w=2070&auto=format&fit=crop");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        color: #E0E6ED;
    }

    /* Sidebar Glassmorphism */
    section[data-testid="stSidebar"] {
        background: rgba(16, 22, 42, 0.75) !important;
        backdrop-filter: blur(12px);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }

    /* Cards & Containers */
    .glass-card {
        background: rgba(23, 32, 54, 0.65);
        backdrop-filter: blur(16px);
        border-radius: 16px;
        padding: 24px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        margin-bottom: 20px;
    }

    /* Custom Metric Styling */
    .metric-value {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(90deg, #00F2FE 0%, #4FACFE 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-top: -10px;
    }

    /* Custom Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #00F2FE 0%, #4FACFE 100%);
        color: #0A0F1E;
        font-weight: 700;
        border-radius: 10px;
        border: none;
        padding: 12px 28px;
        transition: all 0.3s ease;
        width: 100%;
        box-shadow: 0 4px 15px rgba(79, 172, 254, 0.4);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(79, 172, 254, 0.6);
        color: #0A0F1E;
    }
</style>
""", unsafe_allow_html=True)

# 3. Load Model and Artifacts
@st.cache_resource
def load_artifacts():
    preprocessor = joblib.load('preprocessor.pkl')
    nn_model = tf.keras.models.load_model('salary_nn_model.keras')
    with open('feature_info.json', 'r') as f:
        meta = json.load(f)
    return preprocessor, nn_model, meta

try:
    preprocessor, nn_model, meta = load_artifacts()
except Exception as e:
    st.error("Error loading deployment artifacts. Make sure `preprocessor.pkl`, `salary_nn_model.keras`, and `feature_info.json` exist.")
    st.stop()

# 4. Header Section
st.markdown("""
<div class="glass-card">
    <h1 style='text-align: center; color: #FFFFFF; font-family: "Inter", sans-serif; margin-bottom: 5px;'>
        💼 Salary Intelligence AI
    </h1>
    <p style='text-align: center; color: #94A3B8; font-size: 1.1rem;'>
        Deep Learning Compensation Intelligence Engine Powered by TensorFlow & SHAP
    </p>
</div>
""", unsafe_allow_html=True)

# 5. Sidebar Navigation / Feature Inputs
st.sidebar.markdown("### 🎛️ Employee Profile")

input_gender = st.sidebar.selectbox("Gender", meta['unique_genders'])
input_education = st.sidebar.selectbox("Education Level", meta['unique_education'])
input_job = st.sidebar.selectbox("Job Title", meta['unique_job_titles'])

input_age = st.sidebar.slider("Age", meta['min_age'], meta['max_age'], int(np.median([meta['min_age'], meta['max_age']])))
input_experience = st.sidebar.slider("Years of Experience", float(meta['min_exp']), float(meta['max_exp']), 5.0, step=0.5)

# Construct Input DataFrame
input_df = pd.DataFrame([{
    'Age': input_age,
    'Gender': input_gender,
    'Education Level': input_education,
    'Job Title': input_job,
    'Years of Experience': input_experience
}])

# 6. Main Dashboard Layout
col1, col2 = st.columns([1, 1.2])

with col1:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 📊 Inputs Summary")
    
    # Display Input Profile
    st.dataframe(
        input_df, 
        use_container_width=True, 
        hide_index=True
    )
    
    predict_btn = st.button("🚀 Predict Target Compensation")
    st.markdown('</div>', unsafe_allow_html=True)

# 7. Prediction & Explainability Logic
if predict_btn or 'prediction' in st.session_state:
    # Process inputs
    processed_input = preprocessor.transform(input_df)
    if hasattr(processed_input, "toarray"):
        processed_input = processed_input.toarray()
        
    predicted_salary = float(nn_model.predict(processed_input, verbose=0)[0][0])
    st.session_state['prediction'] = predicted_salary

    with col2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 🎯 Predicted Annual Salary")
        st.markdown(f'<div class="metric-value">${predicted_salary:,.2f}</div>', unsafe_allow_html=True)
        
        # Interactive Gauge Chart
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=predicted_salary,
            number={'prefix': "$", 'valueformat': ",.0f", 'font': {'color': '#FFFFFF'}},
            gauge={
                'axis': {'range': [20000, 250000], 'tickcolor': "#94A3B8"},
                'bar': {'color': "#00F2FE"},
                'bgcolor': "rgba(0,0,0,0.2)",
                'bordercolor': "rgba(255,255,255,0.1)",
                'steps': [
                    {'range': [20000, 80000], 'color': 'rgba(255, 255, 255, 0.05)'},
                    {'range': [80000, 150000], 'color': 'rgba(255, 255, 255, 0.1)'},
                    {'range': [150000, 250000], 'color': 'rgba(255, 255, 255, 0.15)'}
                ]
            }
        ))
        fig_gauge.update_layout(
            height=200, 
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_gauge, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # SHAP Attribution Section
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 🔍 Local Feature Attribution (SHAP Explanation)")
    
    with st.spinner("Computing real-time SHAP feature contributions..."):
        # Dummy background sample creation for fast local explanation
        bg_sample = np.zeros((10, processed_input.shape[1]))
        explainer = shap.DeepExplainer(nn_model, bg_sample)
        shap_vals = explainer.shap_values(processed_input, check_additivity=False)
        
        if isinstance(shap_vals, list):
            shap_matrix = shap_vals[0]
        else:
            shap_matrix = shap_vals
            
        if len(shap_matrix.shape) == 3:
            shap_matrix = shap_matrix.squeeze(-1)

        # Build Top Feature Impact DataFrame
        shap_df = pd.DataFrame({
            'Feature': meta['clean_feature_names'],
            'SHAP Impact ($)': shap_matrix[0]
        }).sort_values(by='SHAP Impact ($)', key=abs, ascending=False).head(8)

        # Plotly Horizontal Bar Chart for SHAP Values
        fig_shap = px.bar(
            shap_df,
            x='SHAP Impact ($)',
            y='Feature',
            orientation='h',
            color='SHAP Impact ($)',
            color_continuous_scale=['#FF4B4B', '#94A3B8', '#00F2FE'],
            text_auto=',.0f'
        )
        fig_shap.update_layout(
            yaxis={'categoryorder': 'total ascending'},
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#E0E6ED'),
            height=320,
            coloraxis_showscale=False,
            margin=dict(l=10, r=10, t=10, b=10)
        )
        st.plotly_chart(fig_shap, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

