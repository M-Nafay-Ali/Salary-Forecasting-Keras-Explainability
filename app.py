import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import keras
from keras import layers, Sequential
import shap
import plotly.graph_objects as go
import plotly.express as px

# 1. Page Configuration
st.set_page_config(
    page_title="Salary Intelligence AI",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 2. Enhanced High-Contrast CSS Styling (Mobile & Mobile Browser Friendly)
st.markdown("""
<style>
    /* Dark Theme Background */
    .stApp {
        background-color: #0F172A;
        color: #F8FAFC;
    }

    /* Sidebar High-Contrast Text Fix */
    section[data-testid="stSidebar"] {
        background-color: #1E293B !important;
    }
    
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] .stMarkdown,
    section[data-testid="stSidebar"] p {
        color: #F1F5F9 !important;
        font-weight: 600 !important;
    }

    /* Card Containers */
    .glass-card {
        background-color: #1E293B;
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #334155;
        margin-bottom: 16px;
    }

    /* Clear Headers */
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #FFFFFF !important;
        text-align: center;
        margin-bottom: 8px;
    }

    .sub-header {
        font-size: 1rem;
        color: #CBD5E1 !important;
        text-align: center;
        margin-bottom: 0px;
    }

    /* Metric Display */
    .metric-value {
        font-size: 2.5rem;
        font-weight: 800;
        color: #38BDF8;
        margin-top: -5px;
        margin-bottom: 10px;
    }

    /* Action Button */
    .stButton>button {
        background: linear-gradient(135deg, #0284C7 0%, #0369A1 100%);
        color: #FFFFFF;
        font-weight: 700;
        border-radius: 8px;
        border: none;
        padding: 12px 20px;
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# 3. Model Builder Function
def build_model(input_dim):
    model = Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(256, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='linear')
    ])
    return model

# 4. Artifact Loader
@st.cache_resource
def load_artifacts():
    try:
        preprocessor = joblib.load('preprocessor.pkl')
    except Exception as e:
        raise RuntimeError(f"Error loading preprocessor.pkl: {e}")

    try:
        with open('feature_info.json', 'r') as f:
            meta = json.load(f)
    except Exception as e:
        raise RuntimeError(f"Error loading feature_info.json: {e}")

    try:
        input_dim = meta.get('input_dim', 174)
        nn_model = build_model(input_dim)
        nn_model.load_weights('salary_nn_weights.weights.h5')
    except Exception as e:
        raise RuntimeError(f"Error loading salary_nn_weights.weights.h5: {e}")

    return preprocessor, nn_model, meta

try:
    preprocessor, nn_model, meta = load_artifacts()
except Exception as e:
    st.error(f"⚠️ Deployment Artifact Error: {e}")
    st.stop()

# 5. High-Contrast Header Section
st.markdown("""
<div class="glass-card">
    <div class="main-header">💼 Salary Intelligence AI</div>
    <div class="sub-header">Deep Learning Compensation Intelligence Engine Powered by TensorFlow & SHAP</div>
</div>
""", unsafe_allow_html=True)

# 6. Sidebar Inputs with Clear Labels
st.sidebar.markdown("<h2 style='color: #FFFFFF;'>🎛️ Employee Profile</h2>", unsafe_allow_html=True)

input_gender = st.sidebar.selectbox("Select Gender", meta['unique_genders'])
input_education = st.sidebar.selectbox("Select Education Level", meta['unique_education'])
input_job = st.sidebar.selectbox("Select Job Title", meta['unique_job_titles'])

input_age = st.sidebar.slider("Age (Years)", meta['min_age'], meta['max_age'], int(np.median([meta['min_age'], meta['max_age']])))
input_experience = st.sidebar.slider("Experience (Years)", float(meta['min_exp']), float(meta['max_exp']), 5.0, step=0.5)

input_df = pd.DataFrame([{
    'Age': input_age,
    'Gender': input_gender,
    'Education Level': input_education,
    'Job Title': input_job,
    'Years of Experience': input_experience
}])

# 7. Dashboard Layout
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 📊 Selected Input Features")
    st.dataframe(input_df, use_container_width=True, hide_index=True)
    predict_btn = st.button("🚀 Calculate Estimated Salary")
    st.markdown('</div>', unsafe_allow_html=True)

# 8. Prediction & Responsive SHAP Visualization
if predict_btn or 'prediction' in st.session_state:
    processed_input = preprocessor.transform(input_df)
    if hasattr(processed_input, "toarray"):
        processed_input = processed_input.toarray()
        
    predicted_salary = float(nn_model.predict(processed_input, verbose=0)[0][0])
    st.session_state['prediction'] = predicted_salary

    with col2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 🎯 Predicted Annual Salary")
        st.markdown(f'<div class="metric-value">${predicted_salary:,.2f}</div>', unsafe_allow_html=True)
        
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=predicted_salary,
            number={'prefix': "$", 'valueformat': ",.0f", 'font': {'color': '#FFFFFF', 'size': 20}},
            gauge={
                'axis': {'range': [20000, 250000], 'tickcolor': "#94A3B8"},
                'bar': {'color': "#38BDF8"},
                'bgcolor': "rgba(0,0,0,0.3)",
                'steps': [
                    {'range': [20000, 80000], 'color': '#334155'},
                    {'range': [80000, 150000], 'color': '#475569'},
                    {'range': [150000, 250000], 'color': '#64748B'}
                ]
            }
        ))
        fig_gauge.update_layout(
            height=180, 
            margin=dict(l=15, r=15, t=10, b=10),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_gauge, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Expanded & Responsive SHAP Feature Plot
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 🔍 Feature Importance (SHAP Breakdown)")
    
    with st.spinner("Calculating feature contributions..."):
        bg_sample = np.zeros((10, processed_input.shape[1]))
        explainer = shap.DeepExplainer(nn_model, bg_sample)
        shap_vals = explainer.shap_values(processed_input, check_additivity=False)
        
        if isinstance(shap_vals, list):
            shap_matrix = shap_vals[0]
        else:
            shap_matrix = shap_vals
            
        if len(shap_matrix.shape) == 3:
            shap_matrix = shap_matrix.squeeze(-1)

        # Truncate long feature names for readability on smaller screens
        clean_names = [name[:25] + "..." if len(name) > 25 else name for name in meta['clean_feature_names']]

        shap_df = pd.DataFrame({
            'Feature': clean_names,
            'SHAP Impact ($)': shap_matrix[0]
        }).sort_values(by='SHAP Impact ($)', key=abs, ascending=False).head(6)

        fig_shap = px.bar(
            shap_df,
            x='SHAP Impact ($)',
            y='Feature',
            orientation='h',
            color='SHAP Impact ($)',
            color_continuous_scale=['#EF4444', '#94A3B8', '#38BDF8'],
            text_auto=',.0f'
        )
        fig_shap.update_layout(
            yaxis={'categoryorder': 'total ascending', 'tickfont': {'size': 12, 'color': '#F8FAFC'}},
            xaxis={'tickfont': {'color': '#F8FAFC'}},
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#F8FAFC'),
            height=420,  # Increased height to un-cram labels
            coloraxis_showscale=False,
            margin=dict(l=10, r=20, t=10, b=10)
        )
        st.plotly_chart(fig_shap, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
