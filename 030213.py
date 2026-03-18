import streamlit as st
import pandas as pd
import joblib
import xgboost
import shap
import matplotlib.pyplot as plt
import numpy as np

# --- 页面基础配置 ---
st.set_page_config(
    page_title="Patient Risk Prediction System",
    page_icon="⚕️",
    layout="wide"
)

# --- 模型加载 ---
@st.cache_resource
def load_model(path):
    """Load .joblib model"""
    try:
        model = joblib.load(path)
        return model
    except FileNotFoundError:
        st.error(f"Error: Model file '{path}' not found.")
        st.error("Please ensure 'xgb_model.joblib' is in the same directory.")
        return None
    except Exception as e:
        st.error(f"Unknown error loading model: {e}")
        return None

# Load model
xgb_model = load_model('xgb_model.joblib')

# --- Feature Definitions ---
FEATURE_COLUMNS = [
    'age', 'BMI', 'gcs', 'sofa', 'septic_shock', 'cancer', 'respiratory_failure',
    'stroke_tia', 'hemoglobin', 'platelet', 'wbc', 'albumin', 'creatinine',
    'pt', 'ptt', 'heartrate', 'respiratoryrate', 'sbp', 'temperature', 'mv',
    'vasopressin', 'azole_antifungal_agents', 'sedative', 'vancomycin'
]

NUMERIC_FEATURES = [
    'age', 'BMI', 'gcs', 'sofa', 'hemoglobin', 'platelet', 'wbc', 'albumin',
    'creatinine', 'pt', 'ptt', 'heartrate', 'respiratoryrate', 'sbp', 'temperature'
]

BINARY_FEATURES = [
    'septic_shock', 'cancer', 'respiratory_failure', 'stroke_tia', 'mv',
    'vasopressin', 'azole_antifungal_agents', 'sedative', 'vancomycin'
]

# Feature display names (English only)
FEATURE_DISPLAY_NAMES = {
    'age': 'Age',
    'BMI': 'BMI',
    'gcs': 'GCS',
    'sofa': 'SOFA',
    'septic_shock': 'Septic Shock',
    'cancer': 'Cancer',
    'respiratory_failure': 'Respiratory Failure',
    'stroke_tia': 'Stroke/TIA',
    'hemoglobin': 'Hemoglobin',
    'platelet': 'Platelet',
    'wbc': 'WBC',
    'albumin': 'Albumin',
    'creatinine': 'Creatinine',
    'pt': 'PT',
    'ptt': 'PTT',
    'heartrate': 'Heart Rate',
    'respiratoryrate': 'Respiratory Rate',
    'sbp': 'SBP',
    'temperature': 'Temperature',
    'mv': 'MV',
    'vasopressin': 'Vasopressin',
    'azole_antifungal_agents': 'Antifungals',
    'sedative': 'Sedative',
    'vancomycin': 'Vancomycin'
}

# Default values
DEFAULT_VALUES = {
    'age': 60.0,
    'BMI': 24.0,
    'gcs': 15.0,
    'sofa': 5.0,
    'hemoglobin': 12.0,
    'platelet': 200.0,
    'wbc': 10.0,
    'albumin': 3.5,
    'creatinine': 1.0,
    'pt': 12.0,
    'ptt': 30.0,
    'heartrate': 80.0,
    'respiratoryrate': 18.0,
    'sbp': 120.0,
    'temperature': 36.5
}

# --- Page Title ---
st.title("⚕️ XGBoost-based Patient Risk Prediction System")
st.markdown("---")

# Sidebar information
with st.sidebar:
    st.header("📋 System Info")
    if xgb_model is not None:
        st.success("✅ Model loaded successfully")
    else:
        st.error("❌ Model loading failed")
    
    st.header("📊 Risk Stratification")
    st.markdown("""
    - 🟢 **Low Risk**: ≤ 7.15%
    - 🟡 **Medium Risk**: 7.15% - 44.45%
    - 🔴 **High Risk**: > 44.45%
    """)
    
    st.warning("""
    **Clinical Disclaimer**
    This tool is for reference only and should not replace professional medical judgment.
    """)

# --- User Input Interface ---
if xgb_model:
    with st.expander("Click to enter patient parameters", expanded=True):
        input_data = {}
        
        with st.form("input_form"):
            st.subheader("📊 Numeric Parameters")
            
            # 3-column layout
            cols = st.columns(3)
            for i, feature in enumerate(NUMERIC_FEATURES):
                with cols[i % 3]:
                    display_name = FEATURE_DISPLAY_NAMES.get(feature, feature)
                    input_data[feature] = st.number_input(
                        label=display_name,
                        min_value=0.0,
                        max_value=200.0 if feature in ['age', 'BMI'] else 1000.0,
                        value=float(DEFAULT_VALUES.get(feature, 50.0)),
                        step=0.1,
                        format="%.1f",
                        key=f"num_{feature}"
                    )
            
            st.markdown("---")
            st.subheader("✅ Binary Parameters")
            
            # 4-column layout
            bin_cols = st.columns(4)
            for i, feature in enumerate(BINARY_FEATURES):
                with bin_cols[i % 4]:
                    display_name = FEATURE_DISPLAY_NAMES.get(feature, feature)
                    value = st.radio(
                        label=display_name,
                        options=['No', 'Yes'],
                        key=f"bin_{feature}",
                        horizontal=True,
                        index=0
                    )
                    input_data[feature] = 1 if value == 'Yes' else 0
            
            submitted = st.form_submit_button("🔮 Predict Risk", type="primary", use_container_width=True)
    
    # --- Prediction and Results ---
    if submitted:
        st.header("📈 Prediction Results & Individualized Explanation")
        
        # Create DataFrame
        input_df = pd.DataFrame([input_data])
        input_df = input_df[FEATURE_COLUMNS]
        
        # Show input data
        with st.expander("View input data details"):
            display_df = input_df.copy()
            display_df.columns = [FEATURE_DISPLAY_NAMES.get(c, c) for c in display_df.columns]
            st.dataframe(display_df, use_container_width=True)
        
        try:
            # Predict probability
            prediction_proba = xgb_model.predict_proba(input_df)[:, 1][0]
            
            # Risk stratification
            if prediction_proba <= 0.0715:
                risk_level = "Low Risk"
                risk_color = "green"
                risk_emoji = "🟢"
            elif 0.0715 < prediction_proba <= 0.4445:
                risk_level = "Medium Risk"
                risk_color = "orange"
                risk_emoji = "🟡"
            else:
                risk_level = "High Risk"
                risk_color = "red"
                risk_emoji = "🔴"
            
            # Display main results
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Mortality Risk", f"{prediction_proba:.2%}")
            with col2:
                st.metric("Risk Level", f"{risk_emoji} {risk_level}")
            with col3:
                st.metric("Survival Probability", f"{(1-prediction_proba):.2%}")
            
            # Risk box
            st.markdown(f"""
            <div style='padding: 20px; border-radius: 10px; background-color: {risk_color}20; 
                        border-left: 5px solid {risk_color}; margin: 20px 0;'>
                <h3 style='color: {risk_color}; margin: 0;'>{risk_emoji} {risk_level}</h3>
                <p style='margin: 10px 0 0 0; font-size: 18px;'>
                    Predicted mortality probability: {prediction_proba:.2%}
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # --- SHAP Individualized Explanation ---
            st.subheader("📊 Individualized Prediction Attribution (SHAP)")
            st.markdown("""
            **Legend**:
            - 🔴 **Red**: Increases mortality risk
            - 🔵 **Blue**: Decreases mortality risk
            - Bar length indicates the magnitude of impact
            """)
            
            try:
                # Create SHAP explainer
                explainer = shap.TreeExplainer(xgb_model)
                
                # Calculate SHAP values
                shap_values = explainer.shap_values(input_df)
                
                # Process SHAP values (for binary classification)
                if isinstance(shap_values, list):
                    shap_values_for_plot = shap_values[1][0] if len(shap_values) > 1 else shap_values[0][0]
                else:
                    shap_values_for_plot = shap_values[0]
                
                # Get expected value
                if hasattr(explainer, 'expected_value'):
                    if isinstance(explainer.expected_value, list):
                        expected_value = explainer.expected_value[1] if len(explainer.expected_value) > 1 else explainer.expected_value[0]
                    else:
                        expected_value = explainer.expected_value
                else:
                    expected_value = 0
                
                # Get feature names (English only)
                feature_names_en = [FEATURE_DISPLAY_NAMES.get(f, f) for f in input_df.columns]
                
                # Create Explanation object
                shap_exp = shap.Explanation(
                    values=shap_values_for_plot,
                    base_values=expected_value,
                    data=input_df.iloc[0].values,
                    feature_names=feature_names_en
                )
                
                # Visualization options
                viz_option = st.radio(
                    "Select visualization:",
                    ["Waterfall Plot", "Bar Chart (All Features)"],
                    horizontal=True
                )
                
                if viz_option == "Waterfall Plot":
                    # Waterfall plot
                    fig, ax = plt.subplots(figsize=(14, 8))
                    
                    shap.waterfall_plot(
                        shap_exp, 
                        show=False, 
                        max_display=15  # Show top 15 features
                    )
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                    
                else:
                    # Bar chart with all features
                    fig, ax = plt.subplots(figsize=(12, 10))
                    
                    # Prepare data
                    plot_df = pd.DataFrame({
                        'Feature': feature_names_en,
                        'SHAP Value': shap_values_for_plot,
                        'Original Value': input_df.iloc[0].values
                    }).sort_values('SHAP Value', key=abs, ascending=True)
                    
                    # Set colors
                    colors = ['red' if x > 0 else 'blue' for x in plot_df['SHAP Value']]
                    
                    # Create horizontal bar chart
                    y_pos = np.arange(len(plot_df))
                    ax.barh(y_pos, plot_df['SHAP Value'], color=colors)
                    ax.set_yticks(y_pos)
                    ax.set_yticklabels(plot_df['Feature'])
                    ax.set_xlabel('SHAP Value (Impact on Prediction)', fontsize=12)
                    ax.set_title('All Features Contribution', fontsize=14, fontweight='bold')
                    
                    # Add vertical line at 0
                    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
                    
                    # Add grid
                    ax.grid(True, axis='x', alpha=0.3)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                
                # Detailed SHAP values table
                with st.expander("View detailed SHAP values"):
                    detail_df = pd.DataFrame({
                        'Feature': feature_names_en,
                        'Original Value': input_df.iloc[0].values,
                        'SHAP Value': shap_values_for_plot,
                        'Impact': ['Increases Risk' if x > 0 else 'Decreases Risk' for x in shap_values_for_plot]
                    }).sort_values('SHAP Value', key=abs, ascending=False)
                    
                    st.dataframe(
                        detail_df.style.format({
                            'Original Value': '{:.2f}',
                            'SHAP Value': '{:.4f}'
                        }),
                        use_container_width=True
                    )
                
                # Interpretation guide
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Base Value", f"{expected_value:.3f}")
                with col2:
                    st.metric("Final Prediction", f"{prediction_proba:.3f}")
                
                st.info("""
                **How to interpret**:
                - The base value is the average model prediction
                - Each feature's SHAP value shows its contribution
                - Sum of all SHAP values + base value = log-odds of final prediction
                - Positive SHAP (red) pushes prediction higher (increases risk)
                - Negative SHAP (blue) pushes prediction lower (decreases risk)
                """)
                
            except Exception as e:
                st.warning(f"SHAP analysis failed, showing basic feature importance: {e}")
                
                try:
                    # Fallback: feature importance bar chart
                    if hasattr(xgb_model, 'feature_importances_'):
                        importance_df = pd.DataFrame({
                            'Feature': [FEATURE_DISPLAY_NAMES.get(f, f) for f in FEATURE_COLUMNS],
                            'Importance': xgb_model.feature_importances_
                        }).sort_values('Importance', ascending=True)
                        
                        fig, ax = plt.subplots(figsize=(12, 8))
                        bars = ax.barh(importance_df['Feature'], importance_df['Importance'])
                        ax.set_xlabel('Feature Importance', fontsize=12)
                        ax.set_title('XGBoost Global Feature Importance', fontsize=14, fontweight='bold')
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close()
                        
                except Exception as e2:
                    st.error(f"Fallback visualization also failed: {e2}")
            
        except Exception as e:
            st.error(f"Error during prediction: {e}")
            st.exception(e)

else:
    st.error("⚠️ Model failed to load. Application cannot run.")
    
    # Debug information
    with st.expander("🔧 Debug Info"):
        import os
        st.write(f"Current working directory: {os.getcwd()}")
        st.write(f"Directory contents: {os.listdir('.')}")
        
        if os.path.exists('xgb_model.joblib'):
            st.write(f"Model file size: {os.path.getsize('xgb_model.joblib')} bytes")
        else:
            st.write("Model file not found")
        if os.path.exists('xgb_model.joblib'):
            st.write(f"模型文件大小: {os.path.getsize('xgb_model.joblib')} 字节")
        else:
            st.write("模型文件不存在")
