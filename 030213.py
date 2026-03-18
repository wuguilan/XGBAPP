import streamlit as st
import pandas as pd
import joblib
import xgboost
import shap
import matplotlib.pyplot as plt
import numpy as np

# --- 页面基础配置 ---
st.set_page_config(
    page_title="患者风险预测与解释系统",
    page_icon="⚕️",
    layout="wide"
)

# --- 模型加载 ---
@st.cache_resource
def load_model(path):
    """加载 .joblib 格式的模型"""
    try:
        model = joblib.load(path)
        return model
    except FileNotFoundError:
        st.error(f"错误：模型文件 '{path}' 未找到。")
        st.error("请确保 'xgb_model.joblib' 文件与您的Streamlit应用在同一个目录下。")
        return None
    except Exception as e:
        st.error(f"加载模型时发生未知错误: {e}")
        return None

# 加载模型
xgb_model = load_model('xgb_model.joblib')

# --- 特征定义 ---
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

# 特征显示名称（用于界面）
FEATURE_DISPLAY_NAMES = {
    'age': '年龄',
    'BMI': 'BMI',
    'gcs': 'GCS评分',
    'sofa': 'SOFA评分',
    'septic_shock': '感染性休克',
    'cancer': '癌症',
    'respiratory_failure': '呼吸衰竭',
    'stroke_tia': '卒中/TIA',
    'hemoglobin': '血红蛋白',
    'platelet': '血小板',
    'wbc': '白细胞',
    'albumin': '白蛋白',
    'creatinine': '肌酐',
    'pt': 'PT',
    'ptt': 'PTT',
    'heartrate': '心率',
    'respiratoryrate': '呼吸频率',
    'sbp': '收缩压',
    'temperature': '体温',
    'mv': '机械通气',
    'vasopressin': '血管加压素',
    'azole_antifungal_agents': '唑类抗真菌药',
    'sedative': '镇静剂',
    'vancomycin': '万古霉素'
}

# 默认值（可根据实际情况调整）
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

# --- 页面标题 ---
st.title("⚕️ 基于XGBoost的患者风险预测与解释系统")
st.markdown("---")

# 侧边栏信息
with st.sidebar:
    st.header("📋 系统信息")
    if xgb_model is not None:
        st.success("✅ 模型加载成功")
    else:
        st.error("❌ 模型加载失败")
    
    st.header("📊 风险分层标准")
    st.markdown("""
    - 🟢 **低风险**: ≤ 7.15%
    - 🟡 **中风险**: 7.15% - 44.45%
    - 🔴 **高风险**: > 44.45%
    """)
    
    st.warning("""
    **临床免责声明**
    本工具仅供参考，不能替代专业医疗判断。
    """)

# --- 用户输入界面 ---
if xgb_model:
    with st.expander("点击此处输入/修改患者指标", expanded=True):
        input_data = {}
        
        with st.form("input_form"):
            st.subheader("📊 数值指标")
            
            # 创建3列布局
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
            st.subheader("✅ 二元指标")
            
            # 创建4列布局
            bin_cols = st.columns(4)
            for i, feature in enumerate(BINARY_FEATURES):
                with bin_cols[i % 4]:
                    display_name = FEATURE_DISPLAY_NAMES.get(feature, feature)
                    value = st.radio(
                        label=display_name,
                        options=['否', '是'],
                        key=f"bin_{feature}",
                        horizontal=True,
                        index=0
                    )
                    input_data[feature] = 1 if value == '是' else 0
            
            submitted = st.form_submit_button("🔮 执行风险预测", type="primary", use_container_width=True)
    
    # --- 预测和结果展示 ---
    if submitted:
        st.header("📈 预测结果与个体化解释")
        
        # 创建DataFrame
        input_df = pd.DataFrame([input_data])
        input_df = input_df[FEATURE_COLUMNS]
        
        # 显示输入数据
        with st.expander("查看输入数据详情"):
            display_df = input_df.copy()
            display_df.columns = [FEATURE_DISPLAY_NAMES.get(c, c) for c in display_df.columns]
            st.dataframe(display_df, use_container_width=True)
        
        try:
            # 预测概率
            prediction_proba = xgb_model.predict_proba(input_df)[:, 1][0]
            
            # 风险分层
            if prediction_proba <= 0.0715:
                risk_level = "低风险"
                risk_color = "green"
                risk_emoji = "🟢"
            elif 0.0715 < prediction_proba <= 0.4445:
                risk_level = "中风险"
                risk_color = "orange"
                risk_emoji = "🟡"
            else:
                risk_level = "高风险"
                risk_color = "red"
                risk_emoji = "🔴"
            
            # 显示主要结果
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("死亡风险概率", f"{prediction_proba:.2%}")
            with col2:
                st.metric("风险等级", f"{risk_emoji} {risk_level}")
            with col3:
                st.metric("存活概率", f"{(1-prediction_proba):.2%}")
            
            # 风险提示框
            st.markdown(f"""
            <div style='padding: 20px; border-radius: 10px; background-color: {risk_color}20; 
                        border-left: 5px solid {risk_color}; margin: 20px 0;'>
                <h3 style='color: {risk_color}; margin: 0;'>{risk_emoji} {risk_level}</h3>
                <p style='margin: 10px 0 0 0; font-size: 18px;'>
                    预测死亡概率: {prediction_proba:.2%}
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # 5. 【新增】计算并显示SHAP个体化解释
        st.subheader("个体化预测归因 (SHAP Waterfall)")
        st.markdown(
            "下图解释了每个特征如何将预测概率从基线值（`base value`）推向最终的输出值。"
            "**红色**的特征是增加风险的因素，**蓝色**的特征是降低风险的因素。"
        )

        # 创建SHAP解释器并计算SHAP值
        explainer = shap.TreeExplainer(xgb_model)
        shap_values = explainer.shap_values(input_df)

        shap.plots.waterfall(
            shap.Explanation(
                values=shap_values[0],
                base_values=explainer.expected_value,
                data=input_df.iloc[0],
                feature_names=input_df.columns
            ),
            show=False
        )

        st.pyplot(plt.gcf())
