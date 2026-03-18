import streamlit as st
import pandas as pd
import joblib
import xgboost
import shap  # 导入SHAP库
import matplotlib.pyplot as plt

# --- 页面基础配置 ---
st.set_page_config(
    page_title="患者风险预测与解释系统(Alfafa-sepsis-mortality)",
    page_icon="⚕️",
    layout="wide"
)


# --- 模型加载 ---
@st.cache_resource  # 使用缓存，避免每次重载模型
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


# 加载您训练好的XGBoost模型
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

# --- 页面标题 ---
st.title("⚕️ 基于XGBoost的患者风险预测与解释系统(Alfafa-sepsis-mortality")
st.markdown("---")

# --- 用户输入界面 ---
# 只有模型成功加载后才显示应用的其余部分
if xgb_model:
    # 使用Expander将输入控件收纳起来，使界面更紧凑
    with st.expander("点击此处输入/修改患者指标", expanded=True):
        input_data = {}  # 用于存储用户输入的字典

        # 创建一个表单
        with st.form("input_form"):
            st.subheader("数值指标")
            # 为数值指标创建更紧凑的多列布局
            num_cols = st.columns(3)
            for i, feature in enumerate(NUMERIC_FEATURES):
                with num_cols[i % 3]:
                    input_data[feature] = st.number_input(
                        label=f"{feature}",
                        step=1.0,
                        format="%.1f"  # 格式化为一位小数
                    )

            st.markdown("<br>", unsafe_allow_html=True)  # 增加一些间距

            st.subheader("二元指标 (是/否)")
            # 为二元指标创建多列布局
            bin_cols = st.columns(4)
            for i, feature in enumerate(BINARY_FEATURES):
                with bin_cols[i % 4]:
                    value = st.radio(
                        label=f"{feature}",
                        options=['否', '是'],
                        key=f"radio_{feature}",  # 保证每个radio有独立的key
                        horizontal=True
                    )
                    input_data[feature] = 1 if value == '是' else 0

            # 表单的提交按钮
            submitted = st.form_submit_button("执行预测")

    # --- 预测和结果展示 ---
    if submitted:
        st.header("📈 预测结果与个体化解释")

        # 1. 将用户输入转换为DataFrame
        input_df = pd.DataFrame([input_data])
        input_df = input_df[FEATURE_COLUMNS]

        # 2. 使用模型进行预测，获取概率
        prediction_proba = xgb_model.predict_proba(input_df)[:, 1][0]

        # 3. 风险分层
        risk_level, risk_color = "", ""
        if prediction_proba <= 0.0715:
            risk_level, risk_color = "低风险", "green"
        elif 0.0715 < prediction_proba <= 0.4445:
            risk_level, risk_color = "中风险", "orange"
        else:
            risk_level, risk_color = "高风险", "red"

        # 4. 在主页面显示结果
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="预测风险概率", value=f"{prediction_proba:.2%}")
        with col2:
            st.markdown(f"### 风险等级: <font color='{risk_color}'>**{risk_level}**</font>", unsafe_allow_html=True)

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
