import streamlit as st
import pandas as pd
import joblib
import xgboost
import shap
import matplotlib.pyplot as plt
import numpy as np

# --- 页面基础配置 ---
st.set_page_config(
    page_title="患者风险预测系统",
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

# 特征显示名称（英文，用于SHAP图）
FEATURE_NAMES_EN = {
    'age': 'Age',
    'BMI': 'BMI',
    'gcs': 'GCS',
    'sofa': 'SOFA',
    'septic_shock': 'Septic Shock',
    'cancer': 'Cancer',
    'respiratory_failure': 'Resp Failure',
    'stroke_tia': 'Stroke/TIA',
    'hemoglobin': 'HGB',
    'platelet': 'PLT',
    'wbc': 'wbc',
    'albumin': 'albumin',
    'creatinine': 'creatinine',
    'pt': 'PT',
    'ptt': 'PTT',
    'heartrate': 'heartrate',
    'respiratoryrate': 'respiratoryrate',
    'sbp': 'sbp',
    'temperature': 'temperature',
    'mv': 'mv',
    'vasopressin': 'Vasopressin',
    'azole_antifungal_agents': '唑类抗真菌药',
    'sedative': 'Sedative',
    'vancomycin': 'Vancomycin'
}

# 默认值
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
st.title("⚕️ 基于XGBoost的患者死亡风险预测系统")
st.markdown("---")

# 侧边栏信息
with st.sidebar:
    st.header("📋 系统信息")
    if xgb_model is not None:
        st.success("✅ 模型加载成功")
        st.write(f"XGBoost版本: {xgboost.__version__}")
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
    所有临床决策都应结合患者具体情况。
    """)

# --- 用户输入界面 ---
if xgb_model:
    with st.expander("点击此处输入患者指标", expanded=True):
        input_data = {}
        
        with st.form("input_form"):
            st.subheader("📊 数值型指标")
            
            # 3列布局
            cols = st.columns(3)
            for i, feature in enumerate(NUMERIC_FEATURES):
                with cols[i % 3]:
                    # 为每个特征添加中文说明和单位
                    if feature == 'age':
                        label = "年龄 (岁)"
                        help_text = "患者年龄，单位：岁"
                    elif feature == 'BMI':
                        label = "身体质量指数 (kg/m²)"
                        help_text = "BMI = 体重(kg) / 身高²(m²)"
                    elif feature == 'gcs':
                        label = "格拉斯哥昏迷评分 (3-15)"
                        help_text = "GCS评分，范围3-15分，分数越低意识障碍越重"
                    elif feature == 'sofa':
                        label = "序贯器官衰竭评分 (0-24)"
                        help_text = "SOFA评分，评估器官功能障碍程度"
                    elif feature == 'hemoglobin':
                        label = "血红蛋白 (g/dL)"
                        help_text = "正常范围：男性13-17，女性12-15"
                    elif feature == 'platelet':
                        label = "血小板 (×10⁹/L)"
                        help_text = "正常范围：100-300"
                    elif feature == 'wbc':
                        label = "白细胞 (×10⁹/L)"
                        help_text = "正常范围：4.0-10.0"
                    elif feature == 'albumin':
                        label = "白蛋白 (g/dL)"
                        help_text = "正常范围：3.5-5.0"
                    elif feature == 'creatinine':
                        label = "肌酐 (mg/dL)"
                        help_text = "正常范围：0.6-1.2"
                    elif feature == 'pt':
                        label = "凝血酶原时间 (秒)"
                        help_text = "正常范围：11-13.5秒"
                    elif feature == 'ptt':
                        label = "部分凝血活酶时间 (秒)"
                        help_text = "正常范围：25-35秒"
                    elif feature == 'heartrate':
                        label = "心率 (次/分)"
                        help_text = "正常范围：60-100次/分"
                    elif feature == 'respiratoryrate':
                        label = "呼吸频率 (次/分)"
                        help_text = "正常范围：12-20次/分"
                    elif feature == 'sbp':
                        label = "收缩压 (mmHg)"
                        help_text = "正常范围：90-120 mmHg"
                    elif feature == 'temperature':
                        label = "体温 (℃)"
                        help_text = "正常范围：36.0-37.5℃"
                    else:
                        label = FEATURE_NAMES_EN.get(feature, feature)
                        help_text = ""
                    
                    input_data[feature] = st.number_input(
                        label=label,
                        help=help_text,
                        min_value=0.0,
                        max_value=200.0 if feature in ['age', 'BMI'] else 1000.0,
                        value=float(DEFAULT_VALUES.get(feature, 50.0)),
                        step=0.1,
                        format="%.1f",
                        key=f"num_{feature}"
                    )
            
            st.markdown("---")
            st.subheader("✅ 二分类指标 (是/否)")
            
            # 4列布局
            bin_cols = st.columns(4)
            for i, feature in enumerate(BINARY_FEATURES):
                with bin_cols[i % 4]:
                    # 为二分类特征添加中文说明
                    if feature == 'septic_shock':
                        label = "感染性休克"
                        help_text = "是否存在感染性休克"
                    elif feature == 'cancer':
                        label = "癌症"
                        help_text = "是否患有恶性肿瘤"
                    elif feature == 'respiratory_failure':
                        label = "呼吸衰竭"
                        help_text = "是否存在呼吸衰竭"
                    elif feature == 'stroke_tia':
                        label = "卒中/TIA"
                        help_text = "是否有卒中或短暂性脑缺血发作史"
                    elif feature == 'mv':
                        label = "机械通气"
                        help_text = "是否使用机械通气"
                    elif feature == 'vasopressin':
                        label = "血管加压素"
                        help_text = "是否使用血管加压素"
                    elif feature == 'azole_antifungal_agents':
                        label = "唑类抗真菌药"
                        help_text = "是否使用唑类抗真菌药物"
                    elif feature == 'sedative':
                        label = "镇静剂"
                        help_text = "是否使用镇静剂"
                    elif feature == 'vancomycin':
                        label = "万古霉素"
                        help_text = "是否使用万古霉素"
                    else:
                        label = FEATURE_NAMES_EN.get(feature, feature)
                        help_text = ""
                    
                    value = st.radio(
                        label=label,
                        help=help_text,
                        options=['否', '是'],
                        key=f"bin_{feature}",
                        horizontal=True,
                        index=0
                    )
                    input_data[feature] = 1 if value == '是' else 0
            
            submitted = st.form_submit_button("🔮 预测死亡风险", type="primary", use_container_width=True)
    
    # --- 预测和结果展示 ---
    if submitted:
        st.header("📈 预测结果与个体化解释")
        
        # 创建DataFrame
        input_df = pd.DataFrame([input_data])
        input_df = input_df[FEATURE_COLUMNS]
        
        # 显示输入数据
        with st.expander("查看输入数据详情"):
            display_df = input_df.copy()
            display_df.columns = [FEATURE_NAMES_EN.get(c, c) for c in display_df.columns]
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
            
            # --- SHAP 个体化解释 ---
            st.subheader("📊 个体化预测归因分析 (SHAP)")
            st.markdown("""
            **图例说明**：
            - 🔴 **红色**：增加死亡风险的因素
            - 🔵 **蓝色**：降低死亡风险的因素
            - 条形长度表示影响程度的大小
            """)
            
            try:
                # 创建SHAP解释器
                explainer = shap.TreeExplainer(xgb_model)
                
                # 计算SHAP值
                shap_values = explainer.shap_values(input_df)
                
                # 处理SHAP值（针对二分类）
                if isinstance(shap_values, list):
                    shap_values_for_plot = shap_values[1][0] if len(shap_values) > 1 else shap_values[0][0]
                else:
                    shap_values_for_plot = shap_values[0]
                
                # 获取期望值
                if hasattr(explainer, 'expected_value'):
                    if isinstance(explainer.expected_value, list):
                        expected_value = explainer.expected_value[1] if len(explainer.expected_value) > 1 else explainer.expected_value[0]
                    else:
                        expected_value = explainer.expected_value
                else:
                    expected_value = 0
                
                # 获取特征名称（英文，用于SHAP图）
                feature_names_en = [FEATURE_NAMES_EN.get(f, f) for f in input_df.columns]
                
                # 创建Explanation对象
                shap_exp = shap.Explanation(
                    values=shap_values_for_plot,
                    base_values=expected_value,
                    data=input_df.iloc[0].values,
                    feature_names=feature_names_en
                )
                
                # 可视化选项
                viz_option = st.radio(
                    "选择可视化方式:",
                    ["瀑布图 (Waterfall Plot)", "条形图 (显示所有特征)"],
                    horizontal=True
                )
                
                if viz_option == "瀑布图 (Waterfall Plot)":
                    # 瀑布图
                    fig, ax = plt.subplots(figsize=(14, 8))
                    
                    shap.waterfall_plot(
                        shap_exp, 
                        show=False, 
                        max_display=15  # 显示前15个最重要的特征
                    )
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                    
                    st.caption("注：瀑布图显示前15个最重要的特征")
                    
                else:
                    # 条形图显示所有特征
                    fig, ax = plt.subplots(figsize=(12, 10))
                    
                    # 准备数据
                    plot_df = pd.DataFrame({
                        '特征': feature_names_en,
                        'SHAP值': shap_values_for_plot,
                        '原始值': input_df.iloc[0].values
                    }).sort_values('SHAP值', key=abs, ascending=True)
                    
                    # 设置颜色
                    colors = ['red' if x > 0 else 'blue' for x in plot_df['SHAP值']]
                    
                    # 创建水平条形图
                    y_pos = np.arange(len(plot_df))
                    ax.barh(y_pos, plot_df['SHAP值'], color=colors)
                    ax.set_yticks(y_pos)
                    ax.set_yticklabels(plot_df['特征'])
                    ax.set_xlabel('SHAP值 (对预测的影响)', fontsize=12)
                    ax.set_title('所有特征对预测的贡献', fontsize=14, fontweight='bold')
                    
                    # 添加垂直线在0处
                    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
                    
                    # 添加网格
                    ax.grid(True, axis='x', alpha=0.3)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                
                # 详细的SHAP值表格
                with st.expander("查看详细的SHAP归因值"):
                    detail_df = pd.DataFrame({
                        '特征': feature_names_en,
                        '原始值': input_df.iloc[0].values,
                        'SHAP值': shap_values_for_plot,
                        '影响方向': ['增加风险' if x > 0 else '降低风险' for x in shap_values_for_plot]
                    }).sort_values('SHAP值', key=abs, ascending=False)
                    
                    st.dataframe(
                        detail_df.style.format({
                            '原始值': '{:.2f}',
                            'SHAP值': '{:.4f}'
                        }),
                        use_container_width=True
                    )
                
                # 解释说明
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("基线值", f"{expected_value:.3f}")
                with col2:
                    st.metric("最终预测值", f"{prediction_proba:.3f}")
                
                st.info("""
                **解读方法**：
                - 基线值是模型对所有患者的平均预测
                - 每个特征的SHAP值表示其对预测的贡献
                - 所有SHAP值之和 + 基线值 = 最终预测值的对数几率
                - 红色(正值)增加风险，蓝色(负值)降低风险
                """)
                
            except Exception as e:
                st.warning(f"SHAP详细分析失败，显示基础特征重要性: {e}")
                
                try:
                    # 备选方案：特征重要性条形图
                    if hasattr(xgb_model, 'feature_importances_'):
                        importance_df = pd.DataFrame({
                            '特征': [FEATURE_NAMES_EN.get(f, f) for f in FEATURE_COLUMNS],
                            '重要性': xgb_model.feature_importances_
                        }).sort_values('重要性', ascending=True)
                        
                        fig, ax = plt.subplots(figsize=(12, 8))
                        bars = ax.barh(importance_df['特征'], importance_df['重要性'])
                        ax.set_xlabel('特征重要性', fontsize=12)
                        ax.set_title('XGBoost全局特征重要性', fontsize=14, fontweight='bold')
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close()
                        
                except Exception as e2:
                    st.error(f"备选可视化也失败: {e2}")
            
        except Exception as e:
            st.error(f"预测过程中发生错误: {e}")
            st.exception(e)

else:
    st.error("⚠️ 模型未能加载，应用无法运行。")
    
    # 调试信息
    with st.expander("🔧 调试信息"):
        import os
        st.write(f"当前工作目录: {os.getcwd()}")
        st.write(f"目录内容: {os.listdir('.')}")
        
        if os.path.exists('xgb_model.joblib'):
            st.write(f"模型文件大小: {os.path.getsize('xgb_model.joblib')} 字节")
        else:
            st.write("模型文件不存在")
