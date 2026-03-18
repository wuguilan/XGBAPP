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
            
                        # --- SHAP 个体化解释 ---
            st.subheader("📊 个体化预测归因分析")
            st.markdown("""
            **红色**：增加死亡风险的因素  
            **蓝色**：降低死亡风险的因素  
            *横坐标表示该特征对预测结果的影响程度*
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
                
                # 获取特征名称（使用显示名称）
                feature_names_display = [FEATURE_DISPLAY_NAMES.get(f, f) for f in input_df.columns]
                
                # 创建包含特征名称的Series作为data
                data_series = pd.Series(
                    input_df.iloc[0].values, 
                    index=feature_names_display
                )
                
                # 创建Explanation对象 - 修复特征名称显示
                shap_exp = shap.Explanation(
                    values=shap_values_for_plot,
                    base_values=expected_value,
                    data=data_series,  # 使用带名称的Series
                    feature_names=feature_names_display  # 明确指定特征名称
                )
                
                # 提供两种可视化选择
                viz_option = st.radio(
                    "选择可视化方式:",
                    ["Waterfall图", "条形图（显示所有特征）"],
                    horizontal=True
                )
                
                if viz_option == "Waterfall图":
                    # 绘制waterfall图
                    fig, ax = plt.subplots(figsize=(14, 8))
                    
                    # 创建waterfall图
                    shap.waterfall_plot(
                        shap_exp, 
                        show=False, 
                        max_display=15  # 显示前15个最重要的特征
                    )
                    
                    # 调整布局
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                    
                else:
                    # 条形图显示所有特征
                    fig, ax = plt.subplots(figsize=(12, 10))
                    
                    # 创建DataFrame用于绘图
                    plot_df = pd.DataFrame({
                        '特征': feature_names_display,
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
                    
                    # 添加网格线
                    ax.grid(True, axis='x', alpha=0.3)
                    
                    # 添加数值标签
                    for i, (val, color) in enumerate(zip(plot_df['SHAP值'], colors)):
                        if val > 0:
                            ax.text(val + 0.01, i, f' {val:.3f}', va='center', fontsize=9)
                        else:
                            ax.text(val - 0.05, i, f'{val:.3f} ', va='center', ha='right', fontsize=9)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                
                # 显示详细的SHAP值表格
                with st.expander("查看详细的SHAP归因值"):
                    # 创建详细的表格
                    detail_df = pd.DataFrame({
                        '特征': feature_names_display,
                        '原始值': input_df.iloc[0].values,
                        'SHAP值': shap_values_for_plot,
                        '影响方向': ['增加风险' if x > 0 else '降低风险' for x in shap_values_for_plot]
                    }).sort_values('SHAP值', key=abs, ascending=False)
                    
                    # 格式化显示
                    st.dataframe(
                        detail_df.style.format({
                            '原始值': '{:.2f}',
                            'SHAP值': '{:.4f}'
                        }).applymap(
                            lambda x: 'color: red' if isinstance(x, (int, float)) and x > 0 else 'color: blue',
                            subset=['SHAP值']
                        ),
                        use_container_width=True
                    )
                
                # 添加解释说明
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("基线预测值", f"{expected_value:.3f}")
                with col2:
                    st.metric("最终预测概率", f"{prediction_proba:.3f}")
                
                st.info("""
                **如何解读SHAP图**:
                - **红色条**：该特征值推高了预测风险
                - **蓝色条**：该特征值降低了预测风险
                - 条的长度表示影响的大小
                - 所有特征的SHAP值之和 + 基线值 = 最终预测值的log-odds
                """)
                
            except Exception as e:
                st.warning(f"SHAP详细分析失败，使用基础特征重要性: {e}")
                
                try:
                    # 备选方案：特征重要性条形图
                    if hasattr(xgb_model, 'feature_importances_'):
                        importance_df = pd.DataFrame({
                            '特征': [FEATURE_DISPLAY_NAMES.get(f, f) for f in FEATURE_COLUMNS],
                            '重要性': xgb_model.feature_importances_
                        }).sort_values('重要性', ascending=True)
                        
                        fig, ax = plt.subplots(figsize=(12, 8))
                        bars = ax.barh(importance_df['特征'], importance_df['重要性'])
                        ax.set_xlabel('特征重要性', fontsize=12)
                        ax.set_title('XGBoost全局特征重要性', fontsize=14, fontweight='bold')
                        
                        # 添加数值标签
                        for i, (bar, val) in enumerate(zip(bars, importance_df['重要性'])):
                            ax.text(val + 0.01, i, f' {val:.3f}', va='center')
                        
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
