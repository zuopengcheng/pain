import pandas as pd
import statsmodels.api as sm
from patsy import dmatrices
import streamlit as st
import numpy as np

# --- 配置参数 ---
FILE_NAME = "疼痛数据（赋值后）.xlsx"
PC_THRESHOLD = 38

# **【修正】**：使用不含空格的列名作为模型预测因子
PREDICTORS = [
    'Social_support',
    'Positive_emotion_score',
    'Negative_emotion_score',
    'Number_of_pain_sites',
    'Pain_level'
]
OUTCOME = 'PC_Status'

# --- 1. 数据预处理函数 (包含列名清洗) ---
@st.cache_data(show_spinner="正在加载和清洗数据...")
def load_and_preprocess_data(file_path, threshold):
    """加载数据，清洗列名，创建二分类因变量，并准备模型数据。"""
    try:
        df = pd.read_excel(file_path)
    except FileNotFoundError:
        st.error(f"错误：文件 '{file_path}' 未找到。请确保文件已放在脚本所在的目录下。")
        return None

    # **【关键修复 1】**：清洗列名：移除两端空格，并将所有内部空格替换为下划线
    df.columns = df.columns.astype(str).str.strip().str.replace(' ', '_')

    # 1. 创建二分类结局变量：PC_Status
    df[OUTCOME] = (df['PC'] >= threshold).astype(int)

    # 2. 检查关键列是否存在 (使用新的下划线命名)
    missing_cols = [col for col in PREDICTORS + ['PC'] if col not in df.columns]

    if missing_cols:
        st.error(f"数据加载失败。数据中缺少以下关键列：{missing_cols}。请检查Excel文件中的列名。")
        st.info("已加载的实际**清洗后**列名列表：" + str(df.columns.tolist()))
        return None

    # 3. 确保分类变量是整数类型
    df['Number_of_pain_sites'] = df['Number_of_pain_sites'].astype(int)
    df['Pain_level'] = df['Pain_level'].astype(int)

    return df

# --- 2. 模型训练函数 ---
@st.cache_resource(show_spinner="正在训练 Logistic 回归模型...")
def train_logistic_model(data):
    """训练多元Logistic回归模型。"""

    # **【修正】**：使用不含反引号的公式字符串
    formula = (
        f"{OUTCOME} ~ "
        f"{PREDICTORS[0]} + {PREDICTORS[1]} + {PREDICTORS[2]} + "
        f"C({PREDICTORS[3]}) + C({PREDICTORS[4]})"
    )

    y, X = dmatrices(formula, data, return_type="dataframe")

    model = sm.Logit(y, X)
    try:
        results = model.fit(disp=False, maxiter=100, method='ncg')
    except Exception as e:
        st.error(f"模型拟合失败，请检查数据质量。错误：{e}")
        return None, None

    return results, X.columns

# --- 3. Streamlit 动态列线图应用 (Dashboard) ---
def main():
    st.set_page_config(page_title="动态列线图预测器", layout="centered")
    st.title("🩺 疼痛灾难化动态列线图预测器 (Python/Streamlit)")
    st.markdown("---")

    # 1. 加载和训练模型
    data = load_and_preprocess_data(FILE_NAME, PC_THRESHOLD)
    if data is None:
        return

    model_results, feature_names = train_logistic_model(data)
    if model_results is None:
        return

    st.sidebar.header("输入患者数据 (Nomogram 输入)")

    # 2. 用户输入控件 (使用新的下划线列名进行数据引用)

    # 2.1 连续变量输入 (Slider)
    st.sidebar.subheader("情绪与社会支持 (连续变量)")

    # 社会支持
    ss_min, ss_max, ss_median = data['Social_support'].agg(['min', 'max', 'median'])
    social_support = st.sidebar.slider(
        '社会支持分数',
        min_value=float(ss_min), max_value=float(ss_max),
        value=float(ss_median), step=1.0
    )

    # 正面情绪
    pe_min, pe_max, pe_median = data['Positive_emotion_score'].agg(['min', 'max', 'median'])
    positive_emotion = st.sidebar.slider(
        '正面情绪分数',
        min_value=float(pe_min), max_value=float(pe_max),
        value=float(pe_median), step=1.0
    )

    # 负面情绪
    ne_min, ne_max, ne_median = data['Negative_emotion_score'].agg(['min', 'max', 'median'])
    negative_emotion = st.sidebar.slider(
        '负面情绪分数',
        min_value=float(ne_min), max_value=float(ne_max),
        value=float(ne_median), step=1.0
    )

    # 2.2 分类变量输入 (Selectbox)
    st.sidebar.subheader("疼痛相关因素 (分类变量)")

    # 疼痛部位数量
    pain_sites_map = {
        "0 (1个部位)": 0, "1 (2个部位)": 1, "2 (>3个部位)": 2
    }
    pain_sites_input = st.sidebar.selectbox(
        '疼痛部位数量',
        options=list(pain_sites_map.keys()),
        index=0
    )
    pain_sites = pain_sites_map[pain_sites_input]

    # 疼痛程度
    pain_level_map = {
        "0 (轻度)": 0, "1 (中度)": 1, "2 (重度)": 2
    }
    pain_level_input = st.sidebar.selectbox(
        '疼痛程度',
        options=list(pain_level_map.keys()),
        index=1
    )
    pain_level = pain_level_map[pain_level_input]

    # 3. 构造预测数据点 (One-Hot 编码处理)
    input_data = {}

    # 截距项
    input_data['Intercept'] = 1.0

    # 连续变量
    input_data['Social_support'] = social_support
    input_data['Positive_emotion_score'] = positive_emotion
    input_data['Negative_emotion_score'] = negative_emotion

    # 分类变量 (使用新的模型特征名称)
    input_data['C(Number_of_pain_sites)[T.1]'] = 1.0 if pain_sites == 1 else 0.0
    input_data['C(Number_of_pain_sites)[T.2]'] = 1.0 if pain_sites == 2 else 0.0

    input_data['C(Pain_level)[T.1]'] = 1.0 if pain_level == 1 else 0.0
    input_data['C(Pain_level)[T.2]'] = 1.0 if pain_level == 2 else 0.0

    # 匹配训练集特征顺序
    prediction_df = pd.Series(input_data).reindex(feature_names, fill_value=0.0)
    prediction_df = pd.DataFrame(prediction_df).T

    # 4. 进行概率预测
    try:
        predicted_prob = model_results.predict(prediction_df)[0]
    except Exception as e:
        st.error(f"预测计算失败。请检查输入值是否在模型的有效范围内。错误信息: {e}")
        return

    # 5. 结果可视化和展示 (仪表盘)
    st.header("预测结果")

    prob_percent = predicted_prob * 100
    st.metric(
        label="预测疼痛灾难化 (PC) 概率",
        value=f"{prob_percent:.2f} %",
        help="此概率是患者发生临床上显著疼痛灾难化 (PCS ≥ 38) 的风险。"
    )

    # 风险等级提示
    if predicted_prob >= 0.7:
        st.error("🚨 **高风险**：极高风险，建议立即进行针对性心理干预。")
    elif predicted_prob >= 0.3:
        st.warning("⚠️ **中等风险**：建议密切监测，并评估引入认知行为疗法 (CBT) 等干预措施。")
    else:
        st.info("✅ **低风险**：风险较低，继续标准疼痛管理。")

    st.markdown("---")

    # 6. 显示模型系数 (Nomogram 权重参考)
    st.subheader("模型权重 (Nomogram 基础)")
    params_df = model_results.params.reset_index().rename(columns={'index': '变量', 0: 'Logistic回归系数'})
    params_df = params_df[params_df['变量'] != 'Intercept']

    # 清理虚拟变量名，使其更易读
    params_df['变量'] = params_df['变量'].str.replace('C\\((.*?)\\)\\[T\\.(\\d)\\]', '\\1 (水平\\2)', regex=True)

    st.dataframe(params_df)
    st.markdown("列线图是基于这些系数（权重）的相对大小来分配各变量的点数。")


if __name__ == "__main__":
    main()