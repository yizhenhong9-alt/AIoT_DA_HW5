import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from matplotlib import rcParams

# -------------------------------
# 中文字型設定（避免框框）
# -------------------------------
rcParams['font.sans-serif'] = ['Microsoft JhengHei']  # Windows 微軟正黑體
rcParams['axes.unicode_minus'] = False

# -------------------------------
# Streamlit 頁面設定
# -------------------------------
st.set_page_config(page_title="AI / Human 文章偵測器", layout="wide")
st.title("🧠 AI / Human 文章偵測器 (英文文章專用)")
st.markdown("""
此應用使用 **sklearn TF-IDF + Logistic Regression** 模型 (`ai_detector.pkl`) 來判斷文章是 **AI生成** 還是 **Human撰寫**。  

⚠️ **注意事項**：
- 本模型對英文文章效果最佳，請勿上傳中文文章。
- 單篇文章檢測可直接輸入文字。
- 批次檢測可上傳 **純文字檔（.txt）**，每篇文章以換行分隔。
- 下方提供模型信心分析，可視化測試集信心分布。
""")

# -------------------------------
# 載入模型
# -------------------------------
@st.cache_resource
def load_model():
    model = joblib.load("model/ai_detector.pkl")
    return model

model = load_model()

# -------------------------------
# 單篇文章檢測
# -------------------------------
st.subheader("✏️ 單篇文章檢測")
text = st.text_area("請輸入英文文章：", height=200)

if st.button("檢測單篇文章"):
    if text.strip() == "":
        st.warning("請輸入文章內容")
    else:
        pred = model.predict([text])[0]
        prob = model.predict_proba([text])[0][1]

        label = "🤖 AI 生成" if pred == 1 else "🧑 Human 撰寫"
        st.subheader(f"預測結果：{label}")
        st.metric("AI 機率", f"{prob:.2%}")

# -------------------------------
# 批次文章檔案上傳（英文文章）
# -------------------------------
st.subheader("📄 批次文章檢測（上傳 .txt 檔，每篇文章一行）")
uploaded_file = st.file_uploader("上傳純文字檔", type=["txt"])

if uploaded_file is not None:
    content = uploaded_file.read().decode("utf-8")
    articles = [line.strip() for line in content.split("\n") if line.strip() != ""]
    if len(articles) == 0:
        st.warning("檔案中沒有有效文章")
    else:
        preds = model.predict(articles)
        probs = model.predict_proba(articles)[:, 1]

        df_result = pd.DataFrame({
            "文章內容": articles,
            "預測結果": ["AI" if p==1 else "Human" for p in preds],
            "AI 機率": [f"{p:.2%}" for p in probs]
        })

        st.markdown("### 批次檢測結果")
        st.dataframe(df_result)

        # 下載結果 CSV
        csv = df_result.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="下載結果 CSV",
            data=csv,
            file_name="batch_result.csv",
            mime="text/csv"
        )
