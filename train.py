import pandas as pd
import joblib
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# =========================
# 參數設定
# =========================
CSV_PATH = "AI_Human.csv"      # Kaggle 原始檔
SAMPLE_PER_CLASS = 50000       # 每一類抽樣數量（可調）
RANDOM_STATE = 42
MODEL_PATH = "model/ai_detector.pkl"

# =========================
# 讀取資料（只讀需要的欄位）
# =========================
print("讀取資料中...")
df = pd.read_csv(
    CSV_PATH,
    usecols=["text", "generated"]
)

df = df.dropna(subset=["text"])

print("原始資料筆數：")
print(df["generated"].value_counts())

# =========================
# 平衡抽樣
# =========================
print("進行平衡抽樣...")

df_ai = df[df["generated"] == 1].sample(
    n=SAMPLE_PER_CLASS,
    random_state=RANDOM_STATE
)

df_human = df[df["generated"] == 0].sample(
    n=SAMPLE_PER_CLASS,
    random_state=RANDOM_STATE
)

df_sample = pd.concat([df_ai, df_human]).sample(
    frac=1,
    random_state=RANDOM_STATE
)

print("抽樣後資料筆數：")
print(df_sample["generated"].value_counts())
# 抽樣後儲存 CSV
df_sample.to_csv("data/train.csv", index=False)

# =========================
# 切分訓練 / 測試資料
# =========================
X = df_sample["text"]
y = df_sample["generated"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=RANDOM_STATE
)

# =========================
# 建立模型 Pipeline
# =========================
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        stop_words="english"
    )),
    ("clf", LogisticRegression(
        max_iter=1000,
        n_jobs=-1
    ))
])

# =========================
# 模型訓練
# =========================
print("開始訓練模型...")
model.fit(X_train, y_train)

# =========================
# 模型評估
# =========================
print("模型評估結果：")
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, digits=4))

# =========================
# 模型信心分析（基礎統計）
# =========================
probs = model.predict_proba(X_test)[:, 1]

ai_probs = probs[y_test == 1]
human_probs = probs[y_test == 0]

print("\n模型信心分析：")
print(f"AI 類別平均信心: {np.mean(ai_probs):.4f}")
print(f"Human 類別平均信心: {np.mean(human_probs):.4f}")
print(f"不確定區間比例 (0.4~0.6): {((probs > 0.4) & (probs < 0.6)).mean():.4f}")

# =========================
# 儲存模型
# =========================
joblib.dump(model, MODEL_PATH)
print(f"\n模型已儲存至 {MODEL_PATH}")
