# 🧠 AI / Human 文章偵測器 (英文文章專用)

此專案提供一個 **AI / Human 文章偵測器**，可以判斷文章是由 **人工撰寫** 還是 **AI生成**。使用 **sklearn TF-IDF + Logistic Regression** 訓練模型，並提供 Streamlit Web App 介面，支援單篇與批次文章檢測。

**Streamlit App 連結**: [https://aiotdahw5-7114056010.streamlit.app/](https://aiotdahw5-7114056010.streamlit.app/)

---

## 🔹 專案檔案結構

```

AIoTData_Hw5/
├── app.py                          # 部署版：可直接在 Streamlit Cloud 運行
├── app-包含視覺分析(本地端可使用).py   # 本地版：保留 train.csv 可做信心分析
├── model/
│   └── ai_detector.pkl             # 訓練好的 sklearn 模型
├── train.py                        # 訓練模型程式
├── requirements.txt                # 專案相依套件
├── .gitignore
└── README.md

````

---

## 🔹 資料集來源

本專案使用的資料集來自 Kaggle:  
**AI vs Human Text Dataset**  
[https://www.kaggle.com/datasets/shanegerami/ai-vs-human-text](https://www.kaggle.com/datasets/shanegerami/ai-vs-human-text)

- 原資料集約 1.1GB，包含兩個欄位：
  - `text`: 文章內容
  - `generated`: 標籤 (0=Human, 1=AI)

> 注意：部署版 `app.py` 並不包含 CSV，以避免 Streamlit Cloud 部署失敗

---

## 🔹 安裝與環境

建議使用 Python 3.10+，並建立虛擬環境：

```bash
# 建立虛擬環境
python -m venv venv
source venv/bin/activate  # Linux / macOS
venv\Scripts\activate     # Windows

# 安裝相依套件
pip install -r requirements.txt
````

---

## 🔹 使用說明

### 1️⃣ 部署版 (Streamlit Cloud)

```bash
streamlit run app.py
```

* **單篇文章檢測**：直接輸入英文文章文字，按下「檢測單篇文章」
* **批次文章檢測**：上傳純文字檔 (.txt)，每篇文章以換行分隔
* 下載批次檢測結果 CSV

> 注意：本部署版不依賴 CSV，因此信心分布圖僅顯示範例或可選擇隨機生成示例

---

### 2️⃣ 本地版 (完整信心分析)

```bash
streamlit run "app-包含視覺分析(本地端可使用).py"
```

* 可以載入 `train.csv` 進行模型信心分析與統計可視化
* 適合本地端深入分析模型性能
* 需保留 `train.csv` 在 `data/` 目錄下

---

## 🔹 功能特色

* 使用 **TF-IDF + Logistic Regression** 訓練的 sklearn 模型
* **單篇文章檢測**：顯示 AI 機率與預測結果
* **批次文章檢測**：上傳 .txt 批次文章，生成表格與下載 CSV
* **信心分析 (本地版)**：直方圖、KDE 曲線、平均信心與不確定比例
* **Streamlit Cloud 部署**：可直接透過瀏覽器使用，不需本地安裝資料集

---

## 🔹 注意事項

* 本模型對 **英文文章效果最佳**，中文文章可能判斷不準
* 部署版不含 `train.csv`，僅保留模型與程式碼
* 若需要完整信心分析功能，請使用本地版並確保 CSV 可用

---

## 🔹 參考資源

* Kaggle Dataset: [https://www.kaggle.com/datasets/shanegerami/ai-vs-human-text](https://www.kaggle.com/datasets/shanegerami/ai-vs-human-text)
* Streamlit 官方文件: [https://docs.streamlit.io/](https://docs.streamlit.io/)
* sklearn TF-IDF + Logistic Regression 教學

```

```
