# Student Performance Prediction Using Linear Regression

## 1. Abstract
Predicts a student's final exam score from study hours, attendance percentage, and previous exam score using **Linear Regression**, with a Streamlit web UI. The model is trained on the Kaggle **Student Performance Factors** dataset (6,607 students).

## 2. Dataset (Kaggle)
- Source: https://www.kaggle.com/datasets/lainguyn123/student-performance-factors
- Columns used: `Hours_Studied`, `Attendance`, `Previous_Scores` → target `Exam_Score`
- Place `StudentPerformanceFactors.csv` in the `data/` folder, **or** run `python download_dataset.py` (needs a Kaggle API token), **or** upload the CSV from the app sidebar.
- If no CSV is found, the app falls back to a 200-row synthetic dataset so it still runs.

## 3. Features of the UI
- Sidebar sliders (ranges adapt to the dataset) and **Predict Score** button with result card, grade, progress bar, and comparison to the dataset average
- Model performance tab: R², MAE, RMSE, Actual vs Predicted plot, coefficient chart, correlations
- Dataset tab: table, summary statistics, score distribution, CSV download
- Sidebar CSV uploader; About tab

## 4. Installation and Running
```bash
pip install -r requirements.txt
streamlit run app.py
```
Opens at http://localhost:8501

## 5. Project Structure
```text
Student-Performance-Prediction/
├── app.py
├── download_dataset.py
├── data/
│   └── StudentPerformanceFactors.csv   (you add this)
├── requirements.txt
├── README.md
└── Project_Report.md
```

## 6. Conclusion
Linear Regression predicts student performance from a few simple inputs, and Streamlit turns the model into an interactive application.
