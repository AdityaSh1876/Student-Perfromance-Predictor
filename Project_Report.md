# Student Performance Prediction using Linear Regression

## 1. Project Overview
Student Performance Prediction is a Machine Learning project that predicts a student's expected final score based on study hours, attendance percentage, and previous exam score.

This project uses only one Machine Learning algorithm: **Linear Regression**. The application is built using Python and Streamlit and provides a simple user interface where users can enter student details and get a predicted score.

## 2. Objectives
- Predict a student's expected final exam score.
- Understand how Linear Regression works.
- Display model evaluation metrics.
- Visualize actual scores against predicted scores.
- Build a simple interactive Machine Learning application.

## 3. Algorithm Used
**Linear Regression** is a supervised Machine Learning algorithm used to predict continuous numerical values. It fits the equation:

`Final_Score = b0 + b1*Study_Hours + b2*Attendance + b3*Previous_Score`

It learns the relationship between Study Hours, Attendance Percentage, Previous Exam Score, and the Final Exam Score. Only Linear Regression is used for training and prediction.

## 4. Technologies Used
- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Matplotlib

## 5. Project Structure
```text
Student-Performance-Prediction/
├── app.py
├── requirements.txt
├── README.md
└── Project_Report.md
```
- `app.py`: Main application, Linear Regression model and UI.
- `requirements.txt`: Required Python libraries.
- `README.md`: Project documentation.
- `Project_Report.md`: Detailed project report.

## 6. Methodology
1. Load the Kaggle Student Performance Factors dataset (6,607 students) from `data/`.
2. Select features (Study_Hours, Attendance, Previous_Score) and target (Final_Score).
3. Split data 80% training / 20% testing.
4. Train the Linear Regression model.
5. Evaluate using MAE, RMSE and R².
6. Predict a new student's score from user input in the UI.

## 7. Evaluation Metrics
- **MAE:** average absolute prediction error.
- **RMSE:** error measure that penalises large mistakes.
- **R² Score:** proportion of score variation explained by the model.

## 8. User Interface
The Streamlit UI has four tabs: Prediction, Model Performance, Dataset, and About. Inputs are given through sidebar sliders.

## 9. Conclusion
The project demonstrates a complete Machine Learning workflow, from data creation to deployment in a web interface, using a single algorithm: Linear Regression.

## 10. Future Scope
- Add more Kaggle features (sleep hours, tutoring sessions, etc.).
- Add more features such as sleep hours or assignments.
- Allow CSV upload for batch predictions.
