# AI-Based Student Performance & Placement Prediction System

> M.Tech Skill Based Assignment — Data Science & Analytics

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-orange?style=flat-square&logo=scikit-learn)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=flat-square&logo=jupyter)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen?style=flat-square)

---

## Overview

This project builds an end-to-end Machine Learning pipeline to predict student placement outcomes and analyze academic performance patterns. It combines supervised classification, unsupervised clustering, dimensionality reduction, and a rule-based AI recommendation engine — all in a single research-grade Jupyter Notebook.

---

## Results at a Glance

| Model | Accuracy | F1 Score | AUC-ROC | CV Score |
|---|---|---|---|---|
| Logistic Regression | 83.7% | 0.882 | 0.913 | 0.820 |
| **Random Forest** | **90.7%** | **0.931** | **0.962** | **0.891** |

> Random Forest outperforms Logistic Regression by **+7% accuracy** and **+0.049 AUC**.

---

## Project Structure

```
├── AI_Student_Placement_Prediction.ipynb   # Main notebook
├── StudentsPerformance.csv                 # Student scores dataset
├── collegePlace.csv                       # Placement dataset
└── README.md
```

---

## Dataset

| Dataset | Rows | Features | Target |
|---|---|---|---|
| StudentsPerformance.csv | 1000 | 8 | Academic scores |
| placementdata.csv | 215 | 8 | PlacedOrNot (0/1) |

**Placement Dataset Columns:**
`Age` · `Gender` · `Stream` · `Internships` · `CGPA` · `Hostel` · `HistoryOfBacklogs` · `PlacedOrNot`

---

## Notebook Sections

### 1. Library Imports & Global Styling
All dependencies imported with a consistent light-theme plot styling applied globally via `matplotlib.rcParams`.

### 2. Load & Validate Datasets
Both datasets loaded with upfront validation — shape, missing values, and duplicate checks reported automatically.

### 3. Exploratory Data Analysis
- Score distribution plots (Math, Reading, Writing)
- Correlation heatmaps for student scores and placement features
- Class balance bar chart
- CGPA by gender (boxplot)
- CGPA by placement status (violin plot)

### 4. Feature Engineering
```python
# Composite average score
student_df['Average_Score'] = (math + reading + writing) / 3

# 4-level performance tier
student_df['Performance_Level'] = pd.cut(
    student_df['Average_Score'],
    bins=[0, 50, 70, 85, 100],
    labels=['Low', 'Medium', 'High', 'Exceptional']
)
```

### 5. Machine Learning Pipeline
- Label encoding for categorical features
- Stratified train/test split (80/20)
- StandardScaler applied correctly — fit on train, transform on test only
- 5-fold cross-validation on both models

### 6. Model Evaluation
- Accuracy, Precision, Recall, F1, AUC-ROC all reported
- Side-by-side confusion matrices
- Grouped bar chart model comparison with value labels and reference lines

### 7. K-Means Clustering
- Elbow method used to determine optimal k = 3
- Students segmented into: High Performers, Average, At-Risk

### 8. PCA Dimensionality Reduction
- 2D projection of student clusters
- Explained variance ratio reported per component

### 9. AI Career Recommendation Engine
```python
def generate_recommendation(cgpa, etest_score=None):
    if cgpa >= 8.5:   → Data Scientist / AI Engineer / Research Analyst
    elif cgpa >= 7.0: → Data Analyst / Business Analyst / ML Engineer
    elif cgpa >= 5.5: → Junior Developer / QA Engineer
    else:             → Skill Bootcamps / Internship Programs
```

---

## Key Findings

- **CGPA** is the single strongest predictor of placement outcome
- **History of backlogs** significantly reduces placement probability
- **Gender** has minimal impact on placement outcome
- Students naturally form **3 distinct academic cohorts** under K-Means
- Writing and reading scores show **strong positive correlation** (r > 0.95)
- Random Forest with `class_weight='balanced'` handles class imbalance effectively

---

## Tech Stack

| Library | Purpose |
|---|---|
| `pandas` | Data loading, cleaning, feature engineering |
| `numpy` | Numerical operations |
| `matplotlib` | Custom plot styling and visualizations |
| `seaborn` | Statistical visualizations |
| `scikit-learn` | ML models, metrics, preprocessing, PCA |

---

## How to Run

```bash
# 1. Clone the repository
git clone https://github.com/mohitthakur11/AI-Based-Student-Performance-Placement-Prediction-System.git
cd student-placement-prediction

# 2. Install dependencies
pip install numpy pandas matplotlib seaborn scikit-learn jupyter

# 3. Launch Jupyter
jupyter notebook AI_Student_Placement_Prediction.ipynb
```

---

## Future Scope

- XGBoost / LightGBM for further accuracy improvements
- SHAP values for model explainability
- Hyperparameter tuning via GridSearchCV
- Neural network-based recommendation engine
- Real-time student performance dashboard

---

## License

This project is submitted as an academic assignment for M.Tech (Data Science and Analytics).  
For educational use only.
