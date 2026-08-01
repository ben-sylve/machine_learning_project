# Attrition Analysis — Machine Learning Showcase

This repository demonstrates applied machine learning and data analysis skills through an end-to-end Jupyter notebook (Attrition_at_IBM.ipynb) that investigates factors associated with employee attrition using the IBM HR dataset.

The goal of this repo is to highlight practical machine learning abilities including data exploration, preprocessing, feature engineering, model building, evaluation, and clear communication of results.

## What’s included

- Attrition_at_IBM.ipynb — a self-contained analysis notebook that walks through the full workflow.
- WA_Fn-UseC_-HR-Employee-Attrition.csv — the dataset used (not included in this repo by default; add it to run the notebook).

## Key skills demonstrated

- Data exploration and cleaning (pandas): inspecting data shapes, types, missing values, and basic descriptive statistics.
- Feature engineering: creation of derived features (e.g., Annual_Salary) and categorical handling for modeling.
- Exploratory visualizations (matplotlib, seaborn): distributions, comparisons, and parallel coordinate visualizations to understand group differences.
- Unsupervised learning (KMeans, hierarchical clustering): segmenting employees to identify groups with higher attrition risk.
- Supervised learning (logistic regression): modeling attrition and interpreting coefficients to identify important predictors.
- Model evaluation: confusion matrix, precision/recall, ROC/AUC, gains and lift charts.
- Multicollinearity diagnostics (VIF) and model selection (AIC) using statsmodels and dmba utilities.

## High-level findings (from the notebook)

- Dataset size: 1,470 employees.
- Observed attrition: 237 employees (the notebook calculates exact counts and summary statistics).
- The notebook estimates a conservative replacement cost and a high replacement-cost scenario using SHRM guidelines; results provide an estimated cost range for the observed attrition.
- Feature-level insights and a logistic regression model are included to help identify employees at higher risk of leaving.

## Reproduce the analysis

1. Install required packages (recommended in a virtual environment):

```bash
pip install -r requirements.txt
# or install individually:
# pip install numpy pandas matplotlib seaborn scikit-learn statsmodels scipy dmba mord
```

2. Place the dataset file (WA_Fn-UseC_-HR-Employee-Attrition.csv) in the repository root.
3. Open and run the Jupyter notebook:

```bash
jupyter notebook Attrition_at_IBM.ipynb
```

The notebook runs sequentially and includes explanatory text and code cells for each analysis step.

## Notes & next steps

- The notebook is written to be reproducible; consider adding a requirements.txt or environment.yml to pin versions for reproducibility.
- If you want, I can add a short results summary (CSV) and a small script to run the models headlessly and export metrics for CI-friendly checks.

## Contact

If you have questions or want to extend this work, open an issue or contact me (ben-sylve) on GitHub.
