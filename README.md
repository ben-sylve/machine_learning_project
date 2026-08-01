# Factors that Lead to Attrition at IBM

An end-to-end machine learning case study identifying which employees are at risk of leaving IBM and why — using cluster analysis to segment employee profiles and logistic regression to quantify the drivers of turnover.

## Business Problem

Employee replacement costs typically run 50%–200% of an employee's annual salary (SHRM). Applied to this dataset's 237 departed employees, that translates to an estimated **$6.8M–$27.2M** in replacement costs. The goal of this analysis is to understand which factors — tenure, department, compensation, and others — are associated with turnover, so at-risk employees can be identified and targeted for retention interventions.

## Dataset

[IBM HR Analytics Employee Attrition & Performance](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset) (`WA_Fn-UseC_-HR-Employee-Attrition.csv`) — 1,470 employees, 35 features covering demographics, compensation, job role, satisfaction, and tenure.

## Approach

The analysis uses two complementary methods:

1. **Cluster Analysis** — Agglomerative (hierarchical) and K-Means clustering segment employees into profiles based on career stage, and attrition rates are compared across profiles.
2. **Logistic Regression** — A classification model quantifies which specific factors drive attrition and scores every employee with a probability of leaving.

### Phase 1: Problem Statement
Quantify the cost of attrition and frame the business question.

### Phase 2: Data Preprocessing and Exploration
- 1,470 rows, 36 columns; no missing values, no meaningful outliers
- Dropped `StandardHours` and `Over18` (constant across all employees, no analytical value)
- Converted categorical fields to ordered categories and created dummy variables
- Standardized numeric features (z-score) ahead of clustering

### Phase 3: Machine Learning Models

**Cluster Analysis**
- Built dendrograms across six linkage methods (Ward, Complete, Average, Weighted, Single, Centroid, Median) and evaluated cluster structure via heatmaps, centroids, and silhouette scores
- Selected **Ward linkage** (3 clusters) as the primary model, cross-validated against **K-Means** — both methods converged on nearly identical employee groupings
- Clusters resolved cleanly into career-stage profiles: **Junior**, **Mid-Career**, and **Senior**, differentiated by age, income, tenure, and total experience
- **Finding:** Junior employees have the highest attrition rate (~20%), roughly double that of Mid-Career and Senior employees — representing ~175–185 at-risk employees

**Logistic Regression**
- Checked multicollinearity via VIF (all values moderate; no features dropped)
- Compared two training strategies: a standard 60/40 train/validation split vs. an oversampled 50/50 split to correct for class imbalance

| Metric | Standard Split | Oversampled |
|---|---|---|
| Accuracy | 85.9% | 74.8% |
| Recall | 36.3% | 67.2% |
| Precision | 67.3% | 35.2% |
| Specificity | 96.3% | 76.3% |
| AUC | 0.84 | 0.81 |

The **oversampled model** was selected as final despite lower accuracy, because missing a true flight risk is costlier than a false alarm — nearly doubling recall (68% vs. 36%) was judged worth the precision trade-off.

### Phase 4: Model Evaluation Summary
Cluster and classification results were cross-checked against each other for consistency (e.g., junior-cluster membership vs. logistic-regression risk scores), reinforcing confidence in the findings.

### Phase 5: Deployment and Insights

**Top drivers of attrition:**

| Variable | Coefficient | Direction |
|---|---|---|
| OverTime | +1.02 | Strongest risk factor |
| Marital Status: Single | +0.77 | Higher risk |
| Education Field: Marketing | +0.52 | Higher risk |
| Department: Sales | +0.40 | Higher risk |
| Business Travel: Non-Travel | −0.47 | Strongest retention factor |
| Gender: Female | −0.22 | Lower risk |
| Environment Satisfaction | −0.19 | Lower risk |

**Key takeaways:**
- **Overtime is the #1 controllable risk factor.**
- Sales and Marketing-background employees show elevated attrition — worth further investigation.
- Reducing unnecessary travel and improving workplace environment satisfaction are practical retention levers.
- ~23% of employees (285) were flagged as at-risk, with Junior-tier employees over-represented — the model output includes a ranked, actionable list of at-risk employees for targeted intervention.

## Tech Stack

- **Data manipulation:** NumPy, pandas
- **Visualization:** Matplotlib, seaborn
- **Machine learning:** scikit-learn (clustering, logistic regression, metrics), SciPy (hierarchical clustering), statsmodels (VIF)
- **Supplementary:** `dmba` (utilities from *Data Mining for Business Analytics*, Shmueli et al.)

## Repository Contents

- `Attrition_at_IBM.ipynb` — main analysis notebook (run cells sequentially to reproduce results)
- `WA_Fn-UseC_-HR-Employee-Attrition.csv` — source dataset

## Skills Demonstrated

- End-to-end analytical workflow: problem framing → EDA → modeling → evaluation → business recommendations
- Unsupervised learning (hierarchical and K-Means clustering, silhouette analysis, dendrogram interpretation)
- Supervised learning (logistic regression, multicollinearity diagnostics via VIF)
- Handling class imbalance via oversampling and making an explicit precision/recall trade-off decision tied to business cost
- Model interpretation and translating coefficients into concrete business recommendations

## References

- Dyerly, R. (2025, January 21). *The myth of replaceability: Preparing for the loss of key employees.* SHRM Executive Network.
- Hees, J. (2015, August 26). *SciPy hierarchical clustering and dendrogram tutorial.* Jörn's Blog.
- Shmueli, G., Bruce, P. C., Yahav, I., Patel, N. R., & Lichtendahl, K. C. (2020). *Data Mining for Business Analytics: Concepts, Techniques, and Applications in Python.* John Wiley & Sons.
