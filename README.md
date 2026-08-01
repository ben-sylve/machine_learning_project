# Factors that Lead to Attrition at IBM

An end-to-end machine learning case study identifying which employees are at risk of leaving and why — using cluster analysis to segment employee profiles and logistic regression to quantify the drivers of turnover.

## Business Problem

Employee replacement costs typically run 50%–200% of an employee's annual salary (SHRM). Applied to this dataset's 237 departed employees, that translates to an estimated **$6.8M–$27.2M** in replacement costs ($28,723–$114,890 per employee). The goal of this analysis is to understand which factors — tenure, department, compensation, and others — are associated with turnover, so at-risk employees can be identified and targeted for retention interventions.

## Dataset

[IBM HR Analytics Employee Attrition & Performance](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset) (`WA_Fn-UseC_-HR-Employee-Attrition.csv`) — 1,470 employees, 35 columns covering demographics, compensation, job role, satisfaction, and tenure. This is a synthetic dataset created by IBM data scientists for analytics training. The attrition rate is 16.1% (237 of 1,470), which introduces a class imbalance addressed later in the modeling.

## Approach

The analysis uses two complementary methods:

1. **Cluster Analysis** — Agglomerative (hierarchical) and K-Means clustering segment employees into profiles based on career stage, and attrition rates are compared across profiles.
2. **Logistic Regression** — A classification model quantifies which specific factors drive attrition and scores every employee with a probability of leaving.

### Phase 1: Problem Statement
Quantify the cost of attrition and frame the business question.

### Phase 2: Data Preprocessing and Exploration
- 1,470 rows, 35 columns; no missing values, no meaningful outliers
- Dropped `StandardHours` and `Over18` (constant across all employees, no analytical value)
- Converted categorical fields to ordered categories and created dummy variables (`drop_first=True`)
- Standardized numeric features (z-score) ahead of clustering

### Phase 3: Machine Learning Models

**Cluster Analysis** — performed on 14 standardized numeric features (age, compensation, tenure, and experience measures).

- Built truncated dendrograms across seven linkage methods (Ward, Complete, Average, Weighted, Single, Centroid, Median). Single, Centroid, and Median produced chained or single-employee branches and were eliminated; Average and Weighted also showed individual-employee branches
- Compared the two survivors, **Ward** and **Complete**, via silhouette scores (0.1153 vs. 0.1210) and clustered heatmaps. Ward was selected on the strength of cleaner visual cluster separation despite the marginally lower score
- Cross-validated with **K-Means** (k=3, chosen via elbow plot; silhouette 0.1410). The two methods converged on nearly identical groupings
- Clusters resolved into career-stage profiles — **Junior**, **Mid-Career**, and **Senior** — differentiated by age, income, tenure, and total experience

| Metric | Ward Linkage | K-Means |
|---|---|---|
| Silhouette Score | 0.1153 | 0.1410 |
| Junior members / attrition | 859 / 176 (20.5%) | 842 / 175 (20.8%) |
| Mid-Career members / attrition | 416 / 39 (9.4%) | 409 / 44 (10.7%) |
| Senior members / attrition | 195 / 22 (11.3%) | 219 / 18 (8.2%) |

**Finding:** Junior employees churn at roughly twice the rate of the other two segments — about 175 departures — making them the clearest target for retention effort.

**Logistic Regression**
- Checked multicollinearity via VIF (all values below 5, highest being `TotalWorkingYears` at 4.65); no features dropped
- Compared two training strategies: a standard 60/40 train/validation split vs. an oversampled training set balanced 50/50 between leavers and stayers

| Metric | Standard Split | Oversampled |
|---|---|---|
| Accuracy | 85.9% | 74.8% |
| Recall | 36.3% | 67.2% |
| Precision | 67.3% | 35.2% |
| Specificity | 96.3% | 76.3% |
| AUC | 0.84 | 0.81 |

The **oversampled model** was selected as final despite lower headline accuracy. The standard model was excellent at confirming who would stay but caught barely a third of actual leavers. Since a missed flight risk costs far more than a false alarm, nearly doubling recall was worth the precision trade-off.

### Phase 4: Model Evaluation Summary
Documents the model-selection rationale on both tracks: why Ward was chosen over Complete linkage and how K-Means independently corroborated it, and why the oversampled regression was preferred over the higher-accuracy standard split.

### Phase 5: Deployment and Insights

**Top drivers of attrition (final model coefficients):**

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
- **Overtime is the #1 risk factor — and it is controllable.** Workload balancing is the highest-leverage intervention available.
- Reducing unnecessary travel and improving environment satisfaction are the other two practical, controllable levers.
- Sales and Marketing-background employees show elevated attrition; the model flags the pattern but not the cause, so this warrants qualitative follow-up before acting.
- Applying a 0.5 probability threshold to the 1,233 currently active employees flags **285 (23%) as at-risk**, again concentrated in the Junior tier. The notebook outputs a ranked list of these employees with their contributing factors — an actionable target list for HR intervention.

## Tech Stack

- **Data manipulation:** NumPy, pandas
- **Visualization:** Matplotlib, seaborn
- **Machine learning:** scikit-learn (clustering, logistic regression, metrics), SciPy (hierarchical clustering), statsmodels (VIF)
- **Supplementary:** `dmba` (utilities from *Data Mining for Business Analytics*, Shmueli et al.)

## Repository Contents

- `Attrition_at_IBM.ipynb` — main analysis notebook

## Running the Analysis

Download the attrition dataset from the website provided. Name it WA_Fn-UseC_-HR-Employee-Attrition.csv. Save it in the same directory as the python notebook.

```bash
pip install numpy pandas matplotlib seaborn scikit-learn scipy statsmodels dmba
jupyter notebook Attrition_at_IBM.ipynb
```

Run cells sequentially to reproduce all results. Random seeds are fixed (`random_state=0` for K-Means, `random_state=1` for train/test splits), so output is deterministic.

## Skills Demonstrated

- End-to-end analytical workflow: problem framing → EDA → modeling → evaluation → business recommendations
- Unsupervised learning: hierarchical and K-Means clustering, linkage method comparison, dendrogram and heatmap interpretation, silhouette analysis
- Supervised learning: logistic regression, multicollinearity diagnostics, ROC/AUC and confusion matrix evaluation
- Handling class imbalance via stratified oversampling, with an explicit precision/recall trade-off justified by business cost rather than by accuracy alone
- Cross-validating unsupervised findings against a second method before trusting them
- Translating model coefficients into ranked, controllable business recommendations

## Limitations

- Silhouette scores are low (0.12–0.14) across all clustering methods, indicating soft cluster boundaries. The segments are supported by dendrogram structure, centroid separation, and agreement between two independent methods rather than by score alone.
- The dataset is synthetic, so effect sizes should not be read as real-world benchmarks.
- Coefficients describe association, not causation — particularly for demographic variables such as marital status and gender.

## References

- Dyerly, R. (2025, January 21). *The myth of replaceability: Preparing for the loss of key employees.* SHRM Executive Network.
- Hees, J. (2015, August 26). *SciPy hierarchical clustering and dendrogram tutorial.* Jörn's Blog.
- Shmueli, G., Bruce, P. C., Yahav, I., Patel, N. R., & Lichtendahl, K. C. (2020). *Data Mining for Business Analytics: Concepts, Techniques, and Applications in Python.* John Wiley & Sons.
