# Healthcare Insurance Claim Prediction (Python, scikit-learn)

**Question:** Can we predict an individual's healthcare insurance claim amount from their demographic and health data?

An insurer that can estimate claim amounts can price premiums by risk, identify high-cost individuals, and flag unusually high predicted claims for review. This project compares eight regression models on 1,332 people and finds that tree-based models, gradient boosting in particular, fit this data best.

**Tools:** Python · pandas · NumPy · matplotlib · scikit-learn (Google Colab)  
**Data:** [Insurance Claim Analysis: Demographic and Health](https://www.kaggle.com/datasets/thedevastator/insurance-claim-analysis-demographic-and-health) (Kaggle), originally by Sumit Kumar Shukla ([data.world/sumitrock](https://data.world/sumitrock))  
**Team:** Jimmy Tran and Steven Ho — CIS 4920 group project, Georgia State University

---

## Results

| Model | Test RMSE ($) | Test R² |
| --- | ---: | ---: |
| **Gradient Boosting** | **4,882.91** | **0.8376** |
| Random Forest | 4,933.59 | 0.8342 |
| Decision Tree | 4,934.22 | 0.8342 |
| Polynomial Regression (degree 2) | 5,573.35 | 0.7884 |
| Support Vector Regression (RBF) | 5,902.57 | 0.7627 |
| Linear Regression | 6,311.19 | 0.7287 |
| k-Nearest Neighbors | 9,652.62 | 0.3654 |
| Neural Network (MLP) | 10,765.61 | 0.2106 |

- **Gradient boosting** explained about 84% of the variation in claims and cut RMSE by 22.6% compared with linear regression.
- **Tree-based models dominated** the top three, capturing non-linear effects such as the large jump in claims for smokers.
- **Smoking, blood pressure and BMI** were associated with higher claim amounts. In the raw data, smokers averaged about $32,050 in claims versus about $8,421 for non-smokers.
- **k-NN, SVR and the neural network** were trained on unscaled features; see *Limitations*.

**Business use:** risk-based premium pricing, identifying high-cost individuals, and flagging unusually high predicted claims for review. (The data has no fraud labels, so the model flags high predictions; it does not detect fraud.)

---

## Data

| Column | Description |
| --- | --- |
| age | Age (5 missing values) |
| gender | male / female |
| bmi | Body mass index |
| bloodpressure | Blood pressure |
| diabetic | Yes / No |
| children | Number of children |
| smoker | Yes / No |
| region | northeast, northwest, southeast, southwest (3 missing values) |
| claim | **Target** — claim amount in dollars (mean $13,253, median $9,370, max $63,770) |
| index, PatientID | Identifier columns (dropped) |

1,340 rows → 1,332 after removing the 8 rows with missing values.

## Approach

1. **Explore** — shape, data types, claim boxplot and summary statistics.
2. **Clean** — drop rows with missing values (less than 1% of the data) and the two identifier columns.
3. **Prepare** — separate the target (`claim`) from features, plot each feature against claim, one-hot encode categorical columns with `pd.get_dummies`, and split into train/test sets (`train_test_split` defaults: 75/25; `np.random.seed(42)` for reproducibility).
4. **Baselines** — linear regression and polynomial regression (degree 2). Train and test R² were close (0.70 vs 0.73; 0.79 vs 0.79), so no sign of overfitting.
5. **Tuned models** — decision tree, random forest, k-NN, SVR, gradient boosting and an MLP neural network, each tuned with `GridSearchCV` (10-fold cross-validation) and refit with the best parameters.
6. **Evaluate** — RMSE and R² on the held-out test set; actual-vs-predicted plots, a decision tree plot and random forest feature importances.

**Best parameters found**

| Model | Best parameters |
| --- | --- |
| Decision Tree | squared_error, max_depth 3, splitter best |
| Random Forest | 50 trees, max_depth 5, bootstrap True |
| k-NN | k = 31 |
| SVR (RBF) | C = 10,000,000, epsilon = 100 |
| Gradient Boosting | 50 trees, learning_rate 0.1, max_depth 3 |
| MLP | learning_rate constant, max_iter 1,000 |

## Limitations and next steps

- **No feature scaling.** Distance- and gradient-based models (k-NN, SVR, MLP) are sensitive to scale, which likely explains their weak results; the MLP also did not converge. Adding `StandardScaler` in a `Pipeline` is the first improvement.
- **Grid edges.** The best k (31) and C (10 million) were at the edge of their search grids, so wider grids may help.
- **Single split.** Final scores come from one train/test split; cross-validated final scores would be more robust.
- **Error size.** Even the best RMSE (~$4,883) is about a third of the average claim, so the model suits risk ranking more than exact cost estimates.
- **Fairness.** Pricing on health traits is regulated; a production model would need fairness and compliance review.

## Run it

The dataset isn't included in this repo. Download `insurance_data.csv` from the [Kaggle page](https://www.kaggle.com/datasets/thedevastator/insurance-claim-analysis-demographic-and-health) and save it as `data/insurance_data.csv`.

```bash
pip install -r requirements.txt
jupyter notebook notebooks/insurance_claim_prediction.ipynb
```

Or open the notebook in Google Colab, upload the downloaded CSV, and change the `read_csv` path to `'insurance_data.csv'`.

## Repository structure

```
├── README.md
├── requirements.txt
├── data/                  # add insurance_data.csv here (not tracked)
└── notebooks/
    └── insurance_claim_prediction.ipynb
```

---

**Jimmy Tran** · [LinkedIn](https://www.linkedin.com/in/jimmytran8/)
