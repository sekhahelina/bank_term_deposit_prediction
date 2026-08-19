# Term Deposit Prediction

## Overview

This project focuses on a **binary classification problem**: predicting whether a bank customer will subscribe to a **term deposit** as a result of a direct marketing campaign.

The main business goal is to help the bank **prioritize customers for outbound calls**. Instead of contacting customers equally, the model estimates the probability of subscription and can help focus marketing resources on customers with a higher likelihood of conversion.

The project covers the complete machine learning workflow, including **EDA, preprocessing, feature engineering, class imbalance experiments, model comparison, hyperparameter tuning, threshold optimization, model interpretation, and error analysis**.

## Dataset

The project uses the **Bank Marketing Dataset**, containing information about customers, previous marketing interactions, the current campaign, and macroeconomic indicators.

**Dataset:** [Bank Marketing Dataset — Kaggle](https://www.kaggle.com/datasets/sahistapatel96/bankadditionalfullcsv)

- **41,188 observations**
- **21 columns**
- Target variable: `y` (`yes` / `no`)
- Approximately **88% `no`** and **11% `yes`**

The strong class imbalance makes accuracy a poor primary metric, so model performance was mainly evaluated using **ROC-AUC and F1-score**.

### Data Leakage

The `duration` feature represents the duration of the last phone call. Although highly predictive, it is only known **after the call has taken place**.

Since the intended use case is to predict which customers should be contacted, `duration` was excluded from the final model to avoid **data leakage** and make the prediction scenario more realistic.

## EDA

Exploratory analysis revealed several important patterns:

- Customers at the younger and older ends of the age distribution showed higher subscription rates than many middle-aged customers.
- The effectiveness of repeated calls generally decreased after several attempts within the same campaign.
- Customers with a successful previous campaign outcome (`poutcome = success`) had a substantially higher subscription rate.
- The `cellular` contact channel was associated with a considerably higher conversion rate than `telephone`.
- Strong correlations were found between several macroeconomic variables, indicating **multicollinearity**.

These findings were used to guide feature preprocessing and model experiments.

## Approach

### Preprocessing

The preprocessing pipeline included:

- Handling missing values
- Grouping similar education categories
- Transforming the special `pdays = 999` value
- Creating the `pdays_contacted` indicator
- Binary encoding of selected categorical features
- One-Hot Encoding for remaining categorical variables
- Scaling numerical features where required
- Stratified train/validation/test splitting

### Feature Engineering

Additional experiments included:

- Investigating highly correlated macroeconomic variables
- Comparing configurations with and without selected correlated features
- Evaluating the effect of outlier handling for tree-based models

### Class Imbalance

Several approaches were compared:

- SMOTE
- SMOTENC
- SMOTE-Tomek
- `class_weight`
- `scale_pos_weight`

Resampling methods improved some individual models but did not consistently improve generalization. The final XGBoost model therefore used **`scale_pos_weight`** instead of synthetic oversampling.

### Models

The following algorithms were evaluated:

- Logistic Regression
- k-Nearest Neighbors
- Decision Tree
- Random Forest
- XGBoost

### Hyperparameter Tuning

Two optimization approaches were used:

- **RandomizedSearchCV** for kNN, Decision Tree, Random Forest, and XGBoost
- **Hyperopt with TPE** for the final XGBoost model

The XGBoost search included parameters such as:

`n_estimators`, `learning_rate`, `max_depth`, `subsample`, `colsample_bytree`, `gamma`, `reg_alpha`, `reg_lambda`, and `min_child_weight`.

## Results

The models were primarily compared using **Validation ROC-AUC**, with F1-score used to evaluate classification performance at the selected threshold.

| Model | Validation ROC-AUC | F1 |
|---|---:|---:|
| Logistic Regression |**0.797** | 0.447 |
| kNN | **0.783** | 0.321 |
| Decision Tree | **0.769** | 0.426 |
| Random Forest | **0.783** | 0.467 |
| **XGBoost** | **0.805** | **0.465** |

**XGBoost achieved the best validation ROC-AUC among the tested models.**

## Final Model

The final model is a **tuned XGBoost classifier** using `scale_pos_weight` to account for class imbalance.

### Performance

| Metric | Result |
|---|---:|
| Validation ROC-AUC | **0.805** |
| Test ROC-AUC | **0.812** |
| Test F1 at threshold 0.50 | 0.49 |
| Test F1 at threshold 0.66 | **0.53** |

### Threshold Tuning

The default classification threshold of `0.50` was optimized on the validation set by maximizing F1-score.

The selected threshold was **0.66**.

On the test set, increasing the threshold resulted in:

- Precision: **0.39 → 0.48**
- Recall: **~0.63 → 0.59**
- F1-score: **0.49 → 0.53**

This made the model more selective when identifying customers as potential subscribers.

## Interpretability

The final XGBoost model was analyzed using **Feature Importance** and **SHAP**.

SHAP analysis was used to understand both the overall influence of features and their impact on individual predictions.

The most influential feature groups included:

- Macroeconomic indicators such as `euribor3m`, `cons.price.idx`, and `cons.conf.idx`
- Previous campaign outcome, particularly `poutcome_success`
- Previous contact information such as `pdays`
- Communication channel, especially `contact_cellular`

Feature importance and SHAP results were also checked from a business perspective to ensure that the model's behavior was reasonably interpretable.

## Business Impact

The model can be used as a **customer prioritization tool** for direct marketing campaigns.

Instead of treating every customer equally, the bank can:

1. Estimate the probability of subscription for each customer
2. Rank customers by predicted probability
3. Prioritize high-probability customers for calls
4. Allocate call-center resources more efficiently

The optimal classification threshold should ultimately depend on the actual business costs of:

- **False Positives** — unnecessary calls
- **False Negatives** — missed potential customers

Therefore, the current threshold of `0.66` is an optimization based on F1-score rather than a universally optimal business threshold.

## Future Improvements

Possible directions for further development include:

- **Business-cost-based threshold optimization** instead of optimizing only F1-score
- Comparison with **LightGBM and CatBoost**
- More extensive hyperparameter optimization using **Optuna** or a larger Hyperopt search
- Additional **feature engineering**, including campaign interactions and customer contact history
- **Time-based validation** to better reproduce a real-world scenario
- Testing model stability across multiple random seeds and cross-validation folds

