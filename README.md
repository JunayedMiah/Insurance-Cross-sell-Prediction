# Explainable Insurance Cross Sell Prediction

Predicting which existing health insurance customers are likely to buy vehicle insurance, with every prediction explained through SHAP and LIME. Built end to end on Databricks with MLflow tracking and Unity Catalog model governance.

![Databricks](https://img.shields.io/badge/Databricks-FF3621?style=flat-square&logo=databricks&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-0194E2?style=flat-square&logo=mlflow&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2452E8?style=flat-square)
![Optuna](https://img.shields.io/badge/Optuna-4169E1?style=flat-square)
![SHAP](https://img.shields.io/badge/SHAP-1C9A5B?style=flat-square)
![LIME](https://img.shields.io/badge/LIME-B5790E?style=flat-square)

## Objective and Business Context

Cross selling is one of the most direct ways an insurer grows revenue from an existing customer base, but only if the right customers are targeted. This project builds a model that predicts whether a health insurance customer will respond positively to a vehicle insurance offer, and goes a step further by explaining why the model made each decision, using Explainable AI techniques (SHAP and LIME) rather than treating the model as a black box.

## Dataset

Kaggle Health Insurance Cross Sell dataset, loaded into Databricks with Spark. About 16 percent of customers in the raw data show interest in the offer, so class imbalance was a factor throughout the project, from evaluation metric choice to model tuning.

## Workflow

**1. Data loading and initial exploration**
Loaded the dataset from Kaggle into Databricks using Spark, and ran an initial exploratory pass to understand feature distributions and confirm the class imbalance (about 16 percent positive rate).

**2. Exploratory data analysis and visualization**
Analyzed each feature against the target using Matplotlib and Seaborn. Two patterns stood out early: customers with a prior vehicle damage history are far more likely to buy, while customers who already have vehicle insurance are far less likely to respond.

**3. Data preprocessing and feature engineering**
Categorical features (Gender, Vehicle Age, Vehicle Damage) were one hot encoded. Numerical features (Age, Annual Premium, Vintage) were scaled with StandardScaler. Engineered a Premium Per Age ratio feature to surface patterns not explicit in the raw columns.

**4. Train test split and baseline models**
Split the data 80/20 with stratification to preserve class balance. Trained three baseline algorithms, Logistic Regression, Random Forest, and XGBoost, evaluated primarily on Precision, Recall, F1 Score, and ROC AUC rather than raw accuracy, since accuracy is misleading on imbalanced data.

**5. Hyperparameter optimization with Optuna**
Tuned the XGBoost model with Optuna instead of manual search or grid search. Ran 50 trials over estimator count, max depth, learning rate, and regularization terms, and used the best trial's parameters for the final model.

**6. Experiment tracking with MLflow**
Logged every run's hyperparameters, metrics, the trained model artifact, and all generated charts as MLflow artifacts. Registered the final model in Unity Catalog for version control and downstream access.

**7. Global and local explainability with SHAP and LIME**
Used SHAP bar and beeswarm plots to identify the strongest global drivers (Vehicle Damage among the top), and SHAP waterfall plots to explain individual customer predictions. Ran LIME as an independent, model agnostic comparison. SHAP gave highly consistent explanations across repeated runs; LIME was noticeably faster per explanation but showed more run to run variation in which features it surfaced.

**8. Final registration and documentation**
Promoted the tuned model from development to the Unity Catalog staging area, and compiled the workflow and SHAP findings into this documentation for reproducibility and handoff.

## Key Results

| Metric | Baseline (Logistic Regression) | Final (Tuned XGBoost) |
|---|---|---|
| ROC AUC | `0.8237` | `0.8328` |
| F1 Score | `0.3804` | `0.4239` |
| Precision | `0.2371` | `0.2737` |
| Recall | `0.9608` | `0.8957` |


## Business Insights

- **Prior vehicle damage is the strongest single driver.** Customers with a damage history respond at a materially higher rate than those without.
- **Vintage matters.** Longer tenured customers are more likely to respond positively, which suggests loyalty and trust play a role in cross sell success.
- **Explanations agree across methods.** SHAP and LIME converge on the same top features for individual customers, which gives the marketing team confidence the model is picking up real signal rather than noise.

These findings let the marketing team prioritize outreach toward damage history and tenure segments with a documented, defensible reason for each targeting decision, not just a score.

## Explainability: SHAP vs LIME

| | SHAP | LIME |
|---|---|---|
| Consistency across runs | High, near identical explanations each time | Lower, some run to run variation |
| Speed per explanation | Slower (exact, game theoretic) | Faster |
| Scope | Global and local | Local only |
| Model assumptions | Exact for tree models via TreeExplainer | Fully model agnostic |

Used together, they act as a cross check on each other. Where SHAP and LIME agree on a customer's top drivers, that agreement is a reliability signal; where they diverge, that feature is worth treating with more caution before acting on it.

## Tech Stack

| Layer | Tools |
|---|---|
| Platform | Databricks Runtime for ML, Unity Catalog |
| Data | Spark, Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Modeling | Scikit learn, XGBoost |
| Tuning | Optuna |
| Explainability | SHAP, LIME |
| Experiment tracking | MLflow |
