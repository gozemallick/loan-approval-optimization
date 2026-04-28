# Loan Approval Optimization

Predictive modeling and offline reinforcement learning for profit-aware credit decisioning.

This project builds an end-to-end loan approval pipeline using LendingClub-style historical loan data. It compares traditional supervised learning models that predict default risk with an offline reinforcement learning policy that directly optimizes loan approval profit.

## Project Goal

Most credit risk models answer one question:

```text
How likely is this applicant to default?
```

This project also asks a business question:

```text
Should this loan be approved if the goal is to maximize expected profit?
```

To answer both, the notebook trains supervised models for default prediction and a Conservative Q-Learning agent for approval policy optimization.

## Repository Structure

| File | Description |
| --- | --- |
| `notebook.ipynb` | Main notebook containing EDA, preprocessing, supervised modeling, ensemble modeling, offline RL, and result comparison. |
| `best_nn_model.pth` | Saved PyTorch neural network weights from the best validation AUC checkpoint. |
| `requirements.txt` | Python dependencies required to run the notebook. |
| `README.md` | Project documentation. |

## Data Pipeline

```text
Raw loan data
      |
      v
Target creation from loan_status
      |
      v
EDA and missing-value analysis
      |
      v
Feature engineering
      |
      v
Preprocessing with imputation, scaling, and one-hot encoding
      |
      v
Train / validation / test split
      |
      +--> Neural Network classifier
      |
      +--> Histogram Gradient Boosting classifier
      |
      +--> NN + GB ensemble
      |
      +--> Offline RL dataset creation
               |
               v
          Discrete CQL approval policy
```

## Dataset and Target

The notebook uses a LendingClub-style accepted loans dataset.

The binary target is created from `loan_status`:

| Target | Meaning | Loan statuses |
| --- | --- | --- |
| `0` | Good loan | `Fully Paid` |
| `1` | Bad loan / default risk | `Charged Off`, `Default`, `Late (31-120 days)`, `Late (16-30 days)` |

Rows with ambiguous or non-final loan statuses are removed so the target reflects completed loan outcomes.

## Feature Engineering

The notebook creates applicant, loan, and credit-history features, including:

- `issue_d_year` and `issue_d_month` from issue date.
- `credit_history_length` from earliest credit line date.
- `emp_length_clean` from employment length text.
- `term_clean` from loan term.
- `grade_ord` and `sub_grade_ord` from loan grade.
- `loan_to_income`.
- `installment_to_income`.

The final feature set combines numerical and categorical variables such as loan amount, interest rate, annual income, DTI, revolving utilization, home ownership, verification status, purpose, address state, and application type.

## Preprocessing

The preprocessing pipeline uses:

- Median imputation for numerical features.
- Most-frequent imputation for categorical features.
- Standard scaling for numerical features.
- One-hot encoding for categorical features.
- Stratified train, validation, and test split.

Outliers and invalid values are also handled by clipping selected financial ratios and replacing infinite values.

## Supervised Models

### Neural Network

The PyTorch MLP uses:

- Linear layers.
- Batch normalization.
- ReLU activation.
- Dropout.
- `BCEWithLogitsLoss` with class imbalance weighting.
- Adam optimizer.
- Early stopping based on validation AUC.

The best weights are saved as:

```text
best_nn_model.pth
```

### Histogram Gradient Boosting

The project also trains a `HistGradientBoostingClassifier`:

```python
HistGradientBoostingClassifier(
    max_depth=6,
    learning_rate=0.05,
    max_iter=200,
    random_state=42
)
```

### Ensemble

The ensemble averages the neural network and gradient boosting probabilities:

```python
ensemble_prob = 0.5 * nn_prob + 0.5 * gb_prob
```

## Supervised Model Results

| Model | Validation AUC | Validation F1 | Test AUC | Test F1 | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| Neural Network | 0.74086 | 0.46627 | 0.73927 | 0.46508 | Strong nonlinear baseline |
| Gradient Boosting | 0.73975 | 0.46775 | 0.73787 | 0.46517 | Stable tree-based model |
| Ensemble (NN + GB) | 0.74185 | 0.46859 | 0.74015 | 0.46789 | Best supervised model |

The ensemble gives the best supervised performance, showing that the neural network and gradient boosting model capture slightly different signals.

## Offline Reinforcement Learning

The loan approval problem is reformulated as a single-step offline RL task.

| Component | Definition |
| --- | --- |
| State | Processed applicant and loan feature vector |
| Action | `0 = deny`, `1 = approve` |
| Reward for deny | `0` |
| Reward for approved fully paid loan | Interest-based profit |
| Reward for approved defaulted loan | Negative principal loss |

The reward function is:

```python
if action == 0:
    reward = 0
elif outcome == 0:
    reward = loan_amnt * int_rate * term_years
else:
    reward = -loan_amnt
```

The RL model uses Discrete Conservative Q-Learning through `d3rlpy`.

## RL Policy Results

| Metric | Value |
| --- | ---: |
| Total Profit on Test Set | 109,434,483.87 |
| Expected Profit per Loan | 1,848.34 |
| Approval Rate | 91.28% |
| Approved and Fully Paid | 44,188 |
| Approved and Defaulted | 9,859 |

The RL policy is more business-oriented than the supervised classifiers because it optimizes reward directly instead of AUC or F1.

## ML vs RL Interpretation

Supervised models predict default probability. They are useful for ranking applicant risk and building conservative approval rules.

Offline RL learns an approval policy. It may approve some risky loans when the expected interest return is high enough, and it may reject some low-risk loans when the profit margin is too small.

A practical deployment strategy would combine both:

| Situation | Suggested action |
| --- | --- |
| RL approves and default risk is low | Approve |
| Supervised model predicts very high default risk | Deny |
| RL and supervised model disagree | Send to human review |

## Installation

Clone the repository:

```bash
git clone https://github.com/gozemallick/loan-approval-optimization.git
cd loan-approval-optimization
```

Install dependencies:

```bash
pip install -r requirements.txt
```

If `d3rlpy` installation fails in a notebook environment, install it separately:

```bash
pip install d3rlpy
```

## Running the Project

Open the notebook:

```bash
jupyter notebook notebook.ipynb
```

Update the dataset path inside the notebook:

```python
DATA_PATH = "/path/to/accepted_2007_to_2018Q4.csv"
```

Then run the notebook cells in order.

## Requirements

Main libraries used:

- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `torch`
- `d3rlpy`
- `joblib`
- `tqdm`

## Limitations

- The reward function is simplified and does not include recovery rates, servicing cost, capital cost, late fees, or collection outcomes.
- The RL setup is a single-step offline formulation, so it cannot fully model repayment timelines.
- The notebook uses historical outcomes, so care is needed to avoid data leakage in real deployment.
- Real credit decisions require fairness, compliance, explainability, and human review.
- The model should not be used for real lending decisions without validation, governance, and regulatory review.

## Future Work

- Add SHAP explanations for supervised models.
- Tune LightGBM, XGBoost, and CatBoost baselines.
- Calibrate predicted default probabilities.
- Add risk-sensitive or constrained RL to control approval risk.
- Improve the reward function with recovery amount, cost of capital, and expected loss.
- Add a deployment script or small API for batch scoring.
- Add saved preprocessing artifacts so the trained model can be reused directly.

## Final Outcome

The best supervised model is the NN + GB ensemble with a test AUC of `0.74015`.

The best profit-focused decision policy is the CQL offline RL model, with an estimated test profit of `109,434,483.87` and expected profit of `1,848.34` per loan under the notebook's reward assumptions.
