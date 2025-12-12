# loan-approval-optimization
Predictive Modeling + Offline RL for Profit-Optimized Credit Decisioning

This project implements an end-to-end pipeline for loan risk prediction and policy optimization using:

EDA & Feature Engineering

Deep Learning Model (MLP)

Gradient Boosting (HistGB)

Ensemble Model (NN + GB)

Offline Reinforcement Learning (CQL)

Comparison of ML classification vs. RL decision policy

The goal is to build both a high-AUC classifier and a profit-maximizing approval policy, and then compare how ML vs. RL behave on real LendingClub-style loan data.
## Project Structure
project/
│── README.md
│── requirements.txt
│── notebook/
│──best_nn_model.pth
│── cql_model.d3

git clone [https://github.com/<your-username>/<repo-name>.git](https://github.com/gozemallick/loan-approval-optimization)
cd <repo-name>
pip install -r requirements.txt
pip install d3rlpy
1. Data Cleaning & Feature Engineering

Main steps:

Parse issue_d → year + month

Compute credit history length

Clean employment length

Extract numeric loan term

Ordinal encode grade/subgrade

Create ratio features:

loan_to_income

installment_to_income

Handle missing values & outliers

One-hot encode categorical columns

Train/Val/Test split (70/15/15)

Output:

X_train_p, X_val_p, X_test_p (processed)

y_train, y_val, y_test
2. Deep Learning Model (MLP)

Architecture:

Linear → BatchNorm → ReLU → Dropout

Linear → BatchNorm → ReLU → Dropout

Linear → ReLU

Output layer (logit)

Training details:

BCEWithLogitsLoss with class imbalance correction

Adam optimizer

Early stopping on Validation AUC

Best weights saved automatically as:
best_nn_model.pth
Final NN Results
| Metric             | Validation | Test   |
| ------------------ | ---------- | ------ |
| **AUC**            | 0.7408     | 0.7393 |
| **F1**             | 0.4663     | 0.4651 |
| **Best Threshold** | 0.56       | —      |

3. Histogram Gradient Boosting Model (HistGB)

Model:
HistGradientBoostingClassifier(
    max_depth=6,
    learning_rate=0.05,
    max_iter=200
)
GB Results
| Metric  | Validation | Test   |
| ------- | ---------- | ------ |
| **AUC** | 0.7397     | 0.7379 |
| **F1**  | 0.4677     | 0.4652 |
4. Ensemble Model (NN + GB)

Combined probability:
ensemble_prob = 0.5 * NN + 0.5 * GB

Ensemble Results
| Metric  | Validation   | Test         |
| ------- | ------------ | ------------ |
| **AUC** | ⭐ **0.7418** | ⭐ **0.7401** |
| **F1**  | ⭐ **0.4686** | ⭐ **0.4679** |

5. Offline Reinforcement Learning (CQL)
🔹 How RL Dataset Was Created

Every loan forms two transitions:
| Action          | Reward                                  |
| --------------- | --------------------------------------- |
| **1 = Approve** | Profit if paid, `-loan_amnt` if default |
| **0 = Reject**  | Always `0`                              |

This forms a single-step MDP suitable for offline RL.

🔹 Algorithm

Discrete CQL (Conservative Q-Learning)

Input: 96 processed features

Output: action (0/1)

RL Policy Results
| Metric                  | Value        |
| ----------------------- | ------------ |
| **Total Profit**        | $109,434,483 |
| **Avg Profit per Loan** | $1848.34     |
| **Approval Rate**       | 91.28%       |
| Fully Paid Approved     | 44,188       |
| Default Approved        | 9,859        |

The RL model directly maximizes profit, not AUC/F1.
6. ML vs RL Comparison
Machine Learning (NN, GB, Ensemble)

Outputs probability of default

Must choose a threshold

Optimizes AUC/F1

Good for ranking risk

Reinforcement Learning (CQL)

Learns profit-maximizing approval decisions

No threshold needed

Optimizes reward, not accuracy metrics

Better business performance

Insight:

AUC/F1 tells you how accurate your classifier is,
RL tells you how to make money.
Requirements
numpy
pandas
scikit-learn
matplotlib
seaborn
torch
d3rlpy
joblib
10. Final Outcomes
🔹 Best Classifier

Ensemble (NN + GB)
AUC = 0.7401, F1 = 0.4679

🔹 Best Decision Policy

CQL RL Model
Expected profit per loan = $1848.34

ML gives risk probabilities,
RL gives business-optimized decisions.
Contact

This project demonstrates:

Full ML pipeline

Deep Learning + GB + Ensemble

Offline RL with Conservative Q-Learning

Business-oriented evaluation
