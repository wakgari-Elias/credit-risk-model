📌 Credit Scoring Business Understanding
1. Basel II Accord and the Need for Interpretable Credit Risk Models

The Basel II Capital Accord provides a regulatory framework that links a bank’s capital requirements directly to the level of credit risk it assumes. This framework strongly influences how credit risk models must be designed and used.

Key implications of Basel II for this project include:

Risk-sensitive capital allocation

Banks must quantify credit risk to determine minimum regulatory capital.

Inaccurate or opaque models can lead to underestimation or overestimation of risk.

Emphasis on transparency and documentation

Models must be explainable to regulators, auditors, and internal risk committees.

Feature selection, assumptions, and transformations must be clearly documented.

Model governance and validation requirements

Credit risk models must be reproducible and regularly validated.

Clear model logic supports stress testing, monitoring, and supervisory review (Pillar 2).

As a result, this project prioritizes interpretable, well-documented, and auditable modeling approaches to ensure alignment with Basel II principles and regulatory expectations.

2. Proxy Definition of Default Risk

A major challenge in this project is the absence of a direct default label (e.g., loan delinquency or repayment failure). Since supervised learning requires a target variable, a proxy for default risk must be constructed.

Reasons a proxy variable is necessary:

The dataset contains transactional and behavioral data only

No explicit loan performance or repayment outcomes are available

Credit scoring models require labeled outcomes for training

Approach used in this project:

Recency: How recently a customer made transactions

Frequency: How often a customer transacts

Monetary: The total value of customer transactions

These RFM metrics are used to approximate customer creditworthiness, where unfavorable behavioral patterns indicate higher risk.

Potential business risks of using a proxy:

Label noise: The proxy may not accurately represent true default behavior

Misclassification risk:

High-risk customers may be incorrectly approved

Creditworthy customers may be unfairly rejected

Strategic risk: Poor proxy assumptions can negatively impact profitability and customer trust

To mitigate these risks, proxy construction is guided by domain knowledge, validated through exploratory data analysis, and interpreted cautiously in credit decision-making.

3. Trade-offs Between Interpretable and High-Performance Models

In regulated financial environments, selecting a credit risk model involves balancing predictive accuracy with interpretability and regulatory compliance.

Simple and interpretable models (e.g., Logistic Regression with WoE):

Advantages:

High transparency and explainability

Strong regulatory acceptance

Stable and easy to monitor over time

Limitations:

Limited ability to capture nonlinear relationships

May underperform on complex behavioral data

Complex models (e.g., Gradient Boosting):

Advantages:

Higher predictive performance

Ability to model nonlinearities and feature interactions

Better utilization of alternative data

Limitations:

Harder to interpret and explain

Increased validation and governance complexity

Potential regulatory resistance

Project approach:

Evaluate both model types

Balance performance with interpretability

Align model usage with Basel II principles and risk governance requirements