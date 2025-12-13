## Credit Scoring Business Understanding

### 1. Basel II Accord and Its Influence on Credit Risk Modeling
# How does the Basel II Accord’s emphasis on risk measurement influence our need for an interpretable and well-documented model?

The Basel II Capital Accord provides an international regulatory framework that requires banks to measure, manage, and hold capital against credit risk in a risk-sensitive manner. In the context of this project, Basel II strongly influences how credit risk models are designed, documented, and used.

Key implications of Basel II for this project include:

- Emphasis on **quantitative risk measurement** rather than subjective judgment  
- Requirement for **transparent and explainable models** that can be reviewed by regulators  
- Strong focus on **model governance**, documentation, and validation  
- Alignment of credit decisions with **capital adequacy and risk exposure**

Because credit risk models directly affect lending decisions and regulatory capital, this project prioritizes:

- Interpretable features and model outputs  
- Clear documentation of assumptions and limitations  
- Reproducible data processing and model training pipelines  

This ensures that the resulting credit risk model can support regulatory review, internal audits, and responsible credit decision-making, in line with Basel II principles.

---

### 2. Need for a Proxy Default Variable
# Since we lack a direct "default" label, why is creating a proxy variable necessary, and what are the potential business risks of making predictions based on this proxy?

A major challenge in this project is the absence of a direct and observable **default label**, such as loan delinquency, missed payments, or charge-offs. Since supervised machine learning models require a target variable, it is necessary to construct a **proxy variable** that approximates credit risk.

In this project, default risk is inferred from customer transaction behavior using **alternative data**, specifically:

- **Recency** – how recently a customer made transactions  
- **Frequency** – how often the customer transacts  
- **Monetary value** – the volume and value of transactions  

The rationale for using an RFM-based proxy includes:

- Many customers lack traditional credit histories  
- Behavioral transaction data provides early risk signals  
- Alternative credit scoring improves financial inclusion  

However, relying on a proxy variable introduces important business risks:

- The proxy may not perfectly represent true repayment behavior  
- Label noise can lead to incorrect classifications  
- High-risk customers may be incorrectly approved  
- Creditworthy customers may be unfairly rejected  

To mitigate these risks, proxy construction is guided by domain knowledge, exploratory data analysis, and conservative assumptions, and model outputs are interpreted cautiously within the broader credit risk framework.

---

### 3. Trade-offs Between Interpretable and High-Performance Models

# What are the key trade-offs between using a simple, interpretable model (like Logistic Regression with WoE) versus a complex, high-performance model (like Gradient Boosting) in a regulated financial context?
In regulated financial environments, credit risk modeling involves balancing **model interpretability** with **predictive performance**.

**Simple and interpretable models**, such as Logistic Regression with Weight of Evidence (WoE), offer several advantages:

- Clear interpretation of feature contributions  
- High transparency for regulators and auditors  
- Stable and well-established use in credit scoring  
- Easier monitoring and validation over time  

However, these models may have limitations:

- Limited ability to capture nonlinear relationships  
- Reduced performance when using complex alternative data  

**Complex machine learning models**, such as Gradient Boosting, provide:

- Higher predictive accuracy  
- Ability to model nonlinearities and interactions  
- Better use of rich behavioral and transactional data  

But they also introduce challenges:

- Reduced interpretability  
- More complex validation and governance requirements  
- Greater difficulty in regulatory justification  

In this project, both model types are explored to balance accuracy and interpretability. This approach aligns with Basel II expectations while leveraging modern machine learning techniques to enhance credit risk prediction.

---
