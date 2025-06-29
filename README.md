# Credit Scoring Model for Bati Bank

## Credit Scoring Business Understanding

1. **How does the Basel II Accord’s emphasis on risk measurement influence our need for an interpretable and well-documented model?**  
   The Basel II Accord mandates that banks accurately measure and manage credit risk to determine capital adequacy. This requires models that are interpretable and well-documented, enabling regulators and stakeholders to understand the risk assessment process and ensure compliance with regulatory standards.

2. **Since we lack a direct "default" label, why is creating a proxy variable necessary, and what are the potential business risks of making predictions based on this proxy?**  
   Without a "default" label, a proxy variable is essential to define a target for model training, such as categorizing customers as high or low risk. However, this proxy may not perfectly align with actual default behavior, risking misclassification that could lead to financial losses or incorrect loan decisions.

3. **What are the key trade-offs between using a simple, interpretable model (like Logistic Regression with WoE) versus a complex, high-performance model (like Gradient Boosting) in a regulated financial context?**  
   Simple models like Logistic Regression with Weight of Evidence (WoE) are transparent and regulator-friendly but may miss complex data patterns. Complex models like Gradient Boosting offer higher predictive accuracy but are less interpretable, posing challenges in regulated environments where explainability is critical.