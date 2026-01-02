# Fraud-Detection-Project
## Project Overview
Credit card fraud, though rare, creates outsized financial and reputational risks. In this project, I built machine learning models to detect fraudulent transactions in a dataset where fraud accounts for only 0.2% of cases. The challenge was not achieving high accuracy which is trivial in such imbalanced data but designing models that maximize recall (catching frauds) while maintaining practical precision (avoiding false alarms).
### Key Findings
- The dataset is extremely imbalanced: 99.8% of transactions are legitimate.
- Top predictors of fraud include PCA derived features such as V17, V14, V3, V12, and V10.
- Model performance:
- Decision Tree: Recall 80%, Precision 84%
- Linear Discriminant Analysis (LDA): Recall 81%, Precision 88% (best balance)
- Logistic Regression and KNN underperformed on recall.
## Business Impact
This model can detect roughly 8 out of 10 fraudulent transactions while keeping false positives low. This balance ensures fraud teams can focus on high risk cases without being overwhelmed. By adjusting thresholds, the models can be tuned to different operational needs  whether the priority is minimizing financial losses or reducing investigation workload.
## Tools & Approach
I used Python with pandas, numpy, scikit learn, seaborn, and matplotlib for data analysis and modeling. Models were saved with pickle and joblib for saving and deployment. Evaluation emphasized recall, precision, and F1 score, supported by confusion matrices and feature importance analysis.
## Conclusion
This project demonstrates that machine learning can provide a reliable fraud detection system, capable of identifying most fraudulent transactions in real time. By prioritizing recall over raw accuracy, I ensured the models address the true business challenge: catching fraud in a sea of legitimate payments.


