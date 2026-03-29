Ensemble Learning Comparison 

This project demonstrates the comparison between different machine learning models, focusing on Ensemble Learning techniques and 
their impact on model performance.

>> Objective
The main goal of this project is to:

-> Understand how ensemble methods improve accuracy
-> Compare single models vs multiple models
-> Analyze overfitting and generalization

>> Models Used
1) Decision Tree
2) BaggingClassifier (with Decision Tree)
3) Random Forest
S4) upport Vector Machine with Bagging

>> Dataset
1) Synthetic dataset created using make_classification
2) Features: 6 numerical features
3) Samples: 10,000
4) Binary classification problem
   
>> Workflow
1) Generate dataset
2) Split into training and testing sets
3) Train different models
4) Evaluate using accuracy
5) Compare performance

>> Results
Test Accuracy:-
Decision Tree: 93.7%
Bagging (Decision Tree): 97.55%
Random Forest: 96.6%
Bagged SVM: 98.65%

Training Accuracy:- 
Decision Tree: 94.6%
Bagging: 99.07%
Random Forest: 97.08%
Bagged SVM: 98.42%

>> Key Insights
1) A single Decision Tree may underfit or overfit depending on depth
2) Ensemble methods like Bagging and Random Forest improve stability
3) Bagging reduces variance by combining multiple models
4) Bagged SVM achieved the best performance on this dataset
   
⚠️ Important Learning

High training accuracy like 100% does not always mean a good model.
The main focus should be on test accuracy and generalization.

🛠️ Technologies Used
Python
NumPy
Pandas
Scikit-learn

>> Conclusion

Ensemble learning methods significantly improved model performance compared to a single model. Among all, Bagged SVM performed the best, 
showing that combining models can lead to better and more reliable predictions.
