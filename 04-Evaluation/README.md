# Week-4
These notes were prepared during week-4 of ML Zoomcamp.

## 4.1 Evaluation metrics: Session overview

The fourth week of Machine Learning Zoomcamp is about different metrics to evaluate a binary classifier. These measures include accuracy, confusion table, precision, recall, ROC curves(TPR, FRP, random model, and ideal model), AUROC, and cross-validation.

## 4.2 Accuracy and Dummy Model

**Accurcy** measures the fraction of correct predictions. Specifically, it is the number of correct predictions divided by the total number of predictions.

We can change the **decision threshold**, it should not be always 0.5. But, in this particular problem, the best decision cutoff, associated with the hightest accuracy (80%), was indeed 0.5.

Note that if we build a **dummy model** in which the decision cutoff is 1, so the algorithm predicts that no clients will churn, the accuracy would be 73%. Thus, we can see that the improvement of the original model with respect to the dummy model is not as high as we would expect.

Therefore, in this problem accuracy can not tell us how good is the model because the dataset is **unbalanced**, which means that there are more instances from one category than the other. This is also known as **class imbalance**.

**Classes and methods:**

* `np.linspace(x,y,z)` - returns a numpy array starting at x until y with a z step
* `Counter(x)` - collection class that counts the number of instances that satisfy the x condition
* `accuracy_score(x, y)` - sklearn.metrics class for calculating the accuracy of a model, given a predicted x dataset and a target y dataset.


The entire code of this project is available in [this jupyter notebook](https://github.com/alexeygrigorev/mlbookcamp-code/blob/master/course-zoomcamp/04-evaluation/notebook.ipynb).

## 4.3 Confusion Table

Confusion table is a way to measure different types of errors and correct decisions that binary classifiers can made. Considering this information, it is possible evaluate the quality of the model by different strategies.

If we predict the probability of churning from a customer, we have the following scenarios:

* No churn - **Negative class**
    * Customer did not churn - **True Negative (TN)**
    * Customer churned - **False Negative (FN)**
* Churn - **Positive class**
    * Customer churned - **True Positive (TP)**
    * Customer did not churn - **False Positive (FP)**

The confusion table help us to summarize the measures explained above in a tabular format, as is shown below:

|**Actual/Predictions**|**Negative**|**Postive**|
|:-:|---|---|
|**Negative**|TN|FP|
|**Postive**|FN|TP|

The **accuracy** corresponds to the sum of TN and TP divided by the total of observations.

The code of this project is available in [this jupyter notebook](https://github.com/alexeygrigorev/mlbookcamp-code/blob/master/course-zoomcamp/04-evaluation/notebook.ipynb).

## 4.4 Precision and Recall
**Precision** tell us the fraction of positive predictions that are correct. It takes into account only the **positive class** (TP and FP - second column of the confusion matrix), as is stated in the following formula:

<p align="center">
    <img src="https://render.githubusercontent.com/render/math?math=\large \frac{TP}{TP %2B FP}"/>
</p>

**Recall** measures the fraction of correctly identified postive instances. It considers parts of the **postive and negative classes** (TP and FN - second row of confusion table). The formula of this metric is presented below:

<p align="center">
    <img src="https://render.githubusercontent.com/render/math?math=\large \frac{TP}{TP %2B FN}"/>
</p>

 In this problem, the precision and recall values were 67% and 54% respectively. So, these measures reflect some errors of our model that accuracy did not notice due to the **class imbalance**.


## 4.5 ROC Curves
- ROC - Receiver Operator Characteristicsm
- WW-2 Radar detectors
- `FPR` = False Positive Rate
- `TPR` = True Positive Rate
- ROC curve looks at all the possible thresholds (FPR Recall)
- randome model
- Ideal model
- TPR vs FPR - model shouldn't go below random baseline
- Can use scikit-learn to plot ROC curve


## 4.6 ROC AUC
- Useful metric for binary classification
- AUC = Area Under the ROC Curve
- ROC curve -  close of ideal point (1.0)
- Another way to quantiy above is area under the curve
- AUC > Great performing model
- For ideal curve, AUC =1.0
- For random curve, AUC =0.5
- Our model, AUC > 0.5 and <= 1.0
- sklearn - `auc` - area under any curve; `roc_auc_score` - computes area under the curve from y_val, y_pred
- How well AUC seperates negative from positive examples.


## 4.7 Cross-Validation
- Parameter tuning for selecting best parameter of our model
- Full train and test - split into parts (k=3) - train and evaluate model - compute mean and std
- `pip install tqdm` to track progress of kfold cross validation
- In `Logistic Regression`, C is regularization parameter

## 4.8 Summary
- Metric - a single number that describes performance a model
- Accuracy - fraction of correct answers, metric can be misleading sometimes
- Precision and Recall - are less misleading when we have class imbalance
- ROC curve - a way to evalute the performance at all thresholds; works with class imbalance too
- K-fold CV - more reliable estimate for performance (mean + std); parameter tuning

## 4.9 Explore more
- F! score
- Evalute prevision and recall at different thresholds, plot P vs R
- AUC
- `Other projects`
    - Calcuate the metrics for datasets from the previsou week

- AMAZING LINK: https://github.com/MemoonaTahira/MLZoomcamp2022/tree/main/Notes/Week_4%20-evaluation_metrics_for_ML_model
