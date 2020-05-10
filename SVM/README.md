# Liver Disease Prediction
 Given blood work results of hundreds of patients, we took different machine learning and data processing approaches to detect patients with liver disease.

# Applied algorithm - SVM

# Steps:

In this exercise we wil be going through the entire data pipeline from preprocessing, data cleaning, staging and modelling on a relatively small dataset (~500 entries). While executing all the steps, we will calculate the performance using  accuracy and F-score for SVM using Support Vector Classification.

## Reading the data:

In the jupiter notebook file the data is loaded from google drive. It is a multivariate data set, which contains 11 columns [Age	Gender	Total_Bilirubin	Direct_Bilirubin	Alkaline_Phosphotase	Alamine_Aminotransferase	Aspartate_Aminotransferase	Total_Protiens	Albumin	Albumin_and_Globulin_Ratio	Liver_Disease]

## Preprocessing:

### Removing duplicates

Duplicate rows can exist in the data because of different reasons. The question regarding whether copies ought to be removed or not relies upon the specific problem context and setting. 

For our situation, duplicates are removed, as in all likelihood it is possible that somebody entered the information for a patient on numerous occasions.

`
data_duplicate = data[data.duplicated(keep = False)]
data_duplicate
`

### Removing null values

We will remove null values or perform imputation before running the model on the provided data. I am using `fillna()` for this reason.
Data imputation of `NaN` values with a specific request statistic(mean, mode, middle) should be possible if adequate setting is accessible. For instance - In our situation, we print those lines whose estimations of section 'Albumin_and_Globulin_Ratio' are missing.

`
data['Albumin_and_Globulin_Ratio'] = data['Albumin_and_Globulin_Ratio'].fillna(data['Albumin_and_Globulin_Ratio'].mode().iloc[0])
data['Albumin_and_Globulin_Ratio'].unique()
`

### Log transform

In the wake of removing the 'Liver_Disease' column from the dataset as it is the label, we show all characteristic in a histogram organization to check if any component has a skewed dispersion. On those highlights, a log change is applied to lower their range.

`
disease_initial = data['Liver_Disease']
features_initial = data.drop('Liver_Disease', axis = 1)
`

## Scaling

Normalization (subtracting mean and scaling difference) is required for some strategies. 

MinMaxScaler - It preserves zero sections if the component matrix is sparse, and is likewise robust to little estimations of S.D. 

`
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
`

## Modelling

Prior to applying any algorithm, we will implement a basic indicator, that will basically restore that each data point has 'Liver_Disease'= True. We will check our metrics(accuracy, TPR, FPR) on that predictor.

`
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
positive_disease= (data['Liver_Disease'] == 1)
positive_disease.astype(int)
report = classification_report(positive_disease, disease)
`

## SVM Trainning and Prediction

### Implementation

Support Vector Machine: SVM aims to find an optimal hyperplane that separates the data into different classes.

`
clf_SVM = SVC(random_state=9)
samples = int(len(X_train) )
results = train_predict(clf_SVM, samples, X_train, y_train, X_test, y_test)
`

### Receiver Operating Characteristic Curve

An extra criterion called as Receiver Operator Characteristics(ROC) curve will be utilized. It plots the curve of True Positive Rate versus the False positive Rate, with a more noteworthy region under the curve demonstrating a better True Positive Rate for the equivalent False Positive Rate. This can be useful for this situation as essentially knowing the quantity of right predictions may not get the job done.

`
pred=clf_SVM.predict(X_test)
fpr, tpr, _ = roc_curve(y_test, pred)
roc_auc = auc(fpr, tpr)
`

### Fine Tuning the Models

We will utilize the grid search strategy to check whether we can improve the accuracy/F-score of SVM with various values for the given hyperparameters.

`
parameters={'kernel':['poly','rbf','linear'], 'C':[0.001,1,1000], 'gamma': [0.00001, 0.0000001]} 
scorer = make_scorer(fbeta_score, beta=2)
_grid_object = GridSearchCV(clf_SVM,parameters,scoring=scorer)
_grid_fit = _grid_object.fit(X_train,y_train)
_best_clf_SVM = _grid_fit.best_estimator_
prediction = (clf_SVM.fit(X_train, y_train)).predict(X_test)
best_prediction = _best_clf_SVM.predict(X_test)
`

`
print("Accuracy score on testing data: {:.4f}".format(accuracy_score(y_test, prediction)))
print("F-score on testing data: {:.4f}".format(fbeta_score(y_test, prediction, beta = 2)))
print("\nOptimized Model\n------")
print("Final accuracy score on the testing data: {:.4f}".format(accuracy_score(y_test, best_prediction)))
print("Final F-score on the testing data: {:.4f}".format(fbeta_score(y_test, best_prediction, beta = 2)))
`
