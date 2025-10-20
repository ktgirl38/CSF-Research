from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, roc_auc_score, auc
import pandas as pd
import matplotlib.pyplot as pyplot
import numpy as np
import sklearn.metrics as met
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB

import dataCleaning as dc


#Keep the metabolite data for the features, set the outcomes (PD or Control) as the target
data=pd.read_csv("cleanData.csv", header=0, index_col="PARENT_SAMPLE_NAME")
y=data.PPMI_COHORT
X=data.drop(data.columns[0:1], axis=1)
X=X.drop("PPMI_COHORT", axis=1)

gbWeights=[]
pd.DataFrame(y)


#Split the data into random subsets representing the test and training data
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42) 


############################################
#   Determine the y probability and prediction for a given model
#
#   Parameters: model - A string containing the name of the model to be used to find the data predictions
#               xTrain - Which training set to use when finding predicting
#               xTest - Which testing set to use when predicting
#               test- A testing variable to easily modify attributes for the given model
#
#   Returns: y_pred - the models predictions for the test data
#            y_pred_prob - the probability that the data is part of the predicted class
############################################
def chooseModel(model, xTrain=X_train, xTest=X_test, yTrain=y_train, x=0):
    match model:
        
        case "Logistic Regression":
            ##########################
            # Logistic Regression
            ##########################
            model=LogisticRegression(solver='liblinear', random_state=42)
            model.fit(xTrain,yTrain)
            y_pred=model.predict(xTest)
            y_pred_proba = model.predict_proba(xTest)
        
        case "KNN":    
            ##########################
            # KNN Neighbors
            ##########################

            knn = KNeighborsClassifier(n_neighbors=22)
            knn.fit(xTrain, yTrain)
            y_pred = knn.predict(xTest)
            y_pred_proba = knn.predict_proba(xTest)
        
        case "SVM":
            ##########################
            # Support Vector Machine
            ##########################

            svm=SVC(kernel="linear", probability=True, random_state=42)
            svm.fit(xTrain,yTrain)
            y_pred = svm.predict(xTest)
            y_pred_proba = svm.predict_proba(xTest)
            
        case "Decision Tree":
            ##########################
            # Decision Tree
            ##########################

            tree = DecisionTreeClassifier(criterion="entropy", min_samples_leaf=0.09, max_features=0.395, random_state=42)
            tree.fit(xTrain,yTrain)
            y_pred = tree.predict(xTest)
            y_pred_proba = tree.predict_proba(xTest)

        case "Random Forest":

            ##########################
            # Random Forest
            ##########################
            forest = RandomForestClassifier(random_state=42, n_estimators=71, min_samples_split=0.29)
            forest.fit(xTrain, yTrain)
            y_pred = forest.predict(xTest)
            y_pred_proba = forest.predict_proba(xTest)

        case "Naive Bayes":

            ##########################
            # Naive Bayes
            ##########################

            nb=GaussianNB()
            nb.fit(xTrain,yTrain)
            y_pred = nb.predict(xTest)
            y_pred_proba = nb.predict_proba(xTest)
        
        case "Gradient Boost":

            ##########################
            # Gradient Boosting
            ##########################

            from sklearn.ensemble import GradientBoostingClassifier

            gb=GradientBoostingClassifier(random_state=42, loss='exponential', learning_rate=0.85, n_estimators=21, max_features=.45)#learning_rate=.15, random_state=40, n_estimators=65, loss="exponential")
            gb.fit(xTrain, yTrain)
            y_pred=gb.predict(xTest)
            gbWeights=gb.feature_importances_
            y_pred_proba = gb.predict_proba(xTest)

    return y_pred, y_pred_proba


############################################
#   Find the auc score for a given model and add it to a plot. If only the auc score is needed
#   the plotting feature can be bypassed.
#
#   Parameters: modelName - A string containing the name of the model to be used to find the data predictions
#               xTrain - Which training set to use when finding predicting
#               xTest - Which testing set to use when predicting
#               plot - A boolean determining whether to plot the fpr and tpr or only return the auc
#
#   Returns: auc_score - a float containing the auc score for the given model
############################################
def roc(modelName, plot=True, xTrain=X_train, xTest=X_test):
    y_pred, y_pred_proba = chooseModel(modelName, xTrain, xTest)
    fpr, tpr, thresh = roc_curve(y_test, y_pred_proba[:,1])
    auc_score = met.auc(fpr, tpr)
    if(plot):
        pyplot.plot(fpr,tpr, label=modelName+" (area = %.02f)" %auc_score)
    
    return auc_score

############################################
#   Compute and display the confusion matrix for a given model
#
#   Parameters: modelName - A string containing the name of the model to be used to find the data predictions
#               xTrain - Which training set to use when finding predicting
#               xTest - Which testing set to use when predicting 
#
#   Returns: tn - The amount of true negatives
#            fp - The amount of false positives
#            fn - The number of false negatives
#            tp - The number of true positives
############################################
def cMatrix(modelName,xTrain=X_train, xTest=X_test):
    y_pred, y_proba = chooseModel(modelName, xTrain=xTrain, xTest=xTest)
    tn,fp,fn,tp=confusion_matrix(y_test, y_pred, labels=["PD", "Control"]).ravel().tolist()
    cm=confusion_matrix(y_test, y_pred, labels=["PD", "Control"])
    disp=ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["PD", "Control"], )
    disp.plot()
    pyplot.title(modelName)
    pyplot.savefig(modelName+" matrix")
    return tn,fp,fn,tp

############################################
#   Print the accuracy, auc, and classification report for a given model
#
#   Parameters: modelName - A string containing the name of the model to be used to find the data predictions
#               xTrain - Which training set to use when finding predicting
#               xTest - Which testing set to use when predicting 
#
#   Returns: Nothing
############################################
def printMetrics(modelName, xTrain=X_train, xTest=X_test, yTrain=y_train):
    print(modelName)
    y_pred, y_pred_proba = chooseModel(modelName)
    acc=met.accuracy_score(y_test, y_pred)
    report = met.classification_report(y_test, y_pred)
    auc_score=roc(modelName, plot=False)
    print("Accuracy:", acc, "\nAUC:", auc_score, "\nClassification Report:\n", report)

############################################
#   Calculate the accuracy, auc, and classification report for a given model
#
#   Parameters: modelName - A string containing the name of the model to be used to find the data predictions
#               xTrain - Which training set to use when finding predicting
#               xTest - Which testing set to use when predicting 
#
#   Returns: acc - the accuracy of the model
#   auc_score - the auc score for the model
############################################
def getMetrics(modelName, xTrain=X_train, xTest=X_test, yTrain=y_train):
    y_pred, y_pred_proba = chooseModel(modelName)
    acc=met.accuracy_score(y_test, y_pred)
    auc_score=roc(modelName, plot=False)
    return acc, auc_score




models = ["KNN", "Logistic Regression", "Decision Tree", "Random Forest", "Naive Bayes", "Gradient Boost", "SVM"]

accList = []
"""
for model in models:
    y_pred, y_pred_proba = chooseModel(model,)
    acc=met.accuracy_score(y_test, y_pred)
    accList.append(acc)

pyplot.plot(models, accList)
pyplot.tick_params("x", labelrotation=20)
pyplot.show()
"""
for model in models:
    roc(model)

pyplot.legend(models)
pyplot.show()

"""
accuracyHigh=0.0
highX=0.0
accuracy = []
xlist = []
x=[True, False]
value=0.0005
#Determine how the accuracy of a base model changes when one of their parameters is changed 
for value in x:
#while(value<0.5):    
    y_pred, y_pred_proba = chooseModel("Gradient Boost", x=value)
    acc=met.accuracy_score(y_test, y_pred)
    accuracy.append(acc)
    xlist.append(value)
    if acc > accuracyHigh:
        accuracyHigh=acc
        highX=value
#    value+=0.0005

pyplot.plot(xlist, accuracy)
pyplot.xlabel("Max depth")
pyplot.ylabel("Accuracy")
pyplot.title("Gradient Boost")
pyplot.show()

#Report at which point of x the accuracy was the highest
print("Accuracy: ", accuracyHigh, "at", highX)

"""

for model in models:
    printMetrics(model)