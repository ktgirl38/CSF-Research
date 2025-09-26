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
import sklearn.feature_selection as fs
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB

from CSFData import getter
import dataCleaning as dc


#Keep the metabolite data for the features, set the outcomes (PD or Control) as the target
X, y = dc.getXY()


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
def chooseModel(model, test=1, xTrain=X_train, xTest=X_test):
    match model:
        case "Logistic Regression":
            ##########################
            # Logistic Regression
            ##########################
            model=LogisticRegression(solver="liblinear", fit_intercept=False, random_state=42)
            model.fit(xTrain,y_train)

            y_pred=model.predict(xTest)
            y_pred_proba = model.predict_proba(xTest)
        
        case "KNN":    
            ##########################
            # KNN Neighbors
            ##########################

            knn = KNeighborsClassifier(n_neighbors=22)
            knn.fit(xTrain, y_train)
            y_pred = knn.predict(xTest)
            y_pred_proba = knn.predict_proba(xTest)
        
        case "SVM":
            ##########################
            # Support Vector Machine
            ##########################

            svm=SVC(kernel=test, probability=True)
            svm.fit(xTrain,y_train)
            y_pred = svm.predict(xTest)
            y_pred_proba = svm.predict_proba(xTest)
            
        case "Decision Tree":
            ##########################
            # Decision Tree
            ##########################

            tree = DecisionTreeClassifier(criterion=test, random_state=42)
            tree.fit(xTrain, y_train)
            y_pred = tree.predict(xTest)
            y_pred_proba = tree.predict_proba(xTest)

        case "Random Forest":

            ##########################
            # Random Forest
            ##########################

            forest = RandomForestClassifier(random_state=42,n_estimators=25, min_samples_leaf=0.2)
            forest.fit(xTrain, y_train)
            y_pred = forest.predict(xTest)
            y_pred_proba = forest.predict_proba(xTest)

        case "Naive Bayes":

            ##########################
            # Naive Bayes
            ##########################

            nb=GaussianNB()
            nb.fit(xTrain,y_train)
            y_pred = nb.predict(xTest)
            y_pred_proba = nb.predict_proba(xTest)
        
        case "Gradient Boost":

            ##########################
            # Gradient Boosting
            ##########################

            from sklearn.ensemble import GradientBoostingClassifier

            gb=GradientBoostingClassifier(random_state=42, learning_rate=1.9, min_samples_leaf=0.05, min_samples_split=0.35)#learning_rate=.15, random_state=40, n_estimators=65, loss="exponential")
            gb.fit(xTrain, y_train)
            y_pred=gb.predict(xTest)
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
def roc(modelName,xTrain, xTest, plot=True ):
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
def cMatrix(modelName,xTrain,xTest):
    y_pred, y_proba = chooseModel(modelName, xTrain, xTest)
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
def printMetrics(modelName, xTrain, xTest):
    print(modelName)
    y_pred, y_pred_proba = chooseModel(modelName, xTrain, xTest)
    acc=met.accuracy_score(y_test, y_pred)
    report = met.classification_report(y_test, y_pred)
    auc_score=roc(modelName, xTrain, xTest, False)
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
def getMetrics(modelName, xTrain, xTest):
    y_pred, y_pred_proba = chooseModel(modelName, xTrain, xTest)
    acc=met.accuracy_score(y_test, y_pred)
    auc_score=roc(modelName, xTrain, xTest, False)
    return acc, auc_score

############################################
#   Preform the Select K Best feature selection algorithm on the given lists
#
#   Parameters: xTrain - Which training set to use when finding predicting
#               xTest - Which testing set to use when predicting 
#               kFeatures - The number of features for the algorithm to select
#
#   Returns: X_newTrain - The training set adjusted to only include features kept by the algorithm
#            X_newTest - The testing set adjusted to only include features kept by the algorithm
############################################
def featureSelectionKBest(xTrain, xTest, kFeatures):
    k_fit=fs.SelectKBest(score_func=fs.f_classif, k=kFeatures)
    k_fit.fit_transform(xTrain,y_train)
    feature_indices=k_fit.get_support(indices=True)
    X_newTrain=xTrain.iloc[:,feature_indices]
    X_newTest=xTest.iloc[:,feature_indices]
    return X_newTrain, X_newTest

############################################
#   Preform the Variance Threshold feature selection algorithm on the given lists
#
#   Parameters: xTrain - Which training set to use when finding predicting
#               xTest - Which testing set to use when predicting 
#               thresh - The threshold at which the algorithm will keep/remove a feature
#
#   Returns: X_newTrain - The training set adjusted to only include features kept by the algorithm
#            X_newTest - The testing set adjusted to only include features kept by the algorithm
############################################
def featureSelectionVariance(xTrain, xTest, thresh):
    selector=fs.VarianceThreshold(threshold=thresh)
    selector.fit_transform(X_train,y_train)
    feature_indices=selector.get_support(indices=True)
    X_newTrain=xTrain.iloc[:,feature_indices]
    X_newTest=xTest.iloc[:,feature_indices] 
    return X_newTrain, X_newTest  

############################################
#   Preform the RFE feature selection algorithm on the given lists
#
#   Parameters: xTrain - Which training set to use when finding predicting
#               xTest - Which testing set to use when predicting 
#               model - which model to perform the selection algorithm with
#               n - The number of features the algorithm will select
#
#   Returns: X_newTrain - The training set adjusted to only include features kept by the algorithm
#            X_newTest - The testing set adjusted to only include features kept by the algorithm
############################################
def featureSelectionRFE(xTrain, xTest, model, n):
    match model:
        case "Random Forest":
            estimator=RandomForestClassifier(random_state=42,n_estimators=25, min_samples_leaf=0.2)
        case "Gradient Boost":
            estimator=GradientBoostingClassifier(random_state=42, learning_rate=1.9, min_samples_leaf=0.05, min_samples_split=0.35)
    selector=fs.RFE(estimator, n_features_to_select=n)
    selector = selector.fit(xTrain, y_train)
    feature_indices=selector.get_support(indices=True)
    X_newTrain=xTrain.iloc[:,feature_indices]
    X_newTest=xTest.iloc[:,feature_indices] 
    return X_newTrain, X_newTest  

############################################
#   Preform the Select From Model feature selection algorithm on the given lists. Only works with Random Forest
#   and gradient boost. Any adjustments done when creating those classifiers must be reflected here.
#
#   Parameters: xTrain - Which training set to use when finding predicting
#               xTest - Which testing set to use when predicting 
#               model - which model to perform the selection algorithm with
#
#   Returns: X_newTrain - The training set adjusted to only include features kept by the algorithm
#            X_newTest - The testing set adjusted to only include features kept by the algorithm
############################################
def featureSelectionFromModel(xTrain, xTest, model):
    match model:
        case "Random Forest":
            estimator=RandomForestClassifier(random_state=40, max_depth=5, ccp_alpha=0.2, n_estimators=50)
        case "Gradient Boost":
            estimator=GradientBoostingClassifier(learning_rate=.15, random_state=40, n_estimators=65)
    selector=fs.SelectFromModel(estimator=estimator)
    selector = selector.fit(xTrain, y_train)
    feature_indices=selector.get_support(indices=True)
    X_newTrain=xTrain.iloc[:,feature_indices]
    X_newTest=xTest.iloc[:,feature_indices] 
    return X_newTrain, X_newTest  



models = ["Logistic Regression", "KNN", "Decision Tree", "Random Forest", "Naive Bayes", "Gradient Boost", "SVM"]


"""
accuracyHigh=0.0
highX=0.0
accuracy = []
xlist = []
x=['sqrt', 'log2']
value=0.05
#Determine how the accuracy of a base model changes when one of their parameters is changed 
#for value in x:
while(value<1):    
    y_pred, y_pred_proba = chooseModel("Gradient Boost", value)
    acc=met.accuracy_score(y_test, y_pred)
    accuracy.append(acc)
    xlist.append(value)
    if acc > accuracyHigh:
        accuracyHigh=acc
        highX=value
    value+=0.05

pyplot.plot(xlist, accuracy)
pyplot.xlabel("Max depth")
pyplot.ylabel("Accuracy")
pyplot.title("Gradient Boost")
pyplot.show()

#Report at which point of x the accuracy was the highest
print("Accuracy: ", accuracyHigh, "at", highX)
"""

printMetrics("Random Forest", X_train, X_test)