import classification as cl
import matplotlib.pyplot as pyplot
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

import sklearn.feature_selection as fs

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
def featureSelectionKBest(kFeatures, xTrain=cl.X_train, xTest=cl.X_test, yTrain=cl.y_train):
    k_fit=fs.SelectKBest(score_func=fs.f_classif, k=kFeatures)
    k_fit.fit_transform(xTrain,yTrain)
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
def featureSelectionVariance(thresh, xTrain=cl.X_train, xTest=cl.X_test, yTrain=cl.y_train):
    selector=fs.VarianceThreshold(threshold=thresh)
    selector.fit_transform(xTrain,yTrain)
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
def featureSelectionRFE(model, n, xTrain=cl.X_train, xTest=cl.X_test, yTrain=cl.y_train):
    match model:
        case "Random Forest":
            estimator=RandomForestClassifier(random_state=42,n_estimators=25, min_samples_leaf=0.2)
        case "Gradient Boost":
            estimator=GradientBoostingClassifier(random_state=42, learning_rate=1.9, min_samples_leaf=0.05, min_samples_split=0.35)
    selector=fs.RFE(estimator, n_features_to_select=n)
    selector = selector.fit(xTrain, yTrain)
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
def featureSelectionFromModel(model, xTrain=cl.X_train, xTest=cl.X_test, yTrain=cl.y_train):
    match model:
        case "Random Forest":
            estimator=RandomForestClassifier(random_state=42,n_estimators=25, min_samples_leaf=0.2)
        case "Gradient Boost":
            estimator=GradientBoostingClassifier(random_state=42, learning_rate=1.9, min_samples_leaf=0.05, min_samples_split=0.35)
    selector=fs.SelectFromModel(estimator=estimator)
    selector = selector.fit(xTrain, yTrain)
    feature_indices=selector.get_support(indices=True)
    X_newTrain=xTrain.iloc[:,feature_indices]
    X_newTest=xTest.iloc[:,feature_indices] 
    return X_newTrain, X_newTest  


model="Gradient Boost"

X_trainVar, X_testVar = featureSelectionVariance(0)

'''
X_train, X_test = featureSelectionKBest(X_trainVar, X_testVar, 60)
print("\nSelect K Best")
cl.printMetrics(model, X_train,  X_test)

#The next two printMetrics calls will only work when the model has previously assigned weights to features
#Comment if the model chosen does not do this

X_train, X_test = featureSelectionRFE(X_trainVar, X_testVar,model, 20)
print("\nRecursive Feature Elimination")
cl.printMetrics(model, X_train, X_test)
'''


X_trainVarRFE, X_testVarRFE = featureSelectionFromModel(model=model)
print("\nSelect From Model")
cl.printMetrics(model, X_trainVarRFE, X_testVarRFE)

#Determine how the resulting accuracy of a feature selection algorithm changes with a given value
accuracyHigh=0.0            #Track the highest accuracy found
highX=0                     #Track where the highest accuracy is found
accuracy = []               #A list of all accuracies found
xList = []                  #A list of all values of x computed
x=5
while(x<100):
    """For Select K Best and RFE, 0<x
       For Variance Selection 0<x<1"""
    #X_trainVarModel, X_testVarModel = featureSelectionVariance(x)
    X_trainVarModel, X_testVarModel = featureSelectionKBest(x)#, xTrain=X_trainVar, xTest=X_testVar)
    #X_trainVarModel, X_testVarModel = featureSelectionRFE(model, x, xTrain=X_trainVar, xTest=X_testVar)
    acc, auc_score = cl.getMetrics(model, xTrain=X_trainVarModel, xTest=X_testVarModel)
    accuracy.append(acc)
    xList.append(x)

    #Determine where the highest accuracy was found
    if acc > accuracyHigh:
        accuracyHigh=acc
        highX=x
    x+=5

#Display where the highest accuracy was found
print("Accuracy: ", accuracyHigh, "at", highX)

#Display a line graph respresenting x values and their corresponding accuracy
pyplot.plot(xList, accuracy)
pyplot.xlabel("# of Features")
pyplot.ylabel("Accuracy")
pyplot.title("Select K Best")
pyplot.show()
