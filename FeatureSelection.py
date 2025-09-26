import classification as cl
import matplotlib.pyplot as pyplot

model="Random Forest"

cl.printMetrics(model, cl.X_train, cl.X_test)
X_trainVar, X_testVar = cl.featureSelectionVariance(cl.X_train, cl.X_test, 0.0)
cl.printMetrics(model, X_trainVar, X_testVar)

X_train, X_test = cl.featureSelectionKBest(X_trainVar, X_testVar, 60)
print("\nSelect K Best")
cl.printMetrics(model, X_train,  X_test)

#The next two printMetrics calls will only work when the model has previously assigned weights to features
#Comment if the model chosen does not do this

X_trainVarRFE, X_testVarRFE = cl.featureSelectionFromModel(cl.X_train, cl.X_test, model)
print("\nSelect From Model")
cl.printMetrics(model, X_trainVarRFE, X_testVarRFE)

X_train, X_test = cl.featureSelectionRFE(X_trainVar, X_testVar,model, 20)
print("\nRecursive Feature Elimination")
cl.printMetrics(model, X_train, X_test)


#Determine how the resulting accuracy of a feature selection algorithm changes with a given value
accuracyHigh=0.0            #Track the highest accuracy found
highX=0                     #Track where the highest accuracy is found
accuracy = []               #A list of all accuracies found
xList = []                  #A list of all values of x computed
x=5
while(x<100):
    """For Select K Best and RFE, 0<x
       For Variance Selection 0<x<1"""
    X_trainVarModel, X_testVarModel = cl.featureSelectionKBest(X_trainVar, X_testVar, x)
    #X_trainVarModel, X_testVarRFE = cl.featureSelectionRFE(X_trainVar, X_testVar, x)
    #X_trainVarModel, X_testVarModel = cl.featureSelectionVariance(cl.X_train, cl.X_test, x)
    acc, auc_score = cl.getMetrics(model, X_trainVarModel, X_testVarModel)
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
