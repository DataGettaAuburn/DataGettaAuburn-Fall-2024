from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize
import pandas as pd
import numpy as np
import math
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import configparser
from Logs import logging as logs
from Data import DataUtil
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import LeaveOneOut
import json

config = configparser.ConfigParser()
config.read('Data//config.ini')

def trainHyperParameters(model, grid, train_x, train_y):
    grid_search = GridSearchCV(estimator=model, param_grid=grid, cv=5, n_jobs=-1, verbose=2)
    grid_search.fit(train_x, train_y)
    return grid_search.best_estimator_

# Run Random Forest Regressor
# Inputs:
    # train_x
    # train_y
    # test_x
    # test_y
def runRFR(train_x, train_y, test_x, test_y):
    rfr = RandomForestRegressor()
    param_grid = {
        'n_estimators': [100, 200, 400, 800, 1200],
        'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
        'criterion': ['squared_error','absolute_error','friedman_mse','poisson'],
        'min_samples_split': [2, 4, 8, 12],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None],
        'bootstrap': [True, False]
    }
    best_rfr = trainHyperParameters(rfr, param_grid, train_x, train_y)
    best_rfr.fit(train_x, train_y)

    #importances = rfr.feature_importances_
    predictions = best_rfr.predict(test_x)
    directionScore, distanceScore = measurePerformance(predictions, test_y)

    return directionScore, distanceScore


def trainHyperParameters(model, grid, train_x, train_y):
    grid_search = GridSearchCV(estimator=model, param_grid=grid, cv=5, n_jobs=-1, verbose=2)
    grid_search.fit(train_x, train_y)
    return grid_search.best_estimator_

# Run Random Forest Regressor
# Inputs:
    # train_x
    # train_y
    # test_x
    # test_y
def runRFR(train_x, train_y, test_x, test_y):
    rfr = RandomForestRegressor()
    param_grid = {
        'n_estimators': [100, 200, 400, 800, 1200],
        'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
        'criterion': ['squared_error','absolute_error','friedman_mse','poisson'],
        'min_samples_split': [2, 4, 8, 12],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None],
        'bootstrap': [True, False]
    }
    best_rfr = trainHyperParameters(rfr, param_grid, train_x, train_y)
    best_rfr.fit(train_x, train_y)

    #importances = rfr.feature_importances_
    predictions = best_rfr.predict(test_x)
    directionScore, distanceScore = measurePerformance(predictions, test_y)

    return directionScore, distanceScore


# Run Decision Tree Training and Testing
# Inputs:
    # train_x
    # train_y
    # test_x
    # test_y
    # max_depth (maximum depth of tree)
    # max_features (number of features considered when looking for best split)
    # max_leaf_nodes (Maximum number of leaf nodes in the tree)
# Ouput:
    # Decision Tree Model, Training Accuracy, Testing Accuracy
def runDT(train_x, train_y, test_x, test_y, max_depth, max_features, max_leaf_nodes, fieldModelType):
    if (config['MODELS']['DTC'] == 'False'):
        return None, None, None
    dt = DecisionTreeClassifier(max_depth=max_depth, max_features=max_features, max_leaf_nodes=max_leaf_nodes) #, class_weight='balanced')
    #dt = DecisionTreeClassifier()   
    # Train Model
    print("training decision tree model...")
    dt.fit(train_x, train_y)
    print("done!")

    # Model Statistics
    print("getting statistics...\n")

    y_trainPred = dt.predict(train_x)
    y_trainProb = dt.predict_proba(train_x)
    
    y_pred = dt.predict(test_x)
    y_predProb = dt.predict_proba(test_x)

    if fieldModelType in "Infield":
        trainStats = get_infield_statistics(train_y, y_trainPred, y_trainProb)
        testStats = get_infield_statistics(test_y, y_pred, y_predProb)
    if fieldModelType in "Outfield":
        trainStats = get_outfield_statistics(train_y, y_trainPred, y_trainProb)
        testStats = get_outfield_statistics(test_y, y_pred, y_predProb)
    

    if (config['LOGGING']['Logs'] == 'True'):
        print("logging statistics...")
        logs.logModel("DecisionTree", dt, trainStats, testStats, [train_x, train_y, test_x, test_y, y_trainPred, y_pred],
                    ["Max Tree Depth: ", max_depth, "Max Tree Features: ", max_features, "Max Leaf Nodes: ", max_leaf_nodes], fieldModelType)
    if (config['LOGGING']['Debug'] == 'True'):
        print("printing statistics...")
        logs.printModel("DecisionTree", dt, trainStats, testStats, [train_x, train_y, test_x, test_y, y_trainPred, y_pred],
                    ["Max Tree Depth: ", max_depth, "Max Tree Features: ", max_features, "Max Leaf Nodes: ", max_leaf_nodes], fieldModelType)
    if (config['LOGGING']['Excel'] == 'True'):
        print("exporting statistics to Excel...")
        logs.ExcelModel("DecisionTree", dt, trainStats, testStats, [train_x, train_y, test_x, test_y, y_trainPred, y_pred],
                    {'Max Tree Depth':max_depth, 'Max Tree Features':max_features, 'Max Leaf Nodes':max_leaf_nodes,
                     'Tree Depth':dt.get_depth(), 'Tree Features': dt.n_features_in_, 'Leaf Nodes': dt.get_n_leaves()}, fieldModelType)
    
    print("done!")

    return dt, trainStats, testStats


# Run Naive Bayes Training and Testing
# Inputs:
    # train_x
    # train_y
    # test_x
    # test_y
    # var_smoothing (ammount of smoothing in the model: 1e-7, 1e-8, 1e-9 [default], 1e-10, 1e-11)
# Ouput:
    # Naive Bayes Model, Training Accuracy, Testing Accuracy
def runNB(train_x, train_y, test_x, test_y, var_smoothing, fieldModelType):
    if (config['MODELS']['NB'] == 'False'):
        return None, None, None
    nb = GaussianNB(var_smoothing=var_smoothing) #class_weight='balanced'
    
    # Train Model
    print("training Naive Bayes model...")
    nb.fit(train_x, train_y)
    print("done!")

    # Model Statistics
    print("getting statistics...")

    y_trainPred = nb.predict(train_x)
    y_trainProb = nb.predict_proba(train_x)

    y_pred = nb.predict(test_x)
    y_predProb = nb.predict_proba(test_x)
    if fieldModelType in "Infield":
        trainStats = get_infield_statistics(train_y, y_trainPred, y_trainProb)
        testStats = get_infield_statistics(test_y, y_pred, y_predProb)
    if fieldModelType in "Outfield":
        trainStats = get_outfield_statistics(train_y, y_trainPred, y_trainProb)
        testStats = get_outfield_statistics(test_y, y_pred, y_predProb)
    
    if (config['LOGGING']['Logs'] == 'True'):
        print("logging statistics...")
        logs.logModel("NaiveBayes", nb, trainStats, testStats, [train_x, train_y, test_x, test_y, y_trainPred, y_pred],
                   ["Var Smoothing: ", var_smoothing], fieldModelType)
    if (config['LOGGING']['Debug'] == 'True'):
        print("printing statistics...")
        logs.printModel("NaiveBayes", nb, trainStats, testStats, [train_x, train_y, test_x, test_y, y_trainPred, y_pred],
                   ["Var Smoothing: ", var_smoothing], fieldModelType)
    if (config['LOGGING']['Excel'] == 'True'):
        print("exporting statistics to Excel...")
        logs.ExcelModel("NaiveBayes", nb, trainStats, testStats, [train_x, train_y, test_x, test_y, y_trainPred, y_pred],{'Var Smoothing':var_smoothing}, fieldModelType)
    
    print("done!")

    return nb, trainStats, testStats

# Run Logistic Regression Training and Testing
# Inputs:
    # train_x
    # train_y
    # test_x
    # test_y
    # lr: learning rate (0 to 1.0)
    # e: epochs (iterations)
# Ouput:
    # Logistic Regression Model, Training Accuracy, Testing Accuracy
def runLogReg(train_x, train_y, test_x, test_y, lr, e, fieldModelType):
    if (config['MODELS']['LR'] == 'False'):
        return None, None, None
    
    logreg = LogisticRegression(C=lr, max_iter=e) #, class_weight='balanced')

    print("training logistic regression model...")
    logreg.fit(train_x, train_y)
    print("done!")

    # Model Statistics
    print("getting statistics...")

    y_trainPred = logreg.predict(train_x)
    y_trainProb = logreg.predict_proba(train_x)

    y_pred = logreg.predict(test_x)
    y_predProb = logreg.predict_proba(test_x)
    if fieldModelType in "Infield":
        trainStats = get_infield_statistics(train_y, y_trainPred, y_trainProb)
        testStats = get_infield_statistics(test_y, y_pred, y_predProb)
    if fieldModelType in "Outfield":
        trainStats = get_outfield_statistics(train_y, y_trainPred, y_trainProb)
        testStats = get_outfield_statistics(test_y, y_pred, y_predProb)
    
    
    if (config['LOGGING']['Logs'] == 'True'):
        print("logging statistics...")
        logs.logModel("LogisticRegression", logreg, trainStats, testStats, [train_x, train_y, test_x, test_y, y_trainPred, y_pred],
                   ["Learning Rate: ", lr, "Epochs: ", e], fieldModelType)
    if (config['LOGGING']['Debug'] == 'True'):
        print("printing statistics...")
        logs.printModel("LogisticRegression", logreg, trainStats, testStats, [train_x, train_y, test_x, test_y, y_trainPred, y_pred],
                    ["Learning Rate: ", lr, "Epochs: ", e], fieldModelType)
    if (config['LOGGING']['Excel'] == 'True'):
        print("exporting statistics to Excel...")
        logs.ExcelModel("LogisticRegression", logreg, trainStats, testStats, [train_x, train_y, test_x, test_y, y_trainPred, y_pred],{"Learning Rate": lr, "Epochs": e}, fieldModelType)
    
    print("done!")

    return logreg, trainStats, testStats

# Run Support Vector Machine Training and Testing
# Inputs:
    # train_x
    # train_y
    # test_x 
    # test_y
    # rC: regularization constant
    # kernel: Kernel type (can be 'linear', 'poly', 'rbf', 'sigmoid')
    # degree: Degree of the polynomial kernel function (ignored by all other kernels)
    # gamma:  Kernel coefficient for 'rbf', 'poly', and 'sigmoid' ('scale')
    # coef0: Independent term in kernel function (0.0)
# Ouput:
    # SVM Model, Training Accuracy, Testing Accuracy
def runSVM(train_x, train_y, test_x, test_y, rC, kernel, degree, gamma, coef0, fieldModelType):
    if (config['MODELS']['SVM'] == 'False'):
        return None, None, None
    
    C = rC  # Regularization parameter

    svm = SVC(C=C, kernel=kernel, degree=degree, gamma=gamma, coef0=coef0, probability=True) # class_weight='balanced'

    print("training SVM model...")
    svm.fit(train_x, train_y)
    print("done!")

    # Model Statistics
    print("getting statistics...")

    y_trainPred = svm.predict(train_x)
    y_trainProb = svm.predict_proba(train_x)

    y_pred = svm.predict(test_x)
    y_predProb = svm.predict_proba(test_x)
    if fieldModelType in "Infield":
        trainStats = get_infield_statistics(train_y, y_trainPred, y_trainProb)
        testStats = get_infield_statistics(test_y, y_pred, y_predProb)
    if fieldModelType in "Outfield":
        trainStats = get_outfield_statistics(train_y, y_trainPred, y_trainProb)
        testStats = get_outfield_statistics(test_y, y_pred, y_predProb)

    if (config['LOGGING']['Logs'] == 'True'):
        print("logging statistics...")
        logs.logModel("SVM", svm, trainStats, testStats, [train_x, train_y, test_x, test_y, y_trainPred, y_pred],
                   ["Regularization Constant: ", rC, "Kernel Type: ", kernel, "Kernel Degree", degree, "Kernel Coefficient (gamma): ", gamma, "Independent Term in Kernel (coef0): ", coef0], fieldModelType)
    if (config['LOGGING']['Debug'] == 'True'):
        print("printing statistics...")
        logs.printModel("SVM", svm, trainStats, testStats, [train_x, train_y, test_x, test_y, y_trainPred, y_pred],
                    ["Regularization Constant: ", rC, "Kernel Type: ", kernel, "Kernel Degree", degree, "Kernel Coefficient (gamma): ", gamma, "Independent Term in Kernel (coef0): ", coef0], fieldModelType)
    if (config['LOGGING']['Excel'] == 'True'):
        print("exporting statistics to Excel...")
        # this because there is an error with the AUC calculations for SVM Models
        trainStat = []
        trainStat.append(trainStats[0])
        trainStat.append(trainStats[1])
        trainStat.append(trainStats[2])
        trainStat.append(trainStats[3])
        trainStat.append([0,0])
        testStat = []
        testStat.append(testStats[0])
        testStat.append(testStats[1])
        testStat.append(testStats[2])
        testStat.append(testStats[3])
        testStat.append([0,0])
        logs.ExcelModel("SVM", svm, trainStat, testStat, [train_x, train_y, test_x, test_y, y_trainPred, y_pred],
                        {"Regularization Constant":rC,"Kernel Type":kernel,"Kernel Degree":degree,"Kernel Coefficient (gamma)":gamma,"Independent Term in Kernel (coef0)":coef0}, fieldModelType)

    print("done!")

    return svm, trainStats, testStats

# Run Random Forest Regressor
# Inputs:
    # train_x
    # train_y
    # test_x
    # test_y
def runRFR(train_x, train_y, test_x, test_y):
    rfr = RandomForestRegressor()
    param_grid = {
        'n_estimators': [100, 200, 400, 800, 1200],
        'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
        'criterion': ['squared_error','absolute_error','friedman_mse','poisson'],
        'min_samples_split': [2, 4, 8, 12],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None],
        'bootstrap': [True, False]
    }
    best_rfr = trainHyperParameters(rfr, param_grid, train_x, train_y)
    best_rfr.fit(train_x, train_y)

    #importances = rfr.feature_importances_
    predictions = best_rfr.predict(test_x)
    directionScore, distanceScore = measurePerformance(predictions, test_y)

    return directionScore, distanceScore

def trainHyperParameters(model, grid, train_x, train_y):
    grid_search = GridSearchCV(estimator=model, param_grid=grid, cv=5, n_jobs=-1, verbose=2)
    grid_search.fit(train_x, train_y)
    return grid_search.best_estimator_

def measurePerformance(predictions, test_y):
    # Values From DataUtil, if you change those: change these, and vise-versa
    ANGLE_RANGE      = 55
    DISTANCE_RANGE   = 450


    direction_bins   = [-float('inf'), -45, -27, -9, 9, 27, 45, float('inf')]

    predictDirection = (predictions[0][0]*ANGLE_RANGE*2) - ANGLE_RANGE
    actualDirection  = (test_y.iloc[0].values[0]*ANGLE_RANGE*2) - ANGLE_RANGE
    errorDirection   = abs(predictDirection - actualDirection)

    predictSlice     = convertAngleToSlice(predictDirection)
    actualSlice      = convertAngleToSlice(actualDirection)

    predictDistance  = predictions[0][1] * 450
    actualDistance   = test_y.iloc[0].values[1] * 450
    errorDistance    = abs(predictDistance - actualDistance)

    directionScore   = 1 - math.sqrt(errorDirection / (2*ANGLE_RANGE))
    distanceScore    = 1 - math.sqrt(errorDistance / DISTANCE_RANGE)

    #print("Direction:\nPredict: ", predictDirection," (", predictSlice,")\tActual: ", actualDirection, " (", actualSlice, "),\tError: ", errorDirection,"\tScore: ", directionScore)
    #print("Distance:\nPredict: ", predictDistance, "\tActual: ", actualDistance, "\tError: ", errorDistance, "\tScore: ",distanceScore,"\n")

    return directionScore, distanceScore

# Function for calculating the weighted score of a model
# Inputs:
    # y_true: actual values
    # y_pred: predicted values
def calculateScore(y_true, y_pred):
    directionWeight = 3
    distanceWeight = 1
    measurePerformance(y_pred, y_true)
    weightedScore = ((directionScore * directionWeight) + (distanceScore * distanceWeight)) / (directionWeight + distanceWeight)
    return weightedScore

def convertAngleToSlice(angle):
    if angle < -27:
        return 1
    elif angle < -9:
        return 2
    elif angle < 9:
        return 3
    elif angle < 27:
        return 4
    else:
        return 5

def colsum(arr, n, m):
    coll = [0] * len(arr)
    for i in range(n):
        su = 0;
        for j in range(m):
            su += arr[j][i]
        coll[i] = su/m
    return coll 



def measurePerformance(predictions, test_y):
    # Values From DataUtil, if you change those: change these, and vise-versa
    ANGLE_RANGE      = 55
    DISTANCE_RANGE   = 450


    direction_bins   = [-float('inf'), -45, -27, -9, 9, 27, 45, float('inf')]

    predictDirection = (predictions[0][0]*ANGLE_RANGE*2) - ANGLE_RANGE
    actualDirection  = (test_y.iloc[0].values[0]*ANGLE_RANGE*2) - ANGLE_RANGE
    errorDirection   = abs(predictDirection - actualDirection)

    predictSlice     = convertAngleToSlice(predictDirection)
    actualSlice      = convertAngleToSlice(actualDirection)

    predictDistance  = predictions[0][1] * 450
    actualDistance   = test_y.iloc[0].values[1] * 450
    errorDistance    = abs(predictDistance - actualDistance)

    directionScore   = 1 - math.sqrt(errorDirection / (2*ANGLE_RANGE))
    distanceScore    = 1 - math.sqrt(errorDistance / DISTANCE_RANGE)

    #print("Direction:\nPredict: ", predictDirection," (", predictSlice,")\tActual: ", actualDirection, " (", actualSlice, "),\tError: ", errorDirection,"\tScore: ", directionScore)
    #print("Distance:\nPredict: ", predictDistance, "\tActual: ", actualDistance, "\tError: ", errorDistance, "\tScore: ",distanceScore,"\n")

    return directionScore, distanceScore

# Function for calculating the weighted score of a model
# Inputs:
    # y_true: actual values
    # y_pred: predicted values
def calculateScore(y_true, y_pred):
    directionWeight = 3
    distanceWeight = 1
    measurePerformance(y_pred, y_true)
    weightedScore = ((directionScore * directionWeight) + (distanceScore * distanceWeight)) / (directionWeight + distanceWeight)
    return weightedScore
    

def convertAngleToSlice(angle):
    if angle < -27:
        return 1
    elif angle < -9:
        return 2
    elif angle < 9:
        return 3
    elif angle < 27:
        return 4
    else:
        return 5

# Function for getting statistics of an infield zone model
# Inputs:
    # pred: predicted values
    # y_test: actual values
# Output:
    # accuracy
def get_infield_statistics(pred, y_test, probs):
    true1 = 0
    true2 = 0
    true3 = 0
    true4 = 0
    true5 = 0

    false1 = 0
    false2 = 0
    false3 = 0
    false4 = 0
    false5 = 0

    totalError = 0

    index = 0
    for i in pred:
        if i == 1:
            if y_test[index] == 1:
                true1 += 1
            else:
                false1 += 1
        if i == 2:
            if y_test[index] == 2:
                true2 += 1
            else:
                false2 += 1
        if i == 3:
            if y_test[index] == 3:
                true3 += 1
            else:
                false3 += 1
        if i == 4:
            if y_test[index] == 4:
                true4 += 1
            else:
                false4 += 1
        if i == 5:
            if y_test[index] == 5:
                true5 += 1
            else:
                false5 += 1
        error = abs(i - y_test[index])
        totalError += error

        index += 1

    totalTrue = true1 + true2 + true3 + true4 + true5
    accuracy = totalTrue / len(y_test)

    # Calculate Recall
    recall = []
    try:
        recall.append(true1 / (true1 + false1))
    except ZeroDivisionError:
        recall.append("No Values")
    try:
        recall.append(true2 / (true2 + false2))
    except ZeroDivisionError:
        recall.append("No Values")
    try:
        recall.append(true3 / (true3 + false3))
    except ZeroDivisionError:
        recall.append("No Values")
    try:
        recall.append(true4 / (true4 + false4))
    except ZeroDivisionError:
        recall.append("No Values")
    try:
        recall.append(true5 / (true5 + false5))
    except ZeroDivisionError:
        recall.append("No Values")
    averageError = totalError / len(y_test)

    # Calculate F1 Scores
    f1_micro = f1_score(y_test, pred, average='micro')
    f1_macro = f1_score(y_test, pred, average='macro')
    f1_weighted = f1_score(y_test, pred, average='weighted')
    f1 = [f1_micro, f1_macro, f1_weighted]

    # For AUC, we need to binarize the labels for multiclass scenario
    y_true_binarized = label_binarize(y_test, classes=[1, 2, 3, 4, 5])

    # AUC (One-vs-Rest)
    # Since AUC is typically used for binary classification, we apply it in a One-vs-Rest manner for multiclass
    try:
        auc_macro = roc_auc_score(y_true_binarized, probs, average='macro', multi_class='ovr')
        auc_weighted = roc_auc_score(y_true_binarized, probs, average='weighted', multi_class='ovr')
        auc = [auc_macro, auc_weighted]
    except:
        auc = "Error"
    return accuracy, averageError, recall, f1, auc


# Function for getting statistics of an infield zone model
# Inputs:
    # pred: predicted values
    # y_test: actual values
# Output:
    # accuracy
def get_outfield_statistics(pred, y_test, probs):
    truearr  = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    falsearr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    totalError = 0

    index = 0
    for i in pred:
        if y_test[index] == i:
            truearr[i-1] += 1
        else:
            falsearr[i-1] += 1
            error = abs(i - y_test[index])
            totalError += error
        index += 1

    totalTrue = sum(truearr)
    accuracy = totalTrue / len(y_test)

    # Calculate Recall
    recall = []
    for i in range(len(truearr)):

        try:
            recall.append(truearr[i] / (truearr[i] + falsearr[i]))
        except ZeroDivisionError:
            recall.append("No Values")
    averageError = totalError / len(y_test)

    # Calculate F1 Scores
    f1_micro = f1_score(y_test, pred, average='micro')
    f1_macro = f1_score(y_test, pred, average='macro')
    f1_weighted = f1_score(y_test, pred, average='weighted')
    f1 = [f1_micro, f1_macro, f1_weighted]

    # For AUC, we need to binarize the labels for multiclass scenario
    y_true_binarized = label_binarize(y_test, classes=[6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])

    # AUC (One-vs-Rest)
    # Since AUC is typically used for binary classification, we apply it in a One-vs-Rest manner for multiclass
    try:
        auc_macro = roc_auc_score(y_true_binarized, probs, average='macro', multi_class='ovr')
        auc_weighted = roc_auc_score(y_true_binarized, probs, average='weighted', multi_class='ovr')
        auc = [auc_macro, auc_weighted]
    except:
        auc = "Error"
    return accuracy, averageError, recall, f1, auc


# will split the data given from the DataFram dF and also compute stats on the splits
# This allows it to be run multiple times
def modelDataSplitting(dF, randomState, testSize, dFType, fieldModelType):

    if("False" in config['DATA']['USE_NEW_PREPROCESSING']):        
        if fieldModelType in "Infield":
            trainingClassSplit = [0, 0, 0, 0, 0] # Section 1 - 5 (5 different sections)
            testingClassSplit = [0, 0, 0, 0, 0]
            Y = dF["FieldSlice"]
        if fieldModelType in "Outfield":
            trainingClassSplit = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] # Section 6 - 20 (15 different sections)
            testingClassSplit = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            Y = dF["FieldSection"]
            
        X = dF[json.loads(config.get('TRAIN',dFType))]
        #X = dF[specific_columns] # pitcher averages
        originalNotNormX = X
        X = DataUtil.normalizeData(X, originalNotNormX)
        xTrain, xTest, yTrain, yTest = train_test_split(X, Y, test_size=testSize, random_state=randomState)
        # adb = AdaBoostClassifier()
        # adb_model = adb.fit(xTrain, yTrain)

        # calculate split information:
        for i in yTrain:
            trainingClassSplit[i-1] += 1

        for i in yTest:
            testingClassSplit[i-1] += 1

        trainingClassPercent = []
        for i in trainingClassSplit:
            trainingClassPercent.append(round(i/len(yTrain), 4))

        testingClassPercent = []
        for i in testingClassSplit:
            testingClassPercent.append(round(i/len(yTest), 4))

        if (config['LOGGING']['Debug'] == 'True'):
            print("Training Class Splits (count, then percentage):")
            print(trainingClassSplit)
            print(trainingClassPercent)
            print("\nTesting Class Splits (count, then percentage):")
            print(testingClassSplit)
            print(testingClassPercent)
    else:
        infieldY = dF[0][['Direction','Distance']]
        infieldX = dF[0][dF[1]] 
        if("True" in config['SPLIT']['TTS']):
            xTrain, xTest, yTrain, yTest = train_test_split(infieldX, infieldY, test_size=0.20, random_state=11)
            
        elif("True" in config['SPLIT']['KFold']):
            kf = KFold(n_splits=5, shuffle=True, random_state=11)
            for train_index, test_index in kf.split(infieldX):
                xTrain, xTest = infieldX.iloc[train_index,:], infieldX.iloc[test_index,:]
                yTrain, yTest = infieldY.iloc[train_index,:], infieldY.iloc[test_index,:]

        elif("True" in config['SPLIT']['LOOCV']):
            loo = LeaveOneOut()
            for train_index, test_index in loo.split(infieldX):
                xTrain, xTest = infieldX.iloc[train_index,:], infieldX.iloc[test_index,:]
                yTrain, yTest = infieldY.iloc[train_index,:], infieldY.iloc[test_index,:]

        else:
            print("No Splitting Method Selected")
            
    return xTrain, xTest, yTrain, yTest
    # GroupKFold: (avoids putting data from the same group in the test set -- useful for Pitcher/Batter ID when we implement that.)