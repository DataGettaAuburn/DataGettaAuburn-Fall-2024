# Imports
from Models import ModelUtil
from Data import Preprocessing, DataUtil
from Logs import logging as logs
import numpy as np

import importlib
import configparser
import pickle

config = configparser.ConfigParser()
config.read('Data//config.ini')

importlib.reload(Preprocessing)
importlib.reload(ModelUtil)
importlib.reload(logs)

import warnings
warnings.filterwarnings("ignore")

infieldDataFrame = []
outfieldDataFrame = []
models = []

def loadData():
    # 1) Load all data from preprocessing 
    newprocessing = 'True' in config['DATA']['USE_NEW_PREPROCESSING']
    infieldDataFrame, outfieldDataFrame = Preprocessing.dataFiltering([], newprocessing)
    return infieldDataFrame, outfieldDataFrame


# Function to train all models based on settings from config
def trainModels(infieldDataFrame, outfieldDataFrame):
    models = {}
    # 2) Trains all Models and exports all data to an Excel Sheet
    max_depth = 50
    max_features = 30
    max_leaf_nodes = 150
    # could also add ways to change it for these hyperparams below for other models
    var_smoothing = 1e-9
    lr = 0.8
    e = 100
    rC = 1
    kernel='linear'
    degree= 1
    gamma= 'scale'
    coef0= 0.0

    runCount = int(config['TRAIN']['TimesRun'])
    if ("False" in config['TRAIN']['Testing']):
        runCount = 1
        print("Not Testing")
    for j in range(1, runCount+1):
            xTrain, xTest, yTrain, yTest = ModelUtil.modelDataSplitting(infieldDataFrame, j, 0.25,'InfieldTrainingFilter', 'Infield')
            xoTrain, xoTest, yoTrain, yoTest = ModelUtil.modelDataSplitting(outfieldDataFrame, j, 0.25,'InfieldTrainingFilter', 'Outfield')

            if("True" in config['MODELS']['DTC']):
                dtOutput = ModelUtil.runDT(xTrain, yTrain, xTest, yTest, max_depth, max_features, max_leaf_nodes, "Infield")
                dtoOutput = ModelUtil.runDT(xoTrain, yoTrain, xoTest, yoTest, max_depth, max_features, max_leaf_nodes, "Outfield")
                models["DTI"] = dtOutput
                models["DTO"] = dtoOutput
                if ("True" in config['DATA']['Pickle']):
                    # Save the model to a file
                    with open('Models/InfieldDecisionTree.pkl', 'wb') as file:
                        pickle.dump(dtOutput, file)
                    with open('Models/OutfieldDecisionTree.pkl', 'wb') as file:
                        pickle.dump(dtoOutput, file)

            if("True" in config['MODELS']['NB']):   
                nbOutput = ModelUtil.runNB(xTrain, yTrain, xTest, yTest, var_smoothing, 'Infield')
                nboOutput = ModelUtil.runNB(xoTrain, yoTrain, xoTest, yoTest, var_smoothing, 'Outfield')
                models["NBI"] = nbOutput
                models["NBO"] = nboOutput
                if ("True" in config['DATA']['Pickle']):
                    # Save the model to a file
                    with open('Models/InfieldNaiveBayes.pkl', 'wb') as file:
                        pickle.dump(nbOutput, file)
                    with open('Models/OutfieldNaiveBayes.pkl', 'wb') as file:
                        pickle.dump(nboOutput, file)

            if("True" in config['MODELS']['LR']):
                logRegOutput = ModelUtil.runLogReg(xTrain, yTrain, xTest, yTest, lr, e, "Infield")
                logRegoOutput = ModelUtil.runLogReg(xoTrain, yoTrain, xoTest, yoTest, lr, e, "Outfield")
                models["LRI"] = logRegOutput
                models["LRO"] = logRegoOutput
                if ("True" in config['DATA']['Pickle']):
                    # Save the model to a file
                    with open('Models/InfieldLogRegression.pkl', 'wb') as file:
                        pickle.dump(logRegOutput, file)
                    with open('Models/OutfieldLogRegression.pkl', 'wb') as file:
                        pickle.dump(logRegoOutput, file)

            if("True" in config['MODELS']['SVM']):
                svmOutput = ModelUtil.runSVM(xTrain, yTrain, xTest, yTest, rC, kernel, degree, gamma, coef0, 'Infield')
                svmoOutput = ModelUtil.runSVM(xoTrain, yoTrain, xoTest, yoTest, rC, kernel, degree, gamma, coef0, 'Outfield')
                models["SVMI"] = svmOutput
                models["SVMO"] = svmoOutput
                if ("True" in config['DATA']['Pickle']):
                    # Save the model to a file
                    with open('Models/InfieldSVM.pkl', 'wb') as file:
                        pickle.dump(svmOutput, file)
                    with open('Models/OutfieldSVM.pkl', 'wb') as file:
                        pickle.dump(svmoOutput, file)

            # if("True" in config['MODELS']['RF']):
            #     for i in range(0, len(trainIn)):
            #         direction, distance = ModelUtil.runRFR(trainIn[i], trainOut[i], testIn[i], testOut[i])
    return models

# Function to load models from their pickle files
def loadModels():
    models = {}
    # Load the models from the files
    if("True" in config['MODELS']['DTC']):
        with open('Models/InfieldDecisionTree.pkl', 'rb') as file:
            dt = pickle.load(file)
            models["DTI"] = dt
        with open('Models/OutfieldDecisionTree.pkl', 'rb') as file:
            dto = pickle.load(file)
            models["DTO"] = dto

    if("True" in config['MODELS']['NB']):   
        with open('Models/InfieldNaiveBayes.pkl', 'rb') as file:
            nb = pickle.load(file)
            models["NBI"] = nb
        with open('Models/OutfieldNaiveBayes.pkl', 'rb') as file:
            nbo = pickle.load(file)
            models["NBO"] = nbo

    if("True" in config['MODELS']['LR']):
        with open('Models/InfieldLogRegression.pkl', 'rb') as file:
            lr = pickle.load(file)
            models["LRI"] = lr
        with open('Models/OutfieldLogRegression.pkl', 'rb') as file:
            lro = pickle.load(file)
            models["LRO"] = lro

    if("True" in config['MODELS']['SVM']):
        with open('Models/InfieldSVM.pkl', 'rb') as file:
            svm = pickle.load(file)
            models["SVMI"] = svm
        with open('Models/OutfieldSVM.pkl', 'rb') as file:
            svmo = pickle.load(file)
            models["SVMO"] = svmo
    
    return models

# Function to output all average pitcher photos from 'Data/PitchMetricAverages_AsOf_2024-03-11.csv'
def outputPitcherAverages(data, pitchingAveragesDF, models):
    predictionKey = []
    predictions = []
    predictionso = []
    for index in range(data.shape[0]):
        if index != 0:
            averageProbs, averageProbso, error = predictSinglePitcherStat(data.iloc[index], models) 
            if error == True:
                print(pitchingAveragesDF.iloc[index]["Pitcher"])
            else:
                player = pitchingAveragesDF.iloc[index][0].replace(", ", "")
                pitch = pitchingAveragesDF.iloc[index]["TaggedPitchType"]
                batterSide = pitchingAveragesDF.iloc[index]["BatterSide"]
                team = pitchingAveragesDF.iloc[index]["PitcherTeam"]

                predictionKey.append([player,pitch,batterSide,team])
                predictions.append(averageProbs)
                predictionso.append(averageProbso)

    # batch_image_to_excel.create_excel() 

    # predictions holds the model prediction outputs
    # predictionKey holds the player, pitch, and batter side information for the corresponding index in the predictions
    return predictionKey, predictions, predictionso

def predictSinglePitcherStat(dataPoint, models):
    averageProbs = []
    averageProbso= []
    modelTypeCount = 0
    averageProbs.append([0,0,0,0,0])
    averageProbso.append([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
    error = False
    # For each selected model (config), add in the predicted probabilities
    if("True" in config['MODELS']['DTC']):
        dt = models["DTI"][0]
        dto = models["DTO"][0]
        try:
            averageProbs += dt.predict_proba([dataPoint])[0]
            averageProbso += dto.predict_proba([dataPoint])[0]
            modelTypeCount += 1
        except:
            error = True

    if("True" in config['MODELS']['NB']):   
        nb = models["NBI"][0]
        nbo = models["NBO"][0]
        try:
            averageProbs += nb.predict_proba([dataPoint])[0]
            averageProbso += nbo.predict_proba([dataPoint])[0]
            modelTypeCount += 1
        except:
            error = True

    if("True" in config['MODELS']['LR']):    
        logReg = models["LRI"][0]
        logRego = models["LRO"][0]
        try:
            averageProbs += logReg.predict_proba([dataPoint])[0]
            averageProbso += logRego.predict_proba([dataPoint])[0]
            modelTypeCount += 1
        except:
            error = True

    if("True" in config['MODELS']['SVM']):
        svm = models["SVMI"][0]
        svmo = models["SVMO"][0]
        try:
            averageProbs += svm.predict_proba([dataPoint])[0]
            averageProbso += svmo.predict_proba([dataPoint])[0]
            modelTypeCount += 1
        except:
            error = True

    # Average the selected model's probabilities 
    averageProbs = averageProbs / modelTypeCount
    averageProbso = averageProbso / modelTypeCount

    return averageProbs, averageProbso, error


# Run this every monday:
# Load the data and the models (train models every monday)
infieldDataFrame, outfieldDataFrame = loadData() # replace CSV data with all current data (including new weekly data)
models = trainModels(infieldDataFrame, outfieldDataFrame) # re-train models on new data (based on config)
    # models = loadModels() # If you do not want to retrain the models run this instead of train

# Connect to database and pull pitcher averages from SQL
cur, conn = DataUtil.databaseConnect()
averagesData, pitchingAveragesDF = DataUtil.getPitcherAverages(cur, infieldDataFrame, outfieldDataFrame, "None")

# Run pitcher average predictions
predictionKey, predictionsIn, predictionso = outputPitcherAverages(averagesData, pitchingAveragesDF, models) # change this to output predictions to the sql database to be read in and visualized when opening that players page

predictions = predictionsIn+predictionso
# combined list where first 5 values are the infield predictions, and the next 15 are the outfield predictions
combined_list = [np.concatenate([a, b], axis=1) for a, b in zip(predictionsIn, predictionso)]

# write to defensive_shift_model_values view (30+ minutes)
DataUtil.writePitcherAverages(cur, conn, predictionKey, combined_list)

# Close the connection
cur.close()
conn.close()