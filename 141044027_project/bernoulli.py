import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from math import log
import numpy as np
from nltk.corpus import stopwords

class Bernoulli:
    #constructor
    def __init__(self, datasetFileName,maxFeature):
        self.fileName = datasetFileName
        # Verikümeleri
        self.X_train = None
        self.X_test = None
        # Labellar
        self.y_train = None
        self.y_test = None

        self.priors = None # Labellar
        self.numLabel = None # Label Sayısı
        self.maxFeature = maxFeature # Vectorizationda max vector uzunluğı


    def evaluation(self,train,test,filterFlag):
        print("::::::::::::: BERNOULLI TEST :::::::::::::::::")
        #dataframe yapısına alma
        train = pd.DataFrame(train,columns=['post','tags'])
        test = pd.DataFrame(test,columns=['post','tags'])

        # Vectorization için hazır hale getrlir
        X_train = train.post
        X_test = test.post
        y_train = train.tags.values
        y_test = test.tags.values

        # feature selection yoksa vectorization olacak
        # feature selection varsa kendi metodunda oluyor
        if not filterFlag:
            vectorizer = CountVectorizer(max_features=self.maxFeature, min_df=5, max_df=0.7, stop_words=stopwords.words('english'),
                                         binary=True)
            X_train = vectorizer.fit_transform(X_train).toarray()
            X_test = vectorizer.transform(X_test).toarray()

        self.checkMaxFeature(X_train) # vectorun uzunluk kontrolu maxtan az olmamalı
        train, priors = self.trainBernoulli(y_train, X_train) # train model
        results = self.applyBernoulli(train, priors, X_test) # test model

        accuracy = 0
        for ind in range(len(results)):
            if results[ind][1] == y_test[ind]:
                accuracy += 1

        print("Accuracy : " + str(accuracy / len(results)))

        return [k[1] for k in results]

    def checkMaxFeature(self,data):
        if len(data[0]) < self.maxFeature:
            self.maxFeature = len(data[0])

    def applyBernoulli(self, train, priors, testDatasetBinary):
        results, score = self.calculateTermsFromDocuments(testDatasetBinary)
        print("Apply Bernoulli NB")
        for i in range(len(testDatasetBinary)):
            for j in range(len(priors)):
                if(priors[j] > 0):
                    score[j] = log(priors[j])
                for k in range(self.maxFeature):
                    if testDatasetBinary[i][k] == 1:
                        score[j] += log(train[j][k])
                    else:
                        score[j] += log(1. - train[j][k])
            results[i] = [i + 1, np.argmax(score)]
            #print(i)

        return results

    def trainBernoulli(self, labelTrain, dataTrainBi):
        train, priors = self.extractVocabularyAndCountDocs(labelTrain, dataTrainBi)
        print("Train Bernoulli NB")

        for i in range(self.numLabel):
            for j in range(self.maxFeature):
                train[i][j] = (train[i][j] + 0.01) / (priors[i] + 0.02) #smootie
            priors[i] = priors[i] / len(labelTrain)

        return train, priors


    def calculateTermsFromDocuments(self, testDatasetBi):

        results = [[0] * 2 for i in range(len(testDatasetBi))]
        score = [0] * self.numLabel
        return results, score

    def extractVocabularyAndCountDocs(self, labelTrain, dataTrainBi):

        trainMatris = [[0] * self.maxFeature for _ in range(self.numLabel)]
        labels = [0] * self.numLabel
        for i in range(len(labelTrain)):
            labels[labelTrain[i]] = labels[labelTrain[i]] + 1
            for j in range(self.maxFeature):
                trainMatris[labelTrain[i]][j] = trainMatris[labelTrain[i]][j] + dataTrainBi[i][j]

        return trainMatris, labels

    ############################################################


