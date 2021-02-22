import numpy as np
import pandas as pd
from helper import cleanText , readFromFile ,plotData , preprocessLabels , scale , Point
import math
import random
from sklearn.metrics import homogeneity_score,completeness_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

random_state = 0
class Clarans:
    #constructor
    def __init__(self,datasetFileName,numMed,numLocal,maxNeighbor,filterFlag):
        self.fileName = datasetFileName
        self.numOfMedoids = numMed
        self.numlocal = numLocal
        self.maxneighbor = maxNeighbor
        self.inputPointsFile = None
        self.filterFlag = filterFlag
        self.numberLabels = None
        self.randomState = 0

        self.mincost = 10000000000000
        self.objects = None  # kümelenecek nesnelerin listesinin başlatılması.
        self.pointIndexes = None
        self.localBestMedoids = []  # local en iyi medoidler
        self.bestMedoids = []  # algoritma sonu en iyi medoid listesi
        # pointer arası tüm mesafelerin depolanması
        self.distanceMatrix = None
        self.labels = None
        self.predictedLabels = None
        #self.preProcess()


    def run2(self):
        df = pd.read_csv(self.fileName)
        df = df[pd.notnull(df['tags'])]
        df['post'] = df['post'].apply(cleanText)
        df = df.sort_values(by='tags')
        self.numberLabels = df['tags'].value_counts().size
        df['tags'] = pd.factorize(df.tags)[0]
        self.labels = df['tags']

        vec = TfidfVectorizer(stop_words="english",max_features=100)
        vec.fit(df.post.values)
        features = vec.transform(df.post.values)

        objects = []
        ind = 0
        for coordinates in features.toarray():
            point = Point()
            point.coordinates = coordinates.tolist()
            point.orgLabel = df.tags.values[ind]
            objects.append(point)
            ind += 1

        self.objects = objects
        self.pointIndexes = len(self.objects)
        self.distanceMatrix = np.asmatrix(np.zeros((self.numOfMedoids, self.pointIndexes)))
        medoids, objects = self.execute()

        pca = PCA(n_components=2, random_state=random_state)
        reduced_features = pca.fit_transform(features.toarray())
        reduced_cluster_centers = pca.transform(self.getCenters(medoids))

        plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=self.getPredicted())
        plt.scatter(reduced_cluster_centers[:, 0], reduced_cluster_centers[:, 1], marker='o', s=100, c='b')

        plt.show()

        ind = 0
        for obj in reduced_features:
            self.objects[ind].x = obj[0]
            self.objects[ind].y = obj[1]
            ind += 1

        print("ScikitLearn Homogenity:" + str(homogeneity_score(df.tags, self.getPredicted())))
        print("ScikitLearn Completeness:" + str(completeness_score(df.tags, self.getPredicted())))
        print("ScikitLearn Silhouette Score:" + str(silhouette_score(features, labels=self.getPredicted())))

        return medoids,objects

    def getCenters(self,indexs):
        result = []
        for objIndx in range(len(self.objects)):
            for index in indexs:
                if index == objIndx:
                    result.append(np.array(self.objects[objIndx].coordinates))
        return np.array(result)

    def getPredicted(self):
        result = np.array([int(point.cluster) for point in self.objects])
        return pd.factorize(result)[0]

    def setOrgLabels(self):
        ind = 0
        for obj in self.objects:
            obj.orgCluster = int(self.labels[ind])
            ind += 1

    def execute(self):
        i = 1
        while i <= self.numlocal: # numLocal threshold
            # distance matrixi sıfırla doldur
            self.distanceMatrix.fill(0)
            self.localBestMedoids = []

            temp_medoids = self.getRandomMedoids() # random medoids secme
            self.setDistance(temp_medoids) # matrisi guncelle
            self.assigningToClusters(temp_medoids) # tum objelerı en yakın medoıdıne kumelee
            cost = self.getTotalDistance(temp_medoids) # cost hesapla

            j = 1
            while j <= self.maxneighbor: # maximum komsu sayısı
                newMedoidIndx = self.getRandomNeighbor(temp_medoids) # random komsu alma
                self.replaceRandomMedoids(temp_medoids, newMedoidIndx) #

                # distance matrisini guncelle
                self.updateDistanceMatrixForNewMedoids(temp_medoids, newMedoidIndx)
                self.assigningToClusters(temp_medoids) # tum objeler yenı clusterlara
                newCost = self.getTotalDistance(temp_medoids) # cost hesapla

                if newCost < cost: # yenı cost kucukse mevcut dongu basına
                    self.localBestMedoids = temp_medoids.copy()
                    cost = newCost
                    j = 1
                    continue
                else:
                    j += 1
                    if j <= self.maxneighbor: # max neighbor ulasmamıssa mevcut dongu basına
                        continue
                    elif cost < self.mincost:
                        self.mincost = cost #update mincost
                        self.bestMedoids = self.localBestMedoids.copy() # sonuc medoıdlerını update et
                        print("Now minimal cost: {} ".format(self.mincost))

                    i += 1
                    if i > self.numlocal: # threshold asılmıssa execute biter
                        self.setDistance(self.bestMedoids)
                        self.assigningToClusters(self.bestMedoids)
                        print("\nMinimal cost: {} \n".format(self.mincost))
                        return self.bestMedoids, self.objects # medoıdler ve objectler return edılır

    # İlk rastgele medoidler olarak işlev gören nesnelerin rastgele dizinlerini döndürür.
    def getRandomMedoids(self):
        return random.sample(range(self.pointIndexes), self.numOfMedoids)

    def getDist(self, objA, objB):
        return self.getDistanceOfPoints(objA, objB)

    @staticmethod
    def getDistanceOfPoints(Point1, Point2):
        return math.sqrt(sum([(float(a) - float(b)) ** 2 for a, b in zip(Point1.coordinates, Point2.coordinates)]))

    # Distance_matrix dizisini hesaplanan uzaklıklarla doldurur.
    def setDistance(self, medoidIndexes):
        for medoidIndex in medoidIndexes:
            for pointIndx in range(self.pointIndexes):
                self.distanceMatrix[medoidIndexes.index(medoidIndex), pointIndx] = self.getDist(self.objects[medoidIndex], self.objects[pointIndx])

    # Medoide en yakın nesneleri, daha spesifik olarak en yakın medoidin indeksini atar.
    def assigningToClusters(self, medoidIndxs):
        for pointIndx, obj in enumerate(self.objects):
            d = 10000000000000
            idx = pointIndx
            for medoidIndx in medoidIndxs:
                distance = self.distanceMatrix[medoidIndxs.index(medoidIndx), pointIndx]
                if distance < d:
                    d = distance
                    idx = medoidIndx
            obj.cluster = idx

    # Toplam maliyeti, yani nesnelerin en yakın medoidlerine olan mesafelerinin toplamını hesaplar.
    def getTotalDistance(self, medoidIndxs):
        totalDistance = 0
        for pointIndx, obj in enumerate(self.objects):
            totalDistance += self.distanceMatrix[medoidIndxs.index(obj.cluster), pointIndx]
        return totalDistance

    def getRandomNeighbor(self, medoidsIndxs):
        newMedoidIndex = random.randrange(0, self.pointIndexes, 1)
        while newMedoidIndex in medoidsIndxs:
            newMedoidIndex = random.randrange(0, self.pointIndexes, 1)

        return newMedoidIndex


    def replaceRandomMedoids(self, medoidIndxs, newMedoidIndx):
        medoidIndxs[random.randrange(0, len(medoidIndxs))] = newMedoidIndx

    # yeni medoidler icin distance matrisi update edilir
    def updateDistanceMatrixForNewMedoids(self, medoidIndxs, newMedoidIndx):
        for pointIndx in range(self.pointIndexes):
            self.distanceMatrix[medoidIndxs.index(newMedoidIndx), pointIndx] = self.getDist(self.objects[newMedoidIndx], self.objects[pointIndx])


