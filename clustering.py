import numpy as np
from pyannote.core import Annotation,Segment, Timeline
from pyannote.audio.embedding.utils import cdist

class Cluster():
    def __init__(self, label, dist_metric=cdist):
        self.label = label
        self.representation = 0
        self.dist_metric = cdist
        self.embeddings = []
        self.segments = []
        
    def distance(self,data):
        feature = np.sum(data['embedding'], axis=0, keepdims=True)
        return self.dist_metric(self.representation, feature, metric='cosine')

    def distanceModel(self,model):
        return self.dist_metric(self.representation, model, metric='cosine')
    
    def updateCluster(self,data):
        self.embeddings.append(data['embedding'])
        self.representation += np.sum(data['embedding'], axis=0, keepdims=True)
        self.segments.append(data['segment'])
        return
    
    def mergeClusters(self,cluster):
        self.embeddings.update(cluster.embeddings)
        self.segments.update(cluster.segments)
        return


class OnlineClustering():
    def __init__(self, uri, threshold=0.5,
                generator_method='string'):
        #pooling_func, distance, 
        self.uri = uri
        self.threshold = threshold
        # store clusters by dict next
        self.clusters = []
        self.generator_method = generator_method
        #self.annotations = Annotation(uri=self.uri)
        
        if self.generator_method == 'string':
            from pyannote.core.util import string_generator
            self.generator = string_generator()
        elif self.generator_method == 'int':
            from pyannote.core.util import int_generator
            self.generator = int_generator()

    
    def getLabels(self):
        """
        return all the cluster labels
        """
        return [cluster.label for cluster in self.clusters]
    
    def getAnnotations(self):
        """
        return annotations of cluster result
        todo: add warning when clusters is empty
        """
        annotation = Annotation(uri=self.uri, modality='speaker')
        for cluster in self.clusters:
            for seg in cluster.segments:
                annotation[seg] = cluster.label
        
        return annotation
    
    def addCluster(self,data):
        label = next(self.generator)
        cluster = Cluster(label)
        cluster.updateCluster(data)
        self.clusters.append(cluster)
        return
        
    def computeDistances(self, data):
        """Compare new coming data with clusters"""
        return [cluster.distance(data) for cluster in self.clusters]
        
    
    def upadateCluster(self,data):
        """add new coming data to clustering result"""
        if len(self.clusters) == 0:
            self.addCluster(data)
            return
        
        distances = self.computeDistances(data)
        if min(distances) > self.threshold:
            self.addCluster(data)
        else:
            indice = distances.index(min(distances))
            to_update_cluster = self.clusters[indice]
            to_update_cluster.updateCluster(data)
        return



class OnlineOracleClustering():
    def __init__(self, uri, threshold=0.5,
                generator_method='string'):
        #pooling_func, distance, 
        self.uri = uri
        self.clusters = {}
        #self.annotations = Annotation(uri=self.uri)
    
    def getLabels(self):
        """
        return all the cluster labels
        """
        return cluster.keys()
    
    def getAnnotations(self):
        """
        return annotations of cluster result
        todo: add warning when clusters is empty
        """
        annotation = Annotation(uri=self.uri, modality='speaker')
        for cluster in self.clusters:
            for seg in cluster.segments:
                annotation[seg] = cluster.label
        
        return annotation
    
    def addCluster(self,data):
        label = data['label']
        cluster = Cluster(label)
        cluster.updateCluster(data)
        self.clusters[label] = cluster
        return
        
    def computeDistances(self, data):
        """Compare new coming data with clusters"""
        distances = []
        for label in self.clusters:
            distances.append(self.clusters[label].distance(data))
        return distances

    def modelDistance(self, model):
        distances = []
        for label in self.clusters:
            distances.append(self.clusters[label].distanceModel(model))
        return distances
    
    def modelsDistances(self, models):

        distances = {}
        for label, model in models.items():
            distances[label] = min(self.modelDistance(model))

        return distances

    
    def upadateCluster(self,data):
        """add new coming data to clustering result"""
        if data['label'] in self.clusters:
            self.clusters[data['label']].updateCluster(data)
        else:
            self.addCluster(data)
        return
    
    def empty(self):
        if len(self.clusters)==0:
            return True
        return False