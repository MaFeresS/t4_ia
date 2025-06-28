import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import torch
import torch.nn as nn

CLASSIFIER=['mlp','svm','rf']

for m in CLASSIFIER:
    if m=='mlp':
        model = MLPClassifier((256,),'relu',random_state=10)
    elif m=='svm':
        model = SVC(kernel='rbf')
    elif m=='rf':
        model =RandomForestClassifier(random_state=10)