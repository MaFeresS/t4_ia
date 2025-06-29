import numpy as np
import sklearn.metrics as skm
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

CLASSIFIER=['mlp','svm','rf']
DATAFILE=['clip','dinov2','resnet34']

for d in DATAFILE:

    train_file=np.load(f"train_{d}_VocPascal.npy")
    with open("VocPascal/train_voc.txt","r") as labelfile:
        labels=[]
        for i in labelfile.readlines():
            labels.append(i.split("\t")[1])
    train_labels=np.array(labels)

    val_file=np.load(f"feat_{d}_VocPascal.npy")
    with open("VocPascal/val_voc.txt","r") as labelfile:
        labels=[]
        for i in labelfile.readlines():
            labels.append(i.split("\t")[1])
    val_labels=np.array(labels)

    for m in CLASSIFIER:
        print(f"MODEL: {m}, ENCODER: {d}")
        if m=='mlp':
            model = MLPClassifier((256,),'relu',random_state=10,max_iter=300)
        elif m=='svm':
            model = SVC(kernel='rbf')
        elif m=='rf':
            model =RandomForestClassifier(random_state=10)
        print(len(train_file),len(train_labels))
        mod=model.fit(train_file,train_labels)
        labels_=mod.predict(val_file)
        acc=skm.accuracy_score(val_labels,labels_)
        print(labels_,acc)