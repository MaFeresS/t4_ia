import numpy as np
import sklearn.metrics as skm
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

CLASSIFIER=['mlp','svm','rf']
DATAFILE=['clip','dinov2','resnet34']

keys=[]
acc_out=[]
cm_store=[]
color=["orange","orange","orange","blue","blue","blue","green","green","green"]

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
            param_grid={'C': [1,10,15,20],'gamma': [0.001,0.005,0.01]}
            model = GridSearchCV(SVC(kernel='rbf'),param_grid=param_grid,cv=5,n_jobs=-1)
        elif m=='rf':
            model =RandomForestClassifier(random_state=10,n_estimators=200)
        print(len(train_file),len(train_labels))
        labels_=model.fit(train_file,train_labels).predict(val_file)
        acc=skm.accuracy_score(val_labels,labels_)
        print(labels_,acc)
        keys.append(f"{m}_{d}")
        acc_out.append(acc)
        cm_store.append(skm.confusion_matrix(val_labels,labels_))
        if m=='svm':
            print(f"best svm: {model.best_params_}")

plt.bar(keys,acc_out,color=color)
plt.title("Accuracy")
plt.xticks(rotation=25)
plt.show()

best_i=None
max=0
for i,a in enumerate(acc_out):
    if a>max:
        max=a
        best_i=i
print(best_i,max)
print("\n")
print(f"{keys[best_i]}\n{cm_store[best_i]}")