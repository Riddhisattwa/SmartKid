import sklearn
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import pprint

#data=pd.read_csv('data/fruit_data.txt', delimiter="\t")
data=load_breast_cancer()
#pprint.pprint(data)

label_names=data['target_names']
labels=data['target']
feature_names=data['feature_names']
features=data['data']


# Look at our data
print(label_names)
print(labels[0])
print(feature_names[0])
print(features[0])

#Split the data
train,test,train_labels,test_labels=train_test_split(features,labels,test_size=0.33,random_state=42)

#Classifier
gnb=GaussianNB()

#Training
model=gnb.fit(train,train_labels)

#Make prediction
preds=gnb.predict(test)
print(preds)

#Evaluate accuracy
print(accuracy_score(test_labels,preds))
print(gnb.score(test,test_labels))