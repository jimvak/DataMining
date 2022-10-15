import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import numpy as np
from pandas.plotting import scatter_matrix
from matplotlib import cm
import seaborn as sns
import math
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score





mydata = pd.read_csv('healthcare-dataset-stroke-data.csv')


feature_names = ['gender', 'age', 'hypertension', 'heart_disease', 'ever_married','work_type','Residence_type','avg_glucose_level','bmi','smoking_status']
X = mydata[feature_names]

y = mydata['stroke']

#efarmogi labelencoder

le = LabelEncoder()

X['gender']= le.fit_transform(X['gender'])

X['ever_married']= le.fit_transform(X['ever_married'])

X['work_type']= le.fit_transform(X['work_type'])

X['Residence_type']= le.fit_transform(X['Residence_type'])

X['smoking_status']= le.fit_transform(X['smoking_status'])


sum_bmi=0

count_bmi=0

for x in X['bmi']:
    if  not math.isnan(x):
        sum_bmi=sum_bmi+x
        count_bmi=count_bmi+1
        
        
avg_bmi = sum_bmi/count_bmi

print(avg_bmi)

for i in range(len(X['bmi'])):
    if math.isnan(X['bmi'][i]):
        X['bmi'][i]=avg_bmi
        



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)


# #dimiourgia montelou
clf=RandomForestClassifier(n_estimators=6000)

# #ekpaideysi
clf.fit(X_train,y_train)

# #test
y_pred=clf.predict(X_test)

precision = precision_score(y_test, y_pred, average='binary')

print('Precision: %.3f' % precision)

recall = recall_score(y_test, y_pred, average='binary')
print('Recall: %.3f' % recall)

score = f1_score(y_test, y_pred, average='binary')
print('F-Measure: %.3f' % score)
