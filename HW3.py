import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import confusion_matrix, classification_report 

dataset = pd.read_csv('diabetes.csv')
dataset.head(20)
x = dataset.iloc[:,:7]
y = dataset.iloc[:,8]

from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(x,y,train_size = 0.8, random_state = 0)

from sklearn.preprocessing import StandardScaler 
sc_X = StandardScaler() 
X_train = sc_X.fit_transform(X_train) 
X_test = sc_X.transform(X_test) 

logReg = LogisticRegression()
prediction = logReg.predict(X_test)
cfm= confusion_matrix(y_test,prediction)
print(classification_report(y_test, prediction)) #END OF NUMBER 1

from sklearn.naive_bayes import GaussianNB 
classifier = GaussianNB() 
classifier.fit(X_train, y_train) 
y2_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y2_pred) 
ac = accuracy_score(y_test, y2_pred) 
print('Accuracy of the naive bayes is: ', ac)
sns.heatmap(cm, annot = True)
print(classification_report(y_test, y2_pred))
#END OF NUMBER 2



dataset2 = pd.read_csv('diabetes.csv')
skf = StratifiedKFold(n_splits=10)
model = LogisticRegression()

x = dataset2.iloc[:,:7]
y = dataset2.iloc[:,8]

from sklearn.preprocessing import StandardScaler 
sc_Xk = StandardScaler() 
xk = sc_X.fit_transform(x) 



from sklearn.model_selection import train_test_split
X_traink,X_testk,y_traink,y_test = train_test_split(xk,y,test_size=1/3,random_state=42, stratify=y)


X_testk
seed = 7

results = model_selection.cross_val_score(model, x, y, cv=skf)
print("Accuracy of K: %.3f%% (%.3f%%)" % (results.mean()*100.0, results.std()*100.0))


