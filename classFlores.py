# import pandas
# Escritorio=pandas.read_csv('IRIS.csv')

print("*************** PANDAS FOR DATA MANIPULATIONS *****************")
import pandas as pd 
Escritorio=pd.read_csv('IRIS.csv')
# print(Escritorio.head(10))
# print(Escritorio.tail(10))
# print(Escritorio.info())
# print(Escritorio.shape)
print(Escritorio.groupby('species').size())
print(Escritorio.head(5))

print("*************** MATPLOTLIB FOR PLOTTING *****************")
# print(Escritorio.head(5))
# print(Escritorio['species'])
# print(Escritorio['sepal_length'])
setosa=Escritorio[Escritorio['species']=='Iris-setosa']
versicolor=Escritorio[Escritorio['species']=='Iris-versicolor']
virginica=Escritorio[Escritorio['species']=='Iris-virginica']

# print(setosa.head(5))
# print(versicolor.head(5))
print(virginica.head(5))

print(setosa.shape)
print(versicolor.shape)
print(virginica.shape)

import matplotlib.pyplot as plt
# fig,ax=plt.subplots()
# fig.set_size_inches(5,4)

# ax.scatter(setosa['sepal_length'],setosa['sepal_width'],facecolor='red')
# ax.scatter(versicolor['sepal_length'],versicolor['sepal_width'],facecolor='green')
# ax.scatter(virginica['sepal_length'],virginica['sepal_width'],facecolor='blue')

fig,ax=plt.subplots()
fig.set_size_inches(5,4)
ax.scatter(setosa['petal_length'],setosa['petal_width'],facecolor='red')
ax.scatter(versicolor['petal_length'],versicolor['petal_width'],facecolor='green')
ax.scatter(virginica['petal_length'],virginica['petal_width'],facecolor='blue')
ax.set_xlabel('petal_length')
ax.set_ylabel('petal_width')
ax.grid()
ax.set_title('IRIS PETALS')
# plt.show()

print("*************** PERFOMING CLASSIFICATION *****************")
print(Escritorio.head(5))
x=Escritorio.drop(['species'],axis=1)
print(x.head(5))
print(type(x))
x=x.to_numpy()
print(type(x))

x=x[:,(2,3)]
print(x[:5])

y=[]

# print(len(Escritorio['species']))
for i in range(len(Escritorio['species'])):
    if Escritorio['species'][i]=='Iris-setosa':
        y.append(0)
    elif Escritorio['species'][i]=='Iris-versicolor':
        y.append(1)
    else:
        y.append(2)
# print(y)

from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test=train_test_split(x,y,test_size=0.6)

from sklearn.linear_model import LogisticRegression

# # Argoritmos De maching Leraning
model=LogisticRegression()

print("*************** TRAINING STEP *****************")

model.fit(x_train,y_train)

print("*************** TEST PREDICTIONS *****************")

y_predictions=model.predict(x_test)
# # print(y_predictions[:30])
# # print(y_test[:30])

# # Metodo de clasificacion
print("*************** EVALUATION METRICS *****************")
from sklearn import metrics 
print('CONFUSION MATRIX IN TESTING DATA')
print(metrics.confusion_matrix(y_test,y_predictions))

print(metrics.classification_report(y_test,y_predictions))




print("*************** PRUEBA *****************")
from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test=train_test_split(x,y,test_size=0.6)

from sklearn import tree

# # Argoritmos De maching Leraning
model2=tree.DecisionTreeClassifier()

print("*************** TRAINING STEP *****************")

model2.fit(x_train,y_train)

print("*************** TEST PREDICTIONS *****************")

y_predictions=model2.predict(x_test)
# # # print(y_predictions[:30])
# # # print(y_test[:30])

# # # Metodo de clasificacion
print("*************** EVALUATION METRICS *****************")
from sklearn import metrics 
print('CONFUSION MATRIX IN TESTING DATA')
print(metrics.confusion_matrix(y_test,y_predictions))

print(metrics.classification_report(y_test,y_predictions))


