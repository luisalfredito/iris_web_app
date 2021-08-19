#Importsmos las librerias necesarias
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC 
import pickle

#Carga de dataos al Dataset
iris = datasets.load_iris()

X = iris.data
y = iris.target

#Separacion de datos para Training y Test
x_train, x_test, y_train, y_test = train_test_split(X, y)

lin_reg = LinearRegression()
log_reg = LogisticRegression()
svc_m = SVC()

#Entrenar modelos

lin_regr = lin_reg.fit(x_train, y_train)
log_regr = log_reg.fit(x_train, y_train)
svc_mo = svc_m.fit(x_train, y_train)

#Generacion la persistencia del modelo
with open('lin_reg.pkl','wb')as li:
    pickle.dump(lin_reg, li)

with open('log_reg.pkl','wb')as lo:
    pickle.dump(log_reg, lo)

with open('svc_m.pkl','wb')as sv:
    pickle.dump(svc_m, sv)











