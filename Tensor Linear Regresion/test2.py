from os import access
import pandas as pd
import numpy as np
import sklearn 
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

data = pd.read_csv("Tensor Linear Regresion/student-mat.csv", sep=";") #reads data seperator = ;


data = data[["G1","G2","G3","studytime","failures","absences"]]
print(data.head()) #printing 5 rows

predict = "G3"
best = 0
X= np.array(data.drop([predict],1))
y = np.array(data[predict])
x_train , x_test ,y_train , y_test  = sklearn.model_selection.train_test_split(X,y, test_size = 0.1)

'''# training model
for i in range(10000):
    x_train , x_test ,y_train , y_test  = sklearn.model_selection.train_test_split(X,y, test_size = 0.1)

    #best fit line  ==>  y = ax + b

    linear = linear_model.LinearRegression()

    linear.fit(x_train,y_train)
    acc = linear.score(x_test,y_test)
    print(acc)

    if acc > best:
        best = acc
        with open("Tensor Linear Regresion/studentmodel.pickle", "wb") as f:        #saves and loads 
            pickle.dump(linear,f)                           #already trained 
                                                            #linear module
'''

pickle_in = open("Tensor Linear Regresion/studentmodel.pickle" ,"rb")       #
linear = pickle.load(pickle_in) 

print("Co : \n", linear.coef_) # = a in each direction
print("Intercept: \n",  linear.intercept_) #  = b

predicitons = linear.predict(x_test)
for x in range(len(predicitons)):
    print(predicitons[x], x_test[x], y_test[x])

acc = linear.score(x_test,y_test)
print(acc)

p="G2"
style.use("ggplot")
pyplot.scatter(data[p],data["G3"])
pyplot.xlabel(p)
pyplot.ylabel("Final Grade")
pyplot.show()
