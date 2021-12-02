from os import access
import pandas as pd
import numpy as np
import sklearn 
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

data = pd.read_csv("student-mat.csv", sep=";") #reads data seperator = ;


data = data[["G1","G3"]]
predict = "G3"
X= np.array(data.drop([predict],1))
y = np.array(data[predict])
x_train , x_test ,y_train , y_test  = sklearn.model_selection.train_test_split(X,y, test_size = 0.1)

'''# training model
best= 0
for i in range(10000):
    x_train , x_test ,y_train , y_test  = sklearn.model_selection.train_test_split(X,y, test_size = 0.1)

    #best fit line  ==>  y = ax + b

    linear = linear_model.LinearRegression()

    linear.fit(x_train,y_train)
    acc = linear.score(x_test,y_test)
    print(acc)

    if acc > best:
        best = acc
        with open("FirstGrade.pickle", "wb") as f:        #saves and loads 
            pickle.dump(linear,f)                           #already trained 
                                                            #linear module
'''

pickle_in = open("FirstGrade.pickle" ,"rb")       #
linear = pickle.load(pickle_in) 

print("Co : \n", linear.coef_) # = a in each direction
print("Intercept: \n",  linear.intercept_) #  = b

predicitons = linear.predict(x_test)
for x in range(len(predicitons)):
    print(predicitons[x], x_test[x], y_test[x])

p="G1"
style.use("ggplot")
pyplot.scatter(data[p],data["G3"])
pyplot.xlabel("First grade")
pyplot.ylabel("Final Grade")

print(linear.coef_[0] , linear.intercept_)
a = linear.coef_[0]
b = linear.intercept_

ax = pyplot.subplot()
t = np.arange(0,21,1)   
s = ((a*t) + b) # a * range x + b
line, = pyplot.plot(t, a*t + b, lw=2)


pyplot.show()
