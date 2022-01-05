from os import access
import pandas as pd
import numpy as np
import sklearn 
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv("datasets\Salary.csv", sep=",") #reads data seperator = ;


data = data[["YearsExperience","Salary"]]
print(data.head()) #printing 5 rows

best = 0
X= np.array(data.drop(["Salary"],1))
y = np.array(data["Salary"])
x_train , x_test ,y_train , y_test  = sklearn.model_selection.train_test_split(X,y, test_size = 0.1)

# training model
for i in range(100):
    x_train , x_test ,y_train , y_test  = sklearn.model_selection.train_test_split(X,y, test_size = 0.1)

    #best fit line  ==>  y = ax + b

    linear = linear_model.LinearRegression()

    linear.fit(x_train,y_train)
    acc = linear.score(x_test,y_test)
    print(acc)

    if acc > best:
        best = acc
        with open("salary.pickle", "wb") as f:        #saves and loads 
            pickle.dump(linear,f)                           #already trained 
                                                            #linear module

pickle_in = open("salary.pickle" ,"rb")       #
linear = pickle.load(pickle_in) 

print("Co : \n", linear.coef_) # = a in each direction
print("Intercept: \n",  linear.intercept_) #  = b

predicitons = linear.predict(x_test)
for x in range(len(predicitons)):
    print(predicitons[x], x_test[x], y_test[x])

acc = linear.score(x_test,y_test)
print(acc)

p="YearsExperience"
style.use("ggplot")
pyplot.scatter(data[p],data["Salary"])
pyplot.xlabel("First grade")
pyplot.ylabel("Final Grade")
print(y_test)
print(predicitons)
print("Mean squared error: %.2f" % mean_squared_error(y_test,predicitons))
print("Coefficient of determination: %.2f" % r2_score(y_test,predicitons))

print(linear.coef_[0] , linear.intercept_)
a = linear.coef_[0]
b = linear.intercept_

#a = 8695.315962920855
#b = 29033.406725344925

pyplot.scatter(x_test, y_test, color="black")

#pyplot.xticks(())
#pyplot.yticks(())



ax = pyplot.subplot()
t = np.arange(0,21,1)   
s = ((a*t) + b) # a * range x + b
line, = pyplot.plot(t, a*t + b,color="blue", lw=1)


pyplot.show()

input()