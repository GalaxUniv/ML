import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing
from matplotlib import style
import matplotlib.pyplot as pyplot
#main

data =  pd.read_csv("Tensor K-Nearest Neighbors Algorithm/car.data")
print(data.head())
'''
  buying  maint door persons lug_boot safety  class
0  vhigh  vhigh    2       2    small    low  unacc
1  vhigh  vhigh    2       2    small    med  unacc
2  vhigh  vhigh    2       2    small   high  unacc
3  vhigh  vhigh    2       2      med    low  unacc
4  vhigh  vhigh    2       2      med    med  unacc
'''
le = preprocessing.LabelEncoder()
buying = le.fit_transform(list(data["buying"]))
maint = le.fit_transform(list(data["maint"]))
door = le.fit_transform(list(data["door"]))
persons = le.fit_transform(list(data["persons"]))
lug_boot = le.fit_transform(list(data["lug_boot"]))
safety = le.fit_transform(list(data["safety"]))
class_=data["class"].map({"unacc": 0, "acc": 1,"good": 2,"vgood": 3})
X = list(zip(buying,maint,door,persons,lug_boot,safety))
Y = list(class_)

x_train , x_test ,y_train , y_test  = sklearn.model_selection.train_test_split(X,Y, test_size = 0.15)

#print(x_train , y_test)

model = KNeighborsClassifier (n_neighbors=8)

model.fit(x_train , y_train)
acc = model.score(x_test, y_test)

predicted = model.predict(x_test)
names = ["unacc","acc","good","vgood"]

for z in range(len(predicted)):
  #print("Predicted:",predicted[z] ,"Data: ",x_test[z], "Actual ",y_test[z])
  print("Predicted:",names[predicted[z]] ,"Data: ",x_test[z], "Actual ",names[y_test[z]])
  n=model.kneighbors([x_test[z]],9 , True)
  print(n)

print(acc)

#find a way to make a chart
#style.use("ggplot")
#pyplot.scatter(data[p],data["G3"])
#pyplot.xlabel("First grade")
#pyplot.ylabel("Final Grade")
#print(y_test)
#print(predicitons)
#print("Mean squared error: %.2f" % mean_squared_error(y_test,predicitons))
#print("Coefficient of determination: %.2f" % r2_score(y_test,predicitons))
#
#print(linear.coef_[0] , linear.intercept_)
#a = linear.coef_[0]
#b = linear.intercept_
#
#pyplot.scatter(x_test, y_test, color="black")
#
##pyplot.xticks(())
##pyplot.yticks(())
#
#
#
#ax = pyplot.subplot()
#t = np.arange(0,21,1)   
#s = ((a*t) + b) # a * range x + b
#line, = pyplot.plot(t, a*t + b,color="blue", lw=3)
#
#
#pyplot.show()





