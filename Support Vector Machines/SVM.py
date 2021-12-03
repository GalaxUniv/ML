import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn import metrics
from sklearn.metrics.pairwise import linear_kernel
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as pyplot

cancer = datasets.load_breast_cancer()


# print(cancer.feature_names)
# print(cancer.target_names)


X = cancer.data
Y = cancer.target

x_train, x_test,y_train,y_test = sklearn.model_selection.train_test_split(X,Y, test_size = 0.2)

# print(x_train)
# print(y_train)

classes = ['malignant' 'benign']

clf = svm.SVC(kernel="linear", C=1)

clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)

acc = metrics.accuracy_score(y_test,y_pred)

#print(y_pred) # prediction
#print(y_test) # test anmsw data
print(x_test)

#find a way to print a chart