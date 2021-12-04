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


classes = ['malignant','benign']

clf = svm.SVC(kernel="linear", C=1)

clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)

acc = metrics.accuracy_score(y_test,y_pred)

for z in range(len(y_pred)):
    print("Predicted:",classes[y_pred[z]] ,"Data ", x_test[z],"Actual ",classes[y_test[z]])

