#import ..classes
from sklearn.datasets import load_iris



def test_svm():
iris = load_iris()
X, Y = zip(*[(x,y) for x,y in zip(iris.data,iris.target) if y in [0, 1]])  #Select 0, 1 data.
svm = SVM(C=1.0, kernel='rbf')
svm.fit(X, Y)
    
    svm.assert_almost_equal(svm.cost, 2.4034163345438264, sv4)
    
    

def test_svc():
iris = load_iris()
X = iris.data
Y = iris.target
svc = SVC()
svc.fit(X, Y)
    
    

