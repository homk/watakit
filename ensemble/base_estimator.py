"""
Define base estimators used in AdaBoostClassifier.
"""

import abc

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import SGDClassifier



class BaseClassifier(BaseEstimator, ClassifierMixin):
    """
    Abstract base class for classification-base_estimators.
    All base_estimators (for classification) must be derived from this class.

    All base_estimators must have
        - fit method
        - predict method
        - classes_ property
    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def fit(self, X, y, sample_weight):
        pass

    @abc.abstractmethod
    def predict(self, X):
        pass

    
    #@abc.abstractproperty
    #def classes_(self):
    #pass




class OneVariableClassifier(BaseClassifier):
    """
    Each weak-learner uses only one component of input vectors.

    Train base_estimator by SGD Classifier.
    """
    def __init__(self, loss='hinge', penalty='l2', alpha=0.0001, l1_ratio=0.15, fit_intercept=True, n_iter=5, shuffle=False, 
                 verbose=0, epsilon=0.1, n_jobs=1, random_state=None, learning_rate='optimal', eta0=0.0, power_t=0.5, class_weight=None, 
                 warm_start=False, rho=None, seed=None):
        """
        Parameters
        ----------
        Most of the parameters are for SGD classifier
        """


    def fit(self, X, y, sample_weight):
        self.clf_list = []
        self.clf_score_list = []
        for Xi in X.T:
            xi = Xi.reshape((-1,1))
            #create classifier
            clf = SGDClassifier(n_iter=100, alpha=0.1, verbose=0) #TODO
            clf.fit(xi, y, sample_weight=sample_weight)
            self.clf_list.append(clf)
            w_score = np.sum([0 if y0!=y1 else w  for (y0,y1,w) in zip(clf.predict(xi),y,sample_weight)]) #weighted score
            self.clf_score_list.append(w_score)
            
        self.classes_ = np.array(sorted(list(set(y))))
        print 'fit!'


    def predict(self, X):
        self.max_score = max(self.clf_score_list)
        self.max_ind = self.clf_score_list.index(self.max_score)
        return self.clf_list[self.max_ind].predict(X[:,self.max_ind:self.max_ind+1]) 
