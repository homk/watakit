import numpy as np
from sklearn import svm as sklearn_svm

"""
TODO: multiple kernel SVM, ...
"""


class SVM(object):
    """
    Simple binary SVM.

    This class is created by wrapping sklern.svm.SVC.
    Differences from the original implementation are the following.
    1. Can handle any labels.
        e.g. Y = ['large', 'small', 'small', ...]

    2. New methods, properties are added:
        support_signs_
        support_alpha_
        support_penalty_  
        b
        w2
        cost
  

    TODO: Nystorm Method, Hyperparameter optimization,....
    """
    def __init__(self, C=1.0, cache_size=1000, class_weight=None, coef0=0.0, 
                 degree=3, gamma=0.0, kernel='rbf', max_iter=-1, probability=False, 
                 shrinking=True, tol=0.001, verbose=False):
        """
        Parameters
        ----------
        C : float
            Regularization parameter.
            If C is big, the boundary is soft.
            If C is small, the boundary is hard. (Good for noisy data.)
        """
        
        #set parameters to a dictionary
        kwargs = {
                 'C' : C,
                 'cache_size' : cache_size,
                 'class_weight' : class_weight,
                 'coef0' : coef0,
                 'degree' : degree,
                 'gamma' : gamma,
                 'kernel' : kernel,
                 'max_iter' : max_iter,
                 'probability' : probability,
                 'shrinking' : shrinking,
                 'tol' : tol,
                 'verbose' : verbose,
                  }

        for key, val in kwargs.items():
            setattr(self, key, val)

        #construct SVC
        self._clf = sklearn_svm.SVC(**kwargs)


    def fit(self, X, Y):
        """
        X : shape = [num_sample x dim_input]
        Y : list of labels
            Y[i] is not necessarily numbers!
        """
        self.X = np.array(X)
        self.Y = Y
        self.num_sample, self.dim_input = self.X.shape        

        #label error check
        self.label_Y = sorted(list(set(Y)))
        if len(self.label_Y) > 2:
            raise ValueError('SVM is binary classifier. \
                              Label sequence Y must not have more than two labels.')
        elif len(self.label_Y) < 1:
            print('Caution: label sequence Y does not contain two labels..')

        Yi = [0 if y == self.label_Y[0] else 1 for y in Y]
        self._clf.fit(X,Yi)

        #Delete cache
        for attr in ['_support_signs_', '_support_alpha_', '_w2', '_cost', '_support_penalty_']:
            if hasattr(self, attr):
                delattr(self, attr)


    def decision_function(self, X):
        """
        We use the following convension:
          sign +1 == self.label_Y[0]
          sign -1 == self.label_Y[1]
        """
        return (-1) * self._clf.decision_function(X)[:,0]


    def predict(self, X):
        """Predict label by input.
        Parameters
        ----------
        X : [num_inputs, dim_input]

        Returns
        Y : [num_inputs]
        """
        return np.array(map(lambda i: self.label_Y[i], self._clf.predict(X)))


    @property
    def support_(self):
        """
        ndarray of indices (0 <= i < num_sample), shape = [num_support_vector]
        
        Indices of support vectors.
        class 0 (+1) comes ahead followed by class 1 (-1).
        See also support_vectors_, dual_coef_.
        """
        return sorted(self._clf.support_)


    @property
    def n_support_(self):
        return len(self.support_)


    @property
    def support_vectors_(self):
        """
        ndarray, shape = [num_support_vector x dim_input]
        
        array of support vectors.
        Column index is orderd in accordance with support_ attribute.       
        """
        return self._clf.support_vectors_


    @property
    def dual_coef_(self):
        """
        ndarray, shape = [num_support_vectors]
        
        dual_coef = alpha * y  (0 < alpha < 1, y= +1 or -1 )
        np.sum(dual_coef) == 1.

        Column index is orderd in accordance with support_ attribute.       
        """
        return self._clf.dual_coef_[0]


    #New Properties
    @property
    def support_signs_(self):
        """
        ndarray, shape = [num_support_vectors]
        
        sign +1 == self.label_Y[0]
        sign -1 == self.label_Y[1]
        Column index is orderd in accordance with support_ attribute.       
        """
        try:
            self._support_signs_ 
        except AttributeError:
            self._support_signs_ = np.array([+1 if self.Y[i] == self.label_Y[0] else -1 for i in  self.support_])
        return self._support_signs_


    @property
    def support_alpha_(self):
        """
        ndarray, shape = [num_support_vectors]
        
        Column index is orderd in accordance with support_ attribute.       
        """
        try:
            self._support_alpha_
        except AttributeError:
            self._support_alpha_ = self.dual_coef_ * self.support_signs_ 
        return self._support_alpha_


    @property
    def b(self):
        """
        Constant (bias) term of the decision function.
        """
        return self._clf.intercept_[0]


    @property
    def w2(self):
        """
        Squared norm of vector w that defines the separating hyperplane.
        Big w2 means small margin. (in RKHS metric.)
        
        margin = 1.0 / ||w||
        """
        try:
            self._w2
        except AttributeError:
            #from IPython import embed; embed()
            self._w2 = np.sum([ self.decision_function(self._clf.support_vectors_) * self.dual_coef_ ] )
        return self._w2


    @property
    def cost(self):
        """
        The value of resulting cost function.

        cost = 0.5 * w2 + C * sum(penalty)
             = sum(alpha) - 0.5 * w2
        """
        try:
            self._cost
        except AttributeError:
            self._cost = np.sum(self.support_alpha_) - 0.5 * self.w2
        return self._cost


    @property
    def support_penalty_(self):
       """
       Penalty terms of support vectors. 
       penalty_i = max(0, 1 - y_i * decision_function(x_i))
       """
       try:
           self._support_penalty_
       except AttributeError:
           self._support_penalty_ = 1 - self.support_signs_ * self.decision_function(self.support_vectors_)  #must be non-negative for support vectors.
       return self._support_penalty_



class NuSVM(SVM):
    """
    NuSVM 

    NuSVM is a binary classifier.


    """
    def __init__(self, nu=0.5, cache_size=1000, coef0=0.0,   #no C, class_weights
                 degree=3, gamma=0.0, kernel='rbf', max_iter=-1, probability=False, 
                 shrinking=True, tol=0.001, verbose=False):
        """
        Parameters
        ----------
        nu : float
            fraction of training errors <= nu <= fraction of support vectors
            
            Small nu yields small number of support vectors at the cost of small margin.
            Large nu yields large number of support vectors and large margin.


        See 'New support vector algorithms' by Scholkopf et al.
        """
        
        #set parameters to a dictionary
        kwargs = {
                 'nu' : nu,
                 'cache_size' : cache_size,
                 #'class_weight' : class_weight,
                 'coef0' : coef0,
                 'degree' : degree,
                 'gamma' : gamma,
                 'kernel' : kernel,
                 'max_iter' : max_iter,
                 'probability' : probability,
                 'shrinking' : shrinking,
                 'tol' : tol,
                 'verbose' : verbose,
                  }

        for key, val in kwargs.items():
            setattr(self, key, val)

        #construct SVC
        self._clf = sklearn_svm.NuSVC(**kwargs)
    


# SVC
# ---
class SVC_base(object):
    """
    Base Class for Support Vector Classifier. (Multiclass SVM)
    
    1. 
    """
    def __init__(self, C=1.0, nu=0.5, kernel='rbf', cache_size=1000, class_weight=None, 
                 coef0=0.0, probability=False, max_iter=-1,
                 multiclass_method = '1v1', svm_type = 'SVM'):
        """
        Parameters
        ----------

        multiclass_method : '1v1', '1vR', 'ECOC'
        svm_type: 'SVM', 'NuSVM'
        """
        #set parameters to a dictionary
        kwargs = {
                 'C' : C,
                 'nu' : nu,
                 'cache_size' : cache_size,
                 'class_weight' : class_weight,
                 'coef0' : coef0,
                 #'degree' : degree,
                 #'gamma' : gamma,
                 'kernel' : kernel,
                 'max_iter' : max_iter,
                 'probability' : probability,
                 #'shrinking' : shrinking,
                 #'tol' : tol,
                 #'verbose' : verbose,
                 'multiclass_method': multiclass_method,
                 'svm_type' : svm_type,
                  }

        for key, val in kwargs.items():
            setattr(self, key, val)


    def decision_function(self, X):
        """
        Decision functions of SVMs.

        Parameters
        ----------
        X : shape = (num_inputs, dim_input)

        Returns
        ------
        decision_function_values : shape = (num_inputs, num_svms)        
        """
        return np.array([svm.decision_function(X) for svm in self.svm_list]).T


    def _clear_cache(self):
        cache_attr = ['_support_', '_support_alphas_']
        for attr in cache_attr:
            if hasattr(self, attr):
                delattr(self, attr)


    #Properties of SVC
    #    'list' prefix indicates a list of properties w.r.t svm_list.
    @property
    def support_list(self):
        return [svm.support_ for svm in self.svm_list]


    @property
    def support_(self):
        """
        A list of support vectors.
        In the multiclass setting, there are multiple svms and each svm has a set of support vectors.
        This property returns the UNION of the sets of support vectors.

        Returns
        ------
        Ind : list of indices of support vectors.
        """
        try: 
            self._support_
        except AttributeError:
            tmp = []
            for svm in self.svm_list:
                tmp.extend(svm.support_)
            self._support_ = sorted(list(set(tmp)))

        return self._support_


    @property
    def n_support_list(self):
        return [svm.n_support_ for svm in self.svm_list]


    @property
    def n_support_(self):
        return len(self.support_)


    @property
    def support_vectors_(self):
        """
        Returns
        ------
        Ind : list of support vectors.
        """
        return np.array([self.X[i] for i in self.support_])


    @property
    def support_alpha_list(self):
        """
        Returns
        -------
        Alpha : ndarray, shape=(num_svm, n_support_)
        """
        try: 
            self._support_alphas_
        except AttributeError:
            self._support_alpha_list = np.empty((len(self.svm_list), self.n_support_))
            for i, svm in enumerate(self.svm_list):
                for k, j in enumerate(svm.support_):
                    self._support_alpha_list[i,self.support_.index(j)] = svm.support_alpha_[k]
        
        return self._support_alpha_list




        

class SVC_1vR(SVC_base):
    """
    Multiclass SVM with 1 vs Rest strategy.
    
    1. 
    """
    def fit(self, X, Y):
        """
        
        """
        self.X = np.array(X)
        self.Y = Y
        self.num_sample, self.dim_input = self.X.shape

        #Check number of classes
        self.label_Y = sorted(list(set(Y)))
        if len(self.label_Y) < 1:
            print('Caution: label sequence Y does not contain two labels..')

        #Create a list of svms.
        self.svm_list = []
        for i, y0 in enumerate(self.label_Y):
            Yi = [0 if y == y0 else 1 for y in self.Y]
            Xi = self.X
            if self.svm_type == 'SVM':
                svm = SVM(self.C)
            elif self.svm_type == 'NuSVM':
                svm = NuSVM(self.nu)

            svm.fit(X, Yi)
            self.svm_list.append(svm)

        #clear cache
        self._clear_cache()


    def predict(self, X):
        """
        Decision function by 1 vs Rest strategy.
        In this strategy, the class label with the maximum svm score is used.
        """
        tmp_predict = np.argmax(self.decision_function(X), 1)
        return [self.label_Y[i] for i in tmp_predict]



        
        


        
        
