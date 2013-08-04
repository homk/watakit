"""
Solve, optimize univariate (i.e. scalar-variable) functions.
"""
import scipy.optimize 
#Other tasks
# fixed point  scipy.optimize.fixed_point
# 

# --------------
# Solve f(x) = y 
# --------------
#According to scipy.org there are four methods to solve f(x) = 0.
#i.e. scipy.optimize. brentq, brenth, ridder, bisect
#Usually brentq is the best choice.

# Wrapper for scipy.optimize
def find_value(f, y=0, x_min=None, x_max=None, boundary_type='()', args = (), maxiter=100, algorithm=None,
               function_type=None, xtol=10**-12, full_output=False, disp=True):
    """
    Lef f be a continuous univariate function.
    Find input x that returns y = f(x).
    Algorithm is automatically selected unless specified.
 
    This wrapper is designed to handle arbitray (not necessarily closed nor bounded) interval.
    
    Parameter
    ---------
    f : callable

    y : float

    x_min, x_max : float, 
        x_min <= x_max
        None represents infinite.

    boundary_type: '()', '(]', '[)', '[]'

    args : tuple of args
        f is called as f(x, *args)

    maxiter : maximum number of iteration in the algorithm

    algorithm : 'brentq', 'brenth', 'ridder', 'bisect' ...
        In the cuurent implementation, 'brentq' is called in any case.

    function_type : 'convex', 'concave', 'monotone', ....

    full_output: bool
        if True, (x_sol, r) is returned where r is scipy RootResults object

    disp : bool
        if True, raise RuntimeError if the algorithm didn't converge.
        
    """
    #setting parameter
    scale = 10.0  #initial guess of typical function scale.
    ratio = 10.0
    num_search = 100

    #Error check
    if x_min is not None and x_max is not None:
        if x_min >= x_max:
            raise ValueError('x_min < x_max must be hold.')
    
    #Utility function
    def r_update(x_r, x_max, n):
        if x_max is None:
            return (x_r, x_r + scale * ratio**n)
        else:
            return (x_r, 1.0/ratio * x_r + (1 - 1.0/ratio) * x_max)

    def l_update(x_min, x_l, n):
        if x_min is None:
            return (x_l - scale * ratio**n, x_l)
        else:
            return (1.0/ratio * x_r + (1 - 1.0/ratio) * x_min, x_l)

    #---------------------
    #Case1. closed interval
    if boundary_type == '[]':
        if x_min is None or x_max is None:
            raise ValueError('x_min and x_max must be given.')

        return _find_value_bdd(f, y=y, x_min=x_min, x_max=x_max, args=args, maxiter=maxiter, algorithm=algorithm,
                               function_type=function_type, xtol=xtol, full_output=full_output, disp=disp)

    #Case2. half open interval
    if boundary_type == '(]':
        if x_max is None:
            raise ValueError('x_max must be given.')

        x_r = x_max
        if x_min is None:
            x_l = x_max - scale
        else:
            x_l = x_min + xtol

        for n in xrange(num_search):            
            try:
                return _find_value_bdd(f, y=y, x_min=x_l, x_max=x_r, args=args, maxiter=maxiter, algorithm=algorithm,
                                       function_type=function_type, xtol=xtol, full_output=full_output, disp=disp)
            except ValueError:
                x_l, x_r = l_update(x_min, x_l, n)

            if n > num_search-2:
                raise ValueError('The value {} can not be found between {} and {}'.format(y, x_l, x_max))

    if boundary_type == '[)':
        if x_min is None:
            raise ValueError('x_min must be given.')

        x_l = x_min
        if x_max is None:
            x_r = x_min +  scale
        else:
            x_r = x_max - xtol
            
        for n in xrange(num_search):            
            try:
                return _find_value_bdd(f, y=y, x_min=x_l, x_max=x_r, args=args, maxiter=maxiter, algorithm=algorithm,
                                       function_type=function_type, xtol=xtol, full_output=full_output, disp=disp)

            except ValueError:
                x_l, x_r = r_update(x_l, x_max, n)

            if n > num_search - 2:
                raise ValueError('The value {} can not be found between {} and {}'.format(y, x_min, x_r))            
    
    #Case3. Open interval
    if boundary_type == '()':
        if x_min is None and x_max is None:
            x_r = scale
            x_l = -scale
        elif x_min is None:
            x_r = x_max - xtol
            x_l = x_max - scale
        elif x_max is None:
            x_r = x_min + scale
            x_l = x_min + xtol
        else:
            x_r = x_max - xtol
            x_l = x_min + xtol
        try:
            return _find_value_bdd(f, y=y, x_min=x_l, x_max=x_r, args=args, maxiter=maxiter, algorithm=algorithm,
                                       function_type=function_type, xtol=xtol, full_output=full_output, disp=disp)
        except ValueError:
            x_l0, x_r0 = l_update(x_min, x_l, 1)
            x_l1, x_r1 = r_update(x_r, x_max, 1)
        for n in xrange(num_search):
            print
            print x_min, x_max
            print x_l, x_r
            print x_l0, x_r0
            print x_l1, x_r1
            try:
                return _find_value_bdd(f, y=y, x_min=x_l1, x_max=x_r1, args=args, maxiter=maxiter, algorithm=algorithm,
                                       function_type=function_type, xtol=xtol, full_output=full_output, disp=disp)
            except ValueError:
                x_l1, x_r1 = r_update(x_r1, x_max, n+1)

            try:
                return _find_value_bdd(f, y=y, x_min=x_l0, x_max=x_r0, args=args, maxiter=maxiter, algorithm=algorithm,
                                       function_type=function_type, xtol=xtol, full_output=full_output, disp=disp)
            except ValueError:
                x_l0, x_r0 = r_update(x_min, x_r0, n+1)
            if n > num_search - 2:
                raise ValueError('The value {} can not be found between {} and {}'.format(y, x_l0, x_r1))   


def _find_value_bdd(f, y, x_min, x_max, args = (), maxiter=100, algorithm=None,
                    function_type=None, xtol=10**-12, full_output=False, disp=True):
    """
    If f has a good structure, we may choose other algorithm.
    (e.g. function_type = 'monotone' or 'convex' or 'nondec' etc...)

    For now we use brentq in any case.
    Even if f is monotone, brentq is faster than the bisection method.
    
    TODO: Choose best algorithm by function_type..
    """
    if y != 0:
        g = lambda x : f(x, *args) - y
        args = ()
    else:
        g = f
    
    return scipy.optimize.brentq(g, a=x_min, b=x_max, args=args, xtol=xtol, maxiter=maxiter, 
                                 full_output=full_output, disp=disp)



# My implementation of Bisection method
def find_value_monotone_bisection(f, y=0, x_min=-1.0, x_max=1.0, args=(), maxiter=100, 
                                  monotone_type='auto', xtol=1e-12, tol=1e-10, full_output=False, dist=True):
    """
    Let f be a monotone univariate function.
    Find input x that returns y = f(x) by 'bisection method'.
    (We recommend ''find_value'' because it is somehow faster.)
    
    Parameter
    ---------
    f : callable
        Objective function.

    y : float
        Target value:

    x_min, x_max: float 
        f takes values in a CLOSED interval [x_min, x_max]

    monotone_type : 'nondec', 'noninc', 'auto'

    tol : float
        Tolerence for f value.


    Comment
    -------

    """

    #Check monotonicity type (between x_min and x_max)
    if monotone_type == 'auto':
        if f(x_max, *args) - f(x_min, *args) > 0:
            monotone_type ='nondec'
        elif f(x_max, *args) - f(x_min, *args) < 0:
            monotone_type = 'noninc'
        else:
            if f(x_min, *args) == y:
                return (x_min + x_max) / 2.0
            else:
                raise ValueError('Function f is constant not equal to {}.'.format(y))

    #main
    if monotone_type == 'nondec':
        x_l = x_min
        x_r = x_max
        x_m = (x_l + x_r) / 2.0
        fm = f(x_m, *args)
        i = 0
        while ( abs(fm - y) > tol or abs(x_r - x_l) > xtol ) and i < maxiter:
            i += 1
            if fm < y:
                x_l = x_m
            elif fm > y:
                x_r = x_m

            x_m = (x_l + x_r) / 2.0
            fm = f(x_m, *args)
            
    elif monotone_type == 'noninc':
        x_l = x_min
        x_r = x_max
        x_m = (x_l + x_r) / 2.0
        fm = f(x_m, *args)
        i = 0
        while ( abs(fm - y) > tol or abs(x_r - x_l) > xtol ) and i < maxiter:
            i += 1
            if fm > y:
                x_l = x_m
            elif fm < y:
                x_r = x_m
            x_m = (x_l + x_r) / 2.0
            fm = f(x_m, *args)

    #Results
    if ( abs(fm - y) > tol or abs(x_r - x_l) > xtol ):
        converged = False
    else:
        converged = True

    if dist == True and converged == False:
        raise RuntimeError('Failed to converge after {} iterations'.format(i))
    
    if full_output is True:
        r = scipy.optimize.zeros.RootResults(x_m, i, i+1, 0 if converged else 1)
        return (x_m, r)
    else:
        return x_m

    
    
    
