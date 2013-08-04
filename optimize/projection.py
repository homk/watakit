"""
Projection onto various sets.
Basically, projection is performed in Euclidian norm.
"""

from scipy import weave


def project_hyperplane(x, a, b=0):
    """
    Project x on to a hyperplane defined as {x| ax + b = 0}

    Parameter
    ---------
    x : n-dim vector 
    a : n-dim vector
    b : scalar
    """
    t = float(b - np.dot(x,a)) / np.dot(a,a)
    return x + a * t
    

def project_positive_orthant(x, write=False):
    """
    Project x onto the positive orthant.
    i.e. {x| xi >= 0}

    Parameter
    ---------
    x : ndarray
    write : if True, the result is over-written in x memory.
            write=True is roughly 5 times farster.

    Executin time
    -------------
    dim_x = 10000 and run 10000 times.
    Have a try cProfile.run('for n in range(10000): code_here')...
        1. project_positive_orthant(x, write=True)  : 0.163 sec 
        2. project_positive_orthant(x, write=False) : 0.935 sec 
        3. [xi if xi > 0 else 0 for xi in x]        : 80.5 sec
    """
    dim_x = int(x.size)
    if write:
        code = """
            for(int i=0; i < dim_x; i++){{
                if(X1(i) < 0)
                    X1(i) = 0;
            }}
            """
        weave.inline(code, arg_names = ['x', 'dim_x'])
    else:  #Default
        z = np.zeros(x.shape)   #Allocate memory to write result
        code = """
            for(int i=0; i < dim_x; i++){{
                if(X1(i) > 0)
                    Z1(i) = X1(i);
            }} 
            """
        weave.inline(code, arg_names=['x', 'z', 'dim_x'])
        return z


def project_simplex(x, r=1):
    """
    Project x onto a simplex set defined by {x | xi >= 0, sum(x) = r}
    """
