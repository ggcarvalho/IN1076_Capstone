from pipetools import pipe
import sys

def square(x):
    return x**2

def cube(x):
    return x**3

def sqrt(x):
    return x**(1/2)

def identity(x):
    return x

functions = {"square"  : square,
             "cube"    : cube,
             "sqrt"    : sqrt,
             "identity": identity}



def pipe_functions(func_names):
    funcs = func_names.split()
    l = len(funcs)
    if l == 1:
        f1 = functions.get(funcs[0]) if not (functions.get(funcs[0]) == None ) else functions.get("identity")
        f2 = functions.get("identity")
        f3 = functions.get("identity")
    elif l == 2:
        f1 =  functions.get(funcs[0]) if not (functions.get(funcs[0]) == None ) else functions.get("identity")
        f2 =  functions.get(funcs[1]) if not (functions.get(funcs[1]) == None ) else functions.get("identity")
        f3 = functions.get("identity")
    elif l == 3:
        f1 =  functions.get(funcs[0]) if not (functions.get(funcs[0]) == None ) else functions.get("identity")
        f2 = functions.get(funcs[1]) if not (functions.get(funcs[1]) == None ) else functions.get("identity")
        f3 = functions.get(funcs[2]) if not (functions.get(funcs[2]) == None ) else functions.get("identity")
    else:
        raise Exception("This program supports only 3 stages of processing!")

    f = pipe | f1 | f2 | f3
    return f

f = pipe_functions("square square asdjasoidj")
print(2 > f)
