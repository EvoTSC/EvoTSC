import random
import sys
from inspect import isclass

__type__ = object


def genFull(pset, min_, max_, type_=None):
    def condition(height, depth):
        return depth == height
    return generate(pset, min_, max_, condition, type_)

def genGrow(pset, min_, max_, type_=None):
    def condition(height, depth):
        return depth == height or depth >= min_
    return generate(pset, min_, max_, condition, type_)

def genHalfAndHalf(pset, min_, max_, type_=None):
    method = random.choice((genGrow, genFull))
    return method(pset, min_, max_, type_)

def generate(pset, min_, max_, condition, type_=__type__):
    if type_ is None:
        type_ = pset.ret
    expr = []
    height = random.randint(min_, max_)
    stack = [(0, type_)]
    while len(stack) != 0:
        if (len(expr)>60):
            expr = []
            type_ = pset.ret
            stack = [(0, type_)]
            height = random.randint(min_, max_)
        depth, type_ = stack.pop()
        if condition(height, depth):
            try:
                term = random.choice(pset.terminals[type_])
                if isclass(term):
                    term = term()
                expr.append(term)
            except:
                try:
                    depth -= 1
                    prim = random.choice(pset.primitives[type_])
                    expr.append(prim)
                    for arg in reversed(prim.args):
                        stack.append((depth, arg))

                except IndexError:
                    _, _, traceback = sys.exc_info()
                    raise IndexError("The gp.generate function tried to add " \
                                      "a primitive of type '%s', but there is " \
                                      "none available." % (type_,), traceback)
        else:
            try:
                prim = random.choice(pset.primitives[type_])
                expr.append(prim)
                for arg in reversed(prim.args):
                    stack.append((depth + 1, arg))
            except:
                try:
                    term = random.choice(pset.terminals[type_])

                except IndexError:
                    _, _, traceback = sys.exc_info()
                    raise IndexError("The gp.generate function tried to add " \
                                      "a terminal of type '%s', but there is " \
                                      "none available." % (type_,), traceback)
                if isclass(term):
                    term = term()
                expr.append(term)
    return expr


def genHalfAndHalfMD(pset, min_, max_, type_=None):
    expr=genHalfAndHalf(pset, min_, max_, type_=None)
    while len(expr)>60:
        expr=genHalfAndHalf(pset, min_, max_, type_=None)
    return expr
