import inspect

def print_signature(a_callable):

    sign=inspect.signature(a_callable)
    str_sign=str(sign)[1:-1]
    liste=str_sign.split(", ")
    for elem in liste:
        print(elem)