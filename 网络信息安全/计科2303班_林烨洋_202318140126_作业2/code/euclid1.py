import random
def gcd_recursive(a, b):
    if b == 0:
        return a
    else:
        return gcd_recursive(b, a % b)
if __name__ == "__main__":
    n = 512
    a = random.getrandbits(n)
    b = random.getrandbits(n)
    result = gcd_recursive(a, b)
    print("gcd({},\n {}) =\n {}".format(hex(a),hex(b),hex(result)))
