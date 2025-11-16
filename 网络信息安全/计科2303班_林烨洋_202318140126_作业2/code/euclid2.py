# euclid2.py
import random
def gcd_iterative(a, b):
    while b != 0:
        a, b = b, a % b
    return a

if __name__ == "__main__":
    n = 512  
    a = random.getrandbits(n)
    b = random.getrandbits(n)
    result = gcd_iterative(a, b)
    print("gcd({},\n {}) =\n {}".format(hex(a),hex(b),hex(result)))
