# extEuclid.py
# 扩展欧几里得算法求模逆元
def getInverse(m, b):
    print("{:>4} {:>6} {:>6} {:>6} {:>6} {:>6} {:>6}".format('Q', 'A1', 'A2', 'A3', 'B1', 'B2', 'B3'))
    print("{:>4} {:>6} {:>6} {:>6} {:>6} {:>6} {:>6}".format('-', 1, 0, m, 0, 1, b))

    A1, A2, A3 = 1, 0, m
    B1, B2, B3 = 0, 1, b

    while True:
        if B3 == 0:
            print("\n无逆元（gcd != 1）")
            return None
        if B3 == 1:
            inverse = B2 % m
            print("\n逆元存在：b^(-1) ≡ {} (mod {})".format(inverse, m))
            return inverse

        Q = A3 // B3
        T1 = A1 - Q * B1
        T2 = A2 - Q * B2
        T3 = A3 - Q * B3

        print("{:>4} {:>6} {:>6} {:>6} {:>6} {:>6} {:>6}".format(Q, A1, A2, A3, B1, B2, B3))

        # 更新
        A1, A2, A3 = B1, B2, B3
        B1, B2, B3 = T1, T2, T3


if __name__ == "__main__":
    m = 1759
    b = 550
    print("计算 {} 在模 {} 下的逆元：\n".format(b, m))
    inverse = getInverse(m, b)

    if inverse is not None:
        # 验证结果正确性
        print("\n验证：({} * {}) % {} = {}".format(b, inverse, m, (b * inverse) % m))