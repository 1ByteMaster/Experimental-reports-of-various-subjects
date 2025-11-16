import random
import math
import time

def text_to_numbers(text):
    """将字符串转为整数列表"""
    return [ord(ch) for ch in text]


def numbers_to_text(numbers):
    """将整数列表转为字符串"""
    return ''.join([chr(num) for num in numbers])


def get_GCD(a, b):
    """欧几里得算法求最大公约数"""
    while b != 0:
        a, b = b, a % b
    return a


def extended_gcd(a, b):
    """扩展欧几里得算法，返回 gcd, x, y 使得 ax + by = gcd"""
    if b == 0:
        return a, 1, 0
    else:
        g, x1, y1 = extended_gcd(b, a % b)
        x = y1
        y = x1 - (a // b) * y1
        return g, x, y


def mod_inverse(e, phi_n):
    """求 e 在模 phi_n 下的逆元 d"""
    g, x, _ = extended_gcd(e, phi_n)
    if g != 1:
        raise Exception("无模逆元")
    return x % phi_n


def is_probable_prime(n, k=5):
    """Miller-Rabin 素性检测"""
    if n < 2:
        return False
    for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]:
        if n % p == 0:
            return n == p
    # 分解 n - 1 = 2^s * t
    s, t = 0, n - 1
    while t % 2 == 0:
        s += 1
        t //= 2
    for _ in range(k):
        a = random.randrange(2, n - 1)
        x = pow(a, t, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(s - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    return True


def get_prime(bits=64):
    """生成指定位数的素数"""
    while True:
        num = random.getrandbits(bits)
        num |= 1  # 确保为奇数
        if is_probable_prime(num):
            return num


def choose_e(phi_n):
    """选取 e，使其与 phi_n 互质"""
    common_e_values = [3, 17, 65537]
    for e in common_e_values:
        if get_GCD(e, phi_n) == 1:
            return e
    # 若不符合，则随机生成
    while True:
        e = random.randrange(2, phi_n)
        if get_GCD(e, phi_n) == 1:
            return e


def calculate_d(e, phi_n):
    """计算 d"""
    return mod_inverse(e, phi_n)


def generate_key(bits=64):
    """生成 RSA 公钥与私钥"""
    p = get_prime(bits)
    q = get_prime(bits)
    while q == p:
        q = get_prime(bits)
    n = p * q
    phi_n = (p - 1) * (q - 1)
    e = choose_e(phi_n)
    d = calculate_d(e, phi_n)
    return (n, e), (n, d)


def encrypt(message, public_key):
    """加密"""
    n, e = public_key
    nums = text_to_numbers(message)
    return [pow(m, e, n) for m in nums]


def decrypt(ciphertext, private_key):
    """解密"""
    n, d = private_key
    m_list = [pow(c, d, n) for c in ciphertext]
    return numbers_to_text(m_list)

def main():
    public_key, private_key = generate_key()
    n, e = public_key
    _, d = private_key

    print("Public Key (n, e):", public_key)
    print("Private Key (n, d):", private_key)

    # 保存密钥
    with open("pkey.txt", "w") as f:
        f.write(f"{n},{e}")
    with open("skey.txt", "w") as f:
        f.write(f"{n},{d}")

    message = input("输入需要加密的明文: ")
    print("原始消息:", message)

    ciphertext = encrypt(message, public_key)
    print("加密消息:", ciphertext)
    with open("cipher.txt", "w") as f:
        f.write(','.join(map(str, ciphertext)))

    decrypted = decrypt(ciphertext, private_key)
    print("解密消息:", decrypted)


if __name__ == "__main__":
    main()
