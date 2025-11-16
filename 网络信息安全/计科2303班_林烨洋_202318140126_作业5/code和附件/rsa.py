import random
import math
import hashlib

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

# 新增：签名函数
def sign(message, private_key):
    """对消息的Hash值进行RSA签名"""
    n, d = private_key
    # 1. 计算消息的SHA-256哈希值（转为字节码后计算）
    message_bytes = message.encode("UTF-8")
    hash_obj = hashlib.sha256(message_bytes)
    hash_hex = hash_obj.hexdigest()  # 哈希值转为16进制字符串
    # 2. 将哈希字符串转为整数（用于RSA签名）
    hash_int = int(hash_hex, 16)
    # 3. 私钥加密（签名过程：s = h^d mod n）
    signature = pow(hash_int, d, n)
    return signature

# 新增：验证函数
def verify(message, public_key, signature):
    """验证RSA签名的有效性"""
    n, e = public_key
    # 1. 计算原始消息的SHA-256哈希值
    message_bytes = message.encode("UTF-8")
    hash_obj = hashlib.sha256(message_bytes)
    original_hash = hash_obj.hexdigest()
    original_hash_int = int(original_hash, 16)
    # 2. 公钥解密签名（验证过程：v = s^e mod n）
    verified_hash_int = pow(signature, e, n)
    # 3. 比对原始哈希值与验证后的哈希值
    return original_hash_int == verified_hash_int

# 修改主函数：实现实验要求的功能
def main():
    # 1. 生成1024bit密钥对（覆盖默认64bit）
    public_key, private_key = generate_key(bits=1024)
    print("Public Key (n, e):", public_key)
    print("Private Key (n, d):", private_key)
    
    # 2. 用户输入明文并签名验证
    message = input("输入需要签名的明文: ")
    print("原始消息:", message)
    # 生成签名
    sig = sign(message, private_key)
    print("RSA数字签名:", sig)
    # 验证签名
    result = verify(message, public_key, sig)
    print("签名验证结果（True=有效，False=无效）:", result)
    
    # 3. 对指定明文“科学是第一生产力!”签名并验证
    specified_message = "科学是第一生产力!"
    print("\n指定明文:", specified_message)
    specified_sig = sign(specified_message, private_key)
    print("指定明文的RSA数字签名:", specified_sig)
    specified_result = verify(specified_message, public_key, specified_sig)
    print("指定明文的签名验证结果:", specified_result)

if __name__ == "__main__":
    main()