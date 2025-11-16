import random
import math
import hashlib
from cryptography.fernet import Fernet
from cryptography.hazmat.backends import default_backend  
import os

# ---------------------- 1. RSA核心工具函数（复用并保持稳定） ----------------------
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
        raise Exception("无模逆元，无法生成RSA密钥")
    return x % phi_n

def is_probable_prime(n, k=5):
    """Miller-Rabin素性检测（确保生成大素数的安全性）"""
    if n < 2:
        return False
    for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]:
        if n % p == 0:
            return n == p
    # 分解 n-1 = 2^s * t
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

def get_prime(bits=1024):
    """生成指定位数的大素数（默认1024bit，符合实验安全要求）"""
    while True:
        num = random.getrandbits(bits)
        num |= 1  # 确保为奇数（排除偶数，提高素性检测效率）
        if is_probable_prime(num):
            return num

def generate_rsa_key(bits=1024):
    """生成RSA密钥对（公钥：(n,e)，私钥：(n,d)）"""
    p = get_prime(bits)
    q = get_prime(bits)
    while q == p:  # 确保p和q是不同素数
        q = get_prime(bits)
    n = p * q
    phi_n = (p - 1) * (q - 1)  # 欧拉函数
    e = 65537  # 常用公钥指数（安全且计算效率高）
    while get_GCD(e, phi_n) != 1:  # 确保e与phi_n互质
        e = random.randrange(3, phi_n, 2)
    d = mod_inverse(e, phi_n)  # 计算私钥指数d
    return (n, e), (n, d)

def rsa_encrypt(data_int, public_key):
    """RSA公钥加密（输入整数，输出模n后的整数）"""
    n, e = public_key
    return pow(data_int, e, n)

def rsa_decrypt(cipher_int, private_key):
    """RSA私钥解密（输入加密后的整数，输出原始整数）"""
    n, d = private_key
    return pow(cipher_int, d, n)

def rsa_sign(hash_int, private_key):
    """RSA私钥签名（对文件哈希值签名，确保不可否认性）"""
    return rsa_encrypt(hash_int, private_key)  # 签名本质是私钥加密哈希值

def rsa_verify(hash_int, signature, public_key):
    """RSA公钥验证签名（校验文件完整性和来源合法性）"""
    verified_hash_int = rsa_decrypt(signature, public_key)  # 验证本质是公钥解密签名
    return hash_int == verified_hash_int

# ---------------------- 2. 实验核心工具函数（AES+文件Hash） ----------------------
def get_file_hash(file_path):
    """计算文件的SHA-256哈希值（16进制字符串输出，符合实验要求）"""
    sha256 = hashlib.sha256()
    with open(file_path, 'rb') as file:
        while True:
            data = file.read(65536)  # 每次读取64KB缓存，避免大文件内存溢出
            if not data:
                break
            sha256.update(data)
    return sha256.hexdigest()  # 返回16进制哈希值（固定64字符）

def generate_aes_key():
    """生成256位AES密钥（Fernet封装，自动符合AES-256标准，返回Base64编码bytes）"""
    return Fernet.generate_key()  # 生成的密钥为44字符Base64编码（解码后32字节=256位）

def aes_encrypt_file(input_file, aes_key, output_file="密文.pdf"):
    """AES加密文件（输入原始文件路径，输出密文文件，符合实验“密文.pdf”命名要求）"""
    fernet = Fernet(aes_key)  # 用AES密钥初始化加密器
    with open(input_file, 'rb') as f_in:
        file_data = f_in.read()  # 读取原始文件二进制数据
    encrypted_data = fernet.encrypt(file_data)  # AES加密
    with open(output_file, 'wb') as f_out:
        f_out.write(encrypted_data)  # 保存密文文件
    print(f"✅ AES加密完成，密文文件已保存为：{output_file}")

def aes_decrypt_file(encrypted_file, aes_key, output_file="解密后的3.密码学实验3.pdf"):
    """AES解密文件（输入密文文件路径，输出解密后文件，便于对比原始文件）"""
    fernet = Fernet(aes_key)  # 用解密得到的AES密钥初始化解密器
    with open(encrypted_file, 'rb') as f_in:
        encrypted_data = f_in.read()  # 读取密文二进制数据
    decrypted_data = fernet.decrypt(encrypted_data)  # AES解密
    with open(output_file, 'wb') as f_out:
        f_out.write(decrypted_data)  # 保存解密文件
    print(f"✅ AES解密完成，解密文件已保存为：{output_file}")

# ---------------------- 3. Alice/Bob核心操作函数（修复AES密钥转换逻辑） ----------------------
def AliceDo(original_file, bob_public_key, alice_private_key):
    """Alice端操作：生成AES密钥→加密文件→签名文件Hash→加密AES密钥（实验核心流程）"""
    # 步骤1：生成256位AES密钥（Base64编码，便于RSA整数转换）
    aes_key = generate_aes_key()  # 类型：bytes（44字符Base64编码）
    aes_key_str = aes_key.decode('utf-8')  # 转为字符串（固定44字符，避免字节长度问题）
    print(f"\nAlice操作：")
    print(f"   生成的AES密钥（Base64）：{aes_key_str}")
    
    # 步骤2：计算原始文件Hash值并签名（确保文件完整性和不可否认性）
    file_hash_hex = get_file_hash(original_file)
    file_hash_int = int(file_hash_hex, 16)  # 哈希值转整数（用于RSA签名）
    file_signature = rsa_sign(file_hash_int, alice_private_key)
    print(f"   对文件的RSA签名（整数）：{file_signature}")
    
    # 步骤3：AES加密原始文件（生成“密文.pdf”）
    aes_encrypt_file(original_file, aes_key)
    
    # 步骤4：用Bob公钥加密AES密钥（确保密钥传输安全，修复核心转换逻辑）
    aes_key_bytes = aes_key_str.encode('utf-8')  # 字符串转字节（长度固定44字节）
    aes_key_int = int.from_bytes(aes_key_bytes, byteorder='big')  # 字节转整数（可逆）
    encrypted_aes_key = rsa_encrypt(aes_key_int, bob_public_key)
    print(f"   用Bob公钥加密后的AES密钥：{encrypted_aes_key}")
    
    # 返回给Bob的关键数据（密文文件需手动传输，此处返回加密后的密钥和签名）
    return encrypted_aes_key, file_signature, aes_key

def BobDo(encrypted_file, encrypted_aes_key, file_signature, bob_private_key, alice_public_key):
    """Bob端操作：解密AES密钥→解密文件→验证签名→校验完整性（实验核心流程）"""
    # 步骤1：用Bob私钥解密AES密钥（修复长度错误，可逆恢复）
    decrypted_aes_key_int = rsa_decrypt(encrypted_aes_key, bob_private_key)  # 解密得到整数
    # 计算整数对应的字节长度（避免固定32字节导致的溢出）
    byte_length = (decrypted_aes_key_int.bit_length() + 7) // 8  # 向上取整到字节
    aes_key_bytes = decrypted_aes_key_int.to_bytes(byte_length, byteorder='big')  # 整数转字节
    # 恢复为Base64字符串（去除可能的冗余空字节，确保44字符）
    aes_key_str = aes_key_bytes.decode('utf-8').strip('\x00')
    aes_key = aes_key_str.encode('utf-8')  # 转回bytes类型（用于Fernet解密）
    
    # 步骤2：AES解密“密文.pdf”
    print(f"\nBob操作：")
    print(f"   解密得到的AES密钥（Base64）：{aes_key_str}")
    aes_decrypt_file(encrypted_file, aes_key)
    
    # 步骤3：计算解密后文件的Hash值（用于完整性校验）
    decrypted_file = "解密后的3.密码学实验3.pdf"
    decrypted_file_hash_hex = get_file_hash(decrypted_file)
    decrypted_file_hash_int = int(decrypted_file_hash_hex, 16)
    
    # 步骤4：用Alice公钥验证签名（校验来源合法性和完整性）
    verify_result = rsa_verify(decrypted_file_hash_int, file_signature, alice_public_key)
    if verify_result:
        print(f"签名验证通过！文件来源：Alice，文件未被篡改")
    else:
        print(f"签名验证失败！文件可能被篡改或签名非法")
    return verify_result

# ---------------------- 4. 主函数（执行完整实验流程） ----------------------
def main():
    # 初始化提示（确保实验前置条件满足）
    print("="*60)
    print("          实验2：RSA+AES+Hash数字签名综合应用")
    print("="*60)
    original_file = "3.密码学实验3.pdf"
    if not os.path.exists(original_file):
        raise FileNotFoundError(f"未找到原始文件：{original_file}，请将该文件放在代码同级目录！")
    
    # 步骤1：生成Alice和Bob的RSA密钥对（1024bit，符合实验安全要求）
    print("\n生成Alice和Bob的RSA密钥对（1024bit）：")
    alice_public_key, alice_private_key = generate_rsa_key(bits=1024)
    bob_public_key, bob_private_key = generate_rsa_key(bits=1024)
    print(f"   Alice公钥：(n={alice_public_key[0]}, e={alice_public_key[1]})")
    print(f"   Alice私钥：(n={alice_private_key[0]}, d={alice_private_key[1]})")
    print(f"   Bob公钥：(n={bob_public_key[0]}, e={bob_public_key[1]})")
    print(f"   Bob私钥：(n={bob_private_key[0]}, d={bob_private_key[1]})")
    
    # 步骤2：Alice执行加密和签名操作
    encrypted_aes_key, file_signature, _ = AliceDo(original_file, bob_public_key, alice_private_key)
    
    # 步骤3：Bob执行解密和验证操作（模拟接收Alice的密文、加密密钥、签名）
    encrypted_file = "密文.pdf"
    BobDo(encrypted_file, encrypted_aes_key, file_signature, bob_private_key, alice_public_key)
    

if __name__ == "__main__":
    main()