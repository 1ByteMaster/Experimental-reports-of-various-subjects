import random
import time
from rsa import get_prime, mod_inverse, get_GCD, encrypt, decrypt
def generate_key(bits=1024):
    p = get_prime(bits)
    q = get_prime(bits)
    while q == p:
        q = get_prime(bits)
    n = p * q
    phi_n = (p - 1) * (q - 1)
    e = 65537  # 固定公钥指数
    if get_GCD(e, phi_n) != 1:
        raise ValueError("e 与 phi_n 不互质")
    d = mod_inverse(e, phi_n)
    return (n, e), (n, d)

def main():
    print("正在生成 1024 位 RSA 密钥对，请稍候...")
    start = time.time()
    public_key, private_key = generate_key(1024)
    end = time.time()
    print(f"密钥生成完成，用时 {end - start:.2f} 秒")

    n, e = public_key
    _, d = private_key
    print("模数 n 位数:", n.bit_length(), "bits")
    print("Public Key (n, e):", public_key)
    print("Private Key (n, d):", private_key)

    message = "hello world! 这是一条明文消息"
    print("\n原始明文:", message)

    # 加密
    start_enc = time.time()
    ciphertext = encrypt(message, public_key)
    end_enc = time.time()
    print("加密用时:", end_enc - start_enc, "秒")

    # 解密
    start_dec = time.time()
    decrypted = decrypt(ciphertext, private_key)
    end_dec = time.time()
    print("解密用时:", end_dec - start_dec, "秒")

    print("解密结果:", decrypted)


if __name__ == "__main__":
    main()
