import random
from pathlib import Path

from gmssl import sm3
from cryptography.fernet import Fernet, InvalidToken


# ============ 全局变量 ============
K: bytes = None  # 对称密钥（Fernet）
S: int = None    # 16位随机整数状态


# ============ 基础函数 ============
def H(message: str) -> str:
    """
    基于 SM3 的哈希函数：输入 UTF-8 字符串，输出 64 字符的16进制哈希
    """
    msg_bytes = message.encode("UTF-8")
    msg_list = [b for b in msg_bytes]
    return sm3.sm3_hash(msg_list)


def E(plaintext_bytes: bytes, key: bytes) -> bytes:
    """
    基于 Fernet(AES) 的加密：输入明文字节和密钥，返回密文字节
    """
    cipher = Fernet(key)
    return cipher.encrypt(plaintext_bytes)


def D(cipher_bytes: bytes, key: bytes) -> bytes:
    """
    基于 Fernet(AES) 的解密：输入密文字节和密钥，返回明文字节
    """
    cipher = Fernet(key)
    return cipher.decrypt(cipher_bytes)


def Comp(h1: str, h2: str) -> bool:
    """
    比较两个16进制哈希字符串是否相等
    """
    return h1 == h2


# ============ Alice 与 Bob ============
def AliceDo():
    """
    - 读取 message1.txt 的全部内容作为 M（UTF-8）
    - 计算 h = SM3(M||S)
    - 生成串 joined = M + "|" + h
    - 用 K 加密，写入 crypto.txt（latin-1 持久化密文字节）
    """
    p = Path("message1.txt")
    if not p.exists():
        raise FileNotFoundError("未找到 message1.txt，请先完成实验1与实验2以生成该文件。")

    # 读取 M：整个文件内容
    M_str = p.read_text(encoding="utf-8")

    # 计算 h = SM3(M||S)
    h = H(M_str + str(S))

    # 拼接并加密
    joined = M_str + "|" + h
    plaintext_bytes = joined.encode("UTF-8")
    encrypted = E(plaintext_bytes, K)

    # 写入：以 latin-1 映射字节->字符，不丢任何字节，便于人眼查看与手动修改
    with open("crypto.txt", "w", encoding="latin-1") as f:
        f.write(encrypted.decode("latin-1"))

    print("Alice 已生成密文并写入 crypto.txt。")
    print("你可以打开 crypto.txt 查看密文内容。")


def BobDo():
    """
    - 从 crypto.txt 读取密文（latin-1 -> bytes）
    - 用 K 解密，得到 "M|h_alice" 字符串
    - 以 rsplit('|', 1) 拆出 M_bob 与 h_alice
    - 本地计算 h_bob = SM3(M_bob||S) 并比较
    - 友好处理各种异常场景，不抛出未捕获错误
    """
    cpath = Path("crypto.txt")
    if not cpath.exists():
        print("验证失败：未找到 crypto.txt，请先让 Alice 生成密文。")
        return False

    try:
        ctext_str = cpath.read_text(encoding="latin-1")
    except Exception as e:
        print(f"验证失败：读取 crypto.txt 时发生异常：{e}")
        return False

    # 还原密文字节
    try:
        cipher_bytes = ctext_str.encode("latin-1")
    except Exception as e:
        print(f"验证失败：密文编码还原异常：{e}")
        return False

    # 解密（捕获 InvalidToken）
    try:
        decrypted = D(cipher_bytes, K)
    except InvalidToken:
        print("验证失败：密文被篡改（无法解密，InvalidToken）。")
        return False
    except Exception as e:
        print(f"验证失败：解密时发生异常：{e}")
        return False

    # 解析明文
    try:
        decrypted_str = decrypted.decode("UTF-8")
    except Exception as e:
        print(f"验证失败：解密后明文 UTF-8 解码失败：{e}")
        return False

    if "|" not in decrypted_str:
        print("验证失败：解密内容格式异常（未找到分隔符 '|'）。")
        return False

    # 只按最后一个 '|' 分割，避免 M 中出现 '|' 的干扰
    M_bob, h_alice = decrypted_str.rsplit("|", 1)

    # 本地计算哈希并比较
    try:
        h_bob = H(M_bob + str(S))
    except Exception as e:
        print(f"验证失败：本地计算哈希时异常：{e}")
        return False

    if Comp(h_alice, h_bob):
        print("通过验证：消息完整。")
        return True
    else:
        print("验证失败：完整性校验不通过（H(M||S) 不一致）。")
        return False


# ============ 主流程 ============
if __name__ == "__main__":
    # 生成全局 K 与 S（不落盘，进程内共享）
    K = Fernet.generate_key()
    S = random.randint(10**15, 10**16 - 1)  # 16 位随机整数

    print("已生成对称密钥 K（Fernet）与 16 位随机状态 S。")

    # 正常场景：Alice 加密 -> Bob 验证通过
    try:
        AliceDo()
        BobDo()
    except FileNotFoundError as e:
        # 常见：没有 message1.txt
        print(str(e))
    except Exception as e:
        print(f"运行时发生异常：{e}")

    # 篡改测试：提示手工改动 1 个 base64 可见字符
    try:
        input("\n请现在手动打开 crypto.txt，修改其中 一个可见的 base64 字符，保存后按回车继续让 Bob 验证...")
        BobDo()
    except Exception as e:
        print(f"交互阶段发生异常：{e}")
