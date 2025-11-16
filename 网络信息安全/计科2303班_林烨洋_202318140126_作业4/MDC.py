# -*- coding: utf-8 -*-
# 实验2：MAC 实现与验证
# 依赖：random, hashlib
import random
import hashlib
from pathlib import Path

ALGOS = [
    ("MD5", hashlib.md5),
    ("SHA1", hashlib.sha1),
    ("SHA256", hashlib.sha256),
    ("SHA512", hashlib.sha512),
]

def read_plaintext_from_message1():
    """按要求：从 message1.txt 的第一行读取明文 m"""
    p = Path("message1.txt")
    if not p.exists():
        raise FileNotFoundError("未找到 message1.txt，请先运行实验1（MDC.py）生成该文件。")
    with p.open("r", encoding="utf-8") as f:
        first_line = f.readline()
    return first_line.rstrip("\n")

def mac_concat(m: str, k: str):
    """返回 {算法名: 16进制MAC} 的有序列表；按实验定义 MAC=H(m||k)"""
    results = []
    data = m.encode("UTF-8") + k.encode("UTF-8")
    for name, ctor in ALGOS:
        digest = ctor(data).hexdigest()
        results.append((name, digest))
    return results

def append_mac_results_to_message1(prefix: str, items):
    """将 'MAC1-算法名:值' 或 'MAC2-算法名:值' 逐行追加到 message1.txt"""
    with open("message1.txt", "a", encoding="utf-8") as f:
        for name, digest in items:
            f.write(f"{prefix}-{name}:{digest}\n")

def print_length_summary(items, label: str):
    print(f"\n[{label}] 各算法MAC输出长度：")
    for name, digest in items:
        bits = len(digest) * 4
        print(f"  {name}: {bits} 位 / {len(digest)} 个16进制字符")

if __name__ == "__main__":
    # 1) 从文件首行读取明文 m
    m = read_plaintext_from_message1()
    print(f"读取明文 m（来自 message1.txt 第一行）：{m}")

    # 2) 生成 64 位随机整数 n，派生 k1
    n = random.getrandbits(64)  # 0 ~ 2^64 - 1
    k1 = hashlib.sha256(str(n).encode()).hexdigest()
    # 3) 计算 MAC1 并追加
    mac1 = mac_concat(m, k1)
    append_mac_results_to_message1("MAC1", mac1)
    print("已将 MAC1-... 结果追加到 message1.txt。")
    print_length_summary(mac1, "MAC1（k1）")

    # 4) 固定 n2=123456，派生 k2
    n2 = 123456
    k2 = hashlib.sha256(str(n2).encode()).hexdigest()
    mac2 = mac_concat(m, k2)
    append_mac_results_to_message1("MAC2", mac2)
    print("已将 MAC2-... 结果追加到 message1.txt。")
    print_length_summary(mac2, "MAC2（k2）")

    print("\n提示：实验环境中密钥应妥善保密。此脚本未把 k1/k2 写入任何文件。")
