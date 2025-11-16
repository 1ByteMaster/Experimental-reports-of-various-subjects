import hashlib
import random

"""
实验2：MAC（消息认证码）实现与验证
目标：基于密钥和明文计算MAC值，验证不同密钥对MAC值的影响
"""

print("=" * 50)
print("实验2：MAC（消息认证码）实现与验证")
print("=" * 50)

# 步骤1：生成第一个密钥 k1
# 生成64位随机整数（范围：0 ~ 2^64 - 1）
n1 = random.randint(0, 2**64 - 1)
print(f"\n生成的随机整数 n1: {n1}")

# 将整数转换为字节流，再计算SHA256哈希值作为密钥k1
k1 = hashlib.sha256(str(n1).encode()).hexdigest()
print(f"密钥 k1 (SHA256哈希): {k1}")
print("  密钥k1需严格保密，仅收发双方知晓！")

# 步骤2：从message1.txt读取明文
try:
    with open("message1.txt", "r", encoding="UTF-8") as f:
        # 读取第一行作为明文m（即实验1中的m1）
        m = f.readline().strip()  # strip()去除换行符
    print(f"\n从 message1.txt 读取的明文 m: {m}")
except FileNotFoundError:
    print("\n错误：message1.txt 文件不存在！请先运行实验1（MDC.py）")
    exit()

# 步骤3：计算第一个MAC值（MAC1 = H(m || k1)）
print("\n正在计算 MAC1 值（使用密钥 k1）...")

# 将明文m的字节流与密钥k1的字节流拼接
m_k1 = m.encode("UTF-8") + k1.encode("UTF-8")

# MD5算法
mac1_md5 = hashlib.md5(m_k1).hexdigest()

# SHA1算法
mac1_sha1 = hashlib.sha1(m_k1).hexdigest()

# SHA256算法
mac1_sha256 = hashlib.sha256(m_k1).hexdigest()

# SHA512算法
mac1_sha512 = hashlib.sha512(m_k1).hexdigest()

# 在控制台显示结果
print(f"\nMAC1-MD5:    {mac1_md5}")
print(f"MAC1-SHA1:   {mac1_sha1}")
print(f"MAC1-SHA256: {mac1_sha256}")
print(f"MAC1-SHA512: {mac1_sha512}")

# 步骤4：将MAC1结果追加到message1.txt文件
with open("message1.txt", "a", encoding="UTF-8") as f:
    f.write(f"MAC1-MD5:{mac1_md5}\n")
    f.write(f"MAC1-SHA1:{mac1_sha1}\n")
    f.write(f"MAC1-SHA256:{mac1_sha256}\n")
    f.write(f"MAC1-SHA512:{mac1_sha512}\n")

print("\n✓ MAC1 值已追加保存到 message1.txt")

# 步骤5：生成第二个密钥 k2
print("\n" + "=" * 50)
n2 = 123456  # 按要求设定为固定值123456
print(f"设定的整数 n2: {n2}")

# 计算k2
k2 = hashlib.sha256(str(n2).encode()).hexdigest()
print(f"密钥 k2 (SHA256哈希): {k2}")
print("  密钥k2需严格保密，仅收发双方知晓！")

# 步骤6：计算第二个MAC值（MAC2 = H(m || k2)）
print("\n正在计算 MAC2 值（使用密钥 k2）...")

# 将明文m的字节流与密钥k2的字节流拼接
m_k2 = m.encode("UTF-8") + k2.encode("UTF-8")

# MD5算法
mac2_md5 = hashlib.md5(m_k2).hexdigest()

# SHA1算法
mac2_sha1 = hashlib.sha1(m_k2).hexdigest()

# SHA256算法
mac2_sha256 = hashlib.sha256(m_k2).hexdigest()

# SHA512算法
mac2_sha512 = hashlib.sha512(m_k2).hexdigest()

# 在控制台显示结果
print(f"\nMAC2-MD5:    {mac2_md5}")
print(f"MAC2-SHA1:   {mac2_sha1}")
print(f"MAC2-SHA256: {mac2_sha256}")
print(f"MAC2-SHA512: {mac2_sha512}")

# 步骤7：将MAC2结果追加到message1.txt文件
with open("message1.txt", "a", encoding="UTF-8") as f:
    f.write(f"MAC2-MD5:{mac2_md5}\n")
    f.write(f"MAC2-SHA1:{mac2_sha1}\n")
    f.write(f"MAC2-SHA256:{mac2_sha256}\n")
    f.write(f"MAC2-SHA512:{mac2_sha512}\n")

print("\n✓ MAC2 值已追加保存到 message1.txt")

# 步骤8：结果分析
print("\n" + "=" * 50)
print("MAC 结果分析：")
print("=" * 50)

print("\nMAC值长度统计：")
print(f"MD5:    {len(mac1_md5)} 个16进制字符 ({len(mac1_md5) * 4} 位)")
print(f"SHA1:   {len(mac1_sha1)} 个16进制字符 ({len(mac1_sha1) * 4} 位)")
print(f"SHA256: {len(mac1_sha256)} 个16进制字符 ({len(mac1_sha256) * 4} 位)")
print(f"SHA512: {len(mac1_sha512)} 个16进制字符 ({len(mac1_sha512) * 4} 位)")

print("\n密钥对MAC值的影响分析：")
print(f"使用相同明文 m: {m}")
print(f"\n使用密钥 k1 时：")
print(f"  MAC1-SHA256: {mac1_sha256}")
print(f"\n使用密钥 k2 时：")
print(f"  MAC2-SHA256: {mac2_sha256}")

# 计算MAC值差异
def count_diff(mac1, mac2):
    return sum(c1 != c2 for c1, c2 in zip(mac1, mac2))

print(f"\nSHA256算法下，MAC1与MAC2的差异: {count_diff(mac1_sha256, mac2_sha256)}/64 个字符")
print("\n结论：相同明文使用不同密钥，生成的MAC值完全不同！")
print("这验证了MAC算法的安全性：攻击者无法在不知道密钥的情况下伪造有效的MAC值。")
print("=" * 50)