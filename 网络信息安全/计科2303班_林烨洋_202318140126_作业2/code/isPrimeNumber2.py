# isPrimeNumber2.py
# 实验四：大整数素性检测（Miller-Rabin 算法）
import random
import time

# ----------- 6N±1法 -----------
def isPrime_6N(n):
    """确定性素数检测：6N±1法"""
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False

    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True


# ----------- Miller-Rabin算法 -----------
def isPrime_MR(n, k=10):
    """
    Miller-Rabin 素性测试
    n: 待测整数
    k: 随机测试次数（越大越准确）
    """
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0:
        return False

    # 将 n-1 写为 2^s * t (t 为奇数)
    t = n - 1
    s = 0
    while t % 2 == 0:
        t //= 2
        s += 1

    # 随机测试 k 次
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
            return False  # 合数
    return True  # 可能是素数


# ----------- 主程序 -----------
if __name__ == "__main__":
    print("实验四：大整数素性检测（Miller-Rabin算法）\n")

    test_cases = [
        ("2^11 - 1", 2**11 - 1),
        ("2^61 - 1", 2**61 - 1),
        ("2^71", 2**71),
        ("2^89 - 1", 2**89 - 1),
    ]

    for label, n in test_cases:
        print("检测：{} = {}\n".format(label, n))

        # 用 6N±1 法（仅对较小数）
        if n.bit_length() < 70:  # 防止大数过慢
            start = time.time()
            result_6N = isPrime_6N(n)
            t1 = time.time() - start
            print("6N±1 法结果：{}，耗时 {:.6f} 秒".format("是素数" if result_6N else "不是素数", t1))
        else:
            print("6N±1 法检测略过（数值过大，效率低）")

        # 用 Miller-Rabin 算法
        start = time.time()
        result_MR = isPrime_MR(n, k=10)
        t2 = time.time() - start
        print("Miller-Rabin 法结果：{}，耗时 {:.6f} 秒".format("可能是素数" if result_MR else "不是素数", t2))
        print("-" * 70)