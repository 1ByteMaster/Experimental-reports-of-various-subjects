# isPrimeNumber1.py
# 实验三：大整数素性检测（6N±1 法）
import math

def isPrime(n):
    """使用 6N±1 法判断 n 是否为素数"""
    if n <= 1:
        return False
    if n <= 3:
        return True
    # 排除能被 2 或 3 整除的数
    if n % 2 == 0 or n % 3 == 0:
        return False
    # 仅检测 6k±1 形式的数
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

if __name__ == "__main__":
    test_numbers = [2**16, 2**31 - 1, 2**61 - 1]
    print("实验三：大整数素性检测（6N±1 法）\n")
    for num in test_numbers:
        print("检测整数 n = {}".format(num))
        result = isPrime(num)
        print("结果：{}".format("是素数" if result else "不是素数"))