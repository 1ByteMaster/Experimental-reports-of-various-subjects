import logging

class DES:
    def __init__(self, debug=False):
        self.debug = debug
        if debug:
            logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    # 初始置换表 (IP)
    IP = [
        58, 50, 42, 34, 26, 18, 10, 2,
        60, 52, 44, 36, 28, 20, 12, 4,
        62, 54, 46, 38, 30, 22, 14, 6,
        64, 56, 48, 40, 32, 24, 16, 8,
        57, 49, 41, 33, 25, 17, 9, 1,
        59, 51, 43, 35, 27, 19, 11, 3,
        61, 53, 45, 37, 29, 21, 13, 5,
        63, 55, 47, 39, 31, 23, 15, 7
    ]
    
    # 逆初始置换表 (IP⁻¹)
    IP_INV = [
        40, 8, 48, 16, 56, 24, 64, 32,
        39, 7, 47, 15, 55, 23, 63, 31,
        38, 6, 46, 14, 54, 22, 62, 30,
        37, 5, 45, 13, 53, 21, 61, 29,
        36, 4, 44, 12, 52, 20, 60, 28,
        35, 3, 43, 11, 51, 19, 59, 27,
        34, 2, 42, 10, 50, 18, 58, 26,
        33, 1, 41, 9, 49, 17, 57, 25
    ]
    
    # 扩展置换表 (E)
    E = [
        32, 1, 2, 3, 4, 5, 
        4, 5, 6, 7, 8, 9, 
        8, 9, 10, 11, 12, 13,
        12, 13, 14, 15, 16, 17,
        16, 17, 18, 19, 20, 21,
        20, 21, 22, 23, 24, 25, 
        24, 25, 26, 27, 28, 29,
        28, 29, 30, 31, 32, 1
    ]
    
    # P盒置换表
    P = [
        16, 7, 20, 21, 29, 12, 28, 17,
        1, 15, 23, 26, 5, 18, 31, 10,
        2, 8, 24, 14, 32, 27, 3, 9,
        19, 13, 30, 6, 22, 11, 4, 25
    ]
    
    # 密钥置换表PC-1
    PC1 = [
        57, 49, 41, 33, 25, 17, 9, 1,
        58, 50, 42, 34, 26, 18, 10, 2,
        59, 51, 43, 35, 27, 19, 11, 3,
        60, 52, 44, 36, 63, 55, 47, 39,
        31, 23, 15, 7, 62, 54, 46, 38,
        30, 22, 14, 6, 61, 53, 45, 37,
        29, 21, 13, 5, 28, 20, 12, 4
    ]
    
    # 密钥置换表PC-2
    PC2 = [
        14, 17, 11, 24, 1, 5, 3, 28,
        15, 6, 21, 10, 23, 19, 12, 4,
        26, 8, 16, 7, 27, 20, 13, 2,
        41, 52, 31, 37, 47, 55, 30, 40,
        51, 45, 33, 48, 44, 49, 39, 56,
        34, 53, 46, 42, 50, 36, 29, 32
    ]
    
    # 每轮左移位数
    SHIFT_SCHEDULE = [1, 1, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 1]
    
    # S盒
    S_BOXES = [
        # S1
        [
            [14, 4, 13, 1, 2, 15, 11, 8, 3, 10, 6, 12, 5, 9, 0, 7],
            [0, 15, 7, 4, 14, 2, 13, 1, 10, 6, 12, 11, 9, 5, 3, 8],
            [4, 1, 14, 8, 13, 6, 2, 11, 15, 12, 9, 7, 3, 10, 5, 0],
            [15, 12, 8, 2, 4, 9, 1, 7, 5, 11, 3, 14, 10, 0, 6, 13]
        ],
        # S2
        [
            [15, 1, 8, 14, 6, 11, 3, 4, 9, 7, 2, 13, 12, 0, 5, 10],
            [3, 13, 4, 7, 15, 2, 8, 14, 12, 0, 1, 10, 6, 9, 11, 5],
            [0, 14, 7, 11, 10, 4, 13, 1, 5, 8, 12, 6, 9, 3, 2, 15],
            [13, 8, 10, 1, 3, 15, 4, 2, 11, 6, 7, 12, 0, 5, 14, 9]
        ],
        # S3
        [
            [10, 0, 9, 14, 6, 3, 15, 5, 1, 13, 12, 7, 11, 4, 2, 8],
            [13, 7, 0, 9, 3, 4, 6, 10, 2, 8, 5, 14, 12, 11, 15, 1],
            [13, 6, 4, 9, 8, 15, 3, 0, 11, 1, 2, 12, 5, 10, 14, 7],
            [1, 10, 13, 0, 6, 9, 8, 7, 4, 15, 14, 3, 11, 5, 2, 12]
        ],
        # S4
        [
            [7, 13, 14, 3, 0, 6, 9, 10, 1, 2, 8, 5, 11, 12, 4, 15],
            [13, 8, 11, 5, 6, 15, 0, 3, 4, 7, 2, 12, 1, 10, 14, 9],
            [10, 6, 9, 0, 12, 11, 7, 13, 15, 1, 3, 14, 5, 2, 8, 4],
            [3, 15, 0, 6, 10, 1, 13, 8, 9, 4, 5, 11, 12, 7, 2, 14]
        ],
        # S5
        [
            [2, 12, 4, 1, 7, 10, 11, 6, 8, 5, 3, 15, 13, 0, 14, 9],
            [14, 11, 2, 12, 4, 7, 13, 1, 5, 0, 15, 10, 3, 9, 8, 6],
            [4, 2, 1, 11, 10, 13, 7, 8, 15, 9, 12, 5, 6, 3, 0, 14],
            [11, 8, 12, 7, 1, 14, 2, 13, 6, 15, 0, 9, 10, 4, 5, 3]
        ],
        # S6
        [
            [12, 1, 10, 15, 9, 2, 6, 8, 0, 13, 3, 4, 14, 7, 5, 11],
            [10, 15, 4, 2, 7, 12, 9, 5, 6, 1, 13, 14, 0, 11, 3, 8],
            [9, 14, 15, 5, 2, 8, 12, 3, 7, 0, 4, 10, 1, 13, 11, 6],
            [4, 3, 2, 12, 9, 5, 15, 10, 11, 14, 1, 7, 6, 0, 8, 13]
        ],
        # S7
        [
            [4, 11, 2, 14, 15, 0, 8, 13, 3, 12, 9, 7, 5, 10, 6, 1],
            [13, 0, 11, 7, 4, 9, 1, 10, 14, 3, 5, 12, 2, 15, 8, 6],
            [1, 4, 11, 13, 12, 3, 7, 14, 10, 15, 6, 8, 0, 5, 9, 2],
            [6, 11, 13, 8, 1, 4, 10, 7, 9, 5, 0, 15, 14, 2, 3, 12]
        ],
        # S8
        [
            [13, 2, 8, 4, 6, 15, 11, 1, 10, 9, 3, 14, 5, 0, 12, 7],
            [1, 15, 13, 8, 10, 3, 7, 4, 12, 5, 6, 11, 0, 14, 9, 2],
            [7, 11, 4, 1, 9, 12, 14, 2, 0, 6, 10, 13, 15, 3, 5, 8],
            [2, 1, 14, 7, 4, 10, 8, 13, 15, 12, 9, 0, 3, 5, 6, 11]
        ]
    ]
    
    @staticmethod
    def text_to_bits(text, size=64):
        """将文本转换为位列表"""
        block_size = size // 8
        data = text.encode('utf-8')
        
        # PKCS#7填充
        pad_len = block_size - (len(data) % block_size)
        data += bytes([pad_len] * pad_len)
        
        # 只取第一个块（64位）
        if len(data) > block_size:
            data = data[:block_size]
            
        bits = []
        for byte in data:
            bits += [int(b) for b in f"{byte:08b}"]
        return bits[:size]
    
    @staticmethod
    def bits_to_hex(bits):
        """将位列表转换为十六进制字符串"""
        hex_str = ''
        for i in range(0, len(bits), 4):
            chunk = bits[i:i+4]
            hex_digit = f"{int(''.join(map(str, chunk)), 2):X}"
            hex_str += hex_digit
        return hex_str
    
    @staticmethod
    def bits_to_text(bits):
        """将位列表转换为文本"""
        bytes_list = []
        for i in range(0, len(bits), 8):
            if i+8 > len(bits):
                break
            byte_bits = bits[i:i+8]
            byte_str = ''.join(str(b) for b in byte_bits)
            bytes_list.append(int(byte_str, 2))
            
        # 移除PKCS#7填充
        if bytes_list:
            pad_len = bytes_list[-1]
            if pad_len > 0 and pad_len <= len(bytes_list):
                bytes_list = bytes_list[:-pad_len]
                
        return bytes(bytes_list).decode('utf-8', errors='ignore')
    
    @staticmethod
    def permute(bits, table):
        """根据置换表重排列位（table 以 1 开始索引）"""
        return [bits[i-1] for i in table]
    
    @staticmethod
    def left_shift(bits, n):
        """循环左移"""
        return bits[n:] + bits[:n]
    
    def generate_subkeys(self, key_bits):
        """生成16轮子密钥"""
        # PC-1置换（64位->56位）
        key = self.permute(key_bits, self.PC1)
        
        # 拆分左右两部分
        left = key[:28]
        right = key[28:]
        
        subkeys = []
        for i in range(16):
            # 根据移位表循环左移
            shift = self.SHIFT_SCHEDULE[i]
            left = self.left_shift(left, shift)
            right = self.left_shift(right, shift)
            
            # 确保移位后长度不变
            if len(left) != 28 or len(right) != 28:
                raise ValueError("子密钥生成错误: 移位后长度异常")
            
            # 合并并PC-2置换（56位->48位）
            combined = left + right
            subkey = self.permute(combined, self.PC2)
            subkeys.append(subkey)
        
        return subkeys
    
    def f_function(self, right, subkey):
        """Feistel网络中的F函数"""
        # 1. 扩展置换（32位->48位）
        expanded = self.permute(right, self.E)
        
        # 2. 与子密钥异或
        xored = [e ^ s for e, s in zip(expanded, subkey)]
        
        # 3. S盒替换（48位->32位）
        result = []
        for i in range(8):
            # 6位输入（首位和第6位决定行，中间4位决定列）
            chunk = xored[i*6:(i+1)*6]
            # 行索引 (0-3): 取第1位和第6位
            row = (chunk[0] << 1) | chunk[5]
            # 列索引 (0-15): 取中间4位
            col = (chunk[1] << 3) | (chunk[2] << 2) | (chunk[3] << 1) | (chunk[4])
            
            # 确保行和列在有效范围内
            if row < 0 or row > 3 or col < 0 or col > 15:
                raise ValueError(f"无效的S盒索引: S{i+1} row={row} col={col}")
            
            # S盒取值并转换为4位二进制
            s_value = self.S_BOXES[i][row][col]
            result += [int(b) for b in f"{s_value:04b}"]
        
        # 4. P盒置换
        return self.permute(result, self.P)
    
    def encrypt_block(self, plain_bits, subkeys):
        """加密一个64位数据块"""
        if self.debug:
            logging.info(f"加密输入: {self.bits_to_hex(plain_bits)}")
        
        # 初始置换
        bits = self.permute(plain_bits, self.IP)
        if self.debug:
            logging.info(f"初始置换后: {self.bits_to_hex(bits)}")
        
        # 拆分为左右两部分
        left = bits[:32]
        right = bits[32:]
        
        if self.debug:
            logging.info(f"初始 L0: {self.bits_to_hex(left)}")
            logging.info(f"初始 R0: {self.bits_to_hex(right)}")
        
        # 16轮Feistel网络
        for i in range(16):
            if self.debug:
                logging.info(f"\n--- 轮次 {i+1} ---")
                logging.info(f"子密钥: {self.bits_to_hex(subkeys[i])}")
            
            # 保存当前右半部分（将用于下一轮的左半部分）
            temp_right = right.copy()
            
            # F函数处理当前右半部分
            f_result = self.f_function(right, subkeys[i])
            if self.debug:
                logging.info(f"F函数输出: {self.bits_to_hex(f_result)}")
            
            # 当前左半部分与F函数结果异或 -> 得到新的右半部分
            new_right = [l ^ f for l, f in zip(left, f_result)]
            if self.debug:
                logging.info(f"异或结果: {self.bits_to_hex(new_right)}")
            
            # 更新左右部分（最后一轮不交换）
            if i < 15:  # 前15轮交换
                left = temp_right
                right = new_right
            else:       # 第16轮不交换：left = new_right, right = temp_right
                left = new_right
                right = temp_right
            
            if self.debug:
                logging.info(f"轮后 L{i+1}: {self.bits_to_hex(left)}")
                logging.info(f"轮后 R{i+1}: {self.bits_to_hex(right)}")
        
        # 合并左右部分（注意：最后一轮后不交换，所以是R16+L16）
        combined = left + right
        if self.debug:
            logging.info(f"合并后: {self.bits_to_hex(combined)}")
        
        # 逆初始置换
        cipher_bits = self.permute(combined, self.IP_INV)
        if self.debug:
            logging.info(f"逆初始置换: {self.bits_to_hex(cipher_bits)}")
            logging.info(f"加密输出: {self.bits_to_hex(cipher_bits)}")
        
        return cipher_bits
    
    def decrypt_block(self, cipher_bits, subkeys):
        """解密一个64位数据块"""
        # 解密时子密钥反转顺序
        return self.encrypt_block(cipher_bits, list(reversed(subkeys)))
    
    def encrypt(self, plain_text, key_text):
        """DES加密"""
        plain_bits = self.text_to_bits(plain_text, 64)
        key_bits = self.text_to_bits(key_text, 64)
        
        subkeys = self.generate_subkeys(key_bits)
        cipher_bits = self.encrypt_block(plain_bits, subkeys)
        
        return self.bits_to_text(cipher_bits)
    
    def decrypt(self, cipher_text, key_text):
        """DES解密"""
        cipher_bits = self.text_to_bits(cipher_text, 64)
        key_bits = self.text_to_bits(key_text, 64)
        
        subkeys = self.generate_subkeys(key_bits)
        plain_bits = self.decrypt_block(cipher_bits, subkeys)
        
        return self.bits_to_text(plain_bits)


def test_des():
    """测试DES算法实现"""
    des = DES(debug=True)  # 启用调试模式
    
    # 标准测试向量 (NIST)
    print("\n标准测试向量 (NIST):")
    plain_hex = "0123456789ABCDEF"
    key_hex = "133457799BBCDFF1"
    expected_cipher_hex = "85E813540F0AB405"
    
    # 将十六进制转换为比特
    plain_bits = []
    for char in plain_hex:
        plain_bits += [int(b) for b in f"{int(char, 16):04b}"]
    
    key_bits = []
    for char in key_hex:
        key_bits += [int(b) for b in f"{int(char, 16):04b}"]
    
    print(f"明文: {plain_hex}")
    print(f"密钥: {key_hex}")
    print(f"预期密文: {expected_cipher_hex}")
    
    # 加密
    subkeys = des.generate_subkeys(key_bits)
    cipher_bits = des.encrypt_block(plain_bits, subkeys)
    cipher_hex = des.bits_to_hex(cipher_bits)
    print(f"实际密文: {cipher_hex}")
    
    # 解密
    decrypted_bits = des.decrypt_block(cipher_bits, des.generate_subkeys(key_bits))
    decrypted_hex = des.bits_to_hex(decrypted_bits)
    print(f"解密结果: {decrypted_hex}")
    

if __name__ == "__main__":
    test_des()
