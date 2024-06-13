from copy import deepcopy

import numpy as np
from PIL import Image
from tqdm import tqdm

from utils import *


# 编码器的建立
class Encoder:
    def __init__(self) -> None:
        pass

    def encode(self, img: Image.Image, path, q=2):
        self.outer = outer(path)
        img = np.array(img.convert("YCbCr"), dtype=np.int16)

        # 将图像的尺寸信息写入码流
        bits = tuple()
        h, w = img.shape[:-1]
        bits = bits + uint2bin(h, depth=16) + uint2bin(w, depth=16)
        self.outer.out(bits)

        # 对图像进行DCT编码
        img = self._dct_encode(img, q)

        img = self._rle_encode(img)
        img = np.array(img)

        # 对图像进行哈夫曼编码
        self._huffman_encode(img.reshape([-1]))

        self.outer.close()

    # DCT编码
    def _dct_encode(self, img: np.ndarray, q):
        factor_matrix = np.asarray(
            [
                [0.35355339,  0.35355339,  0.35355339,  0.35355339,
                 0.35355339,  0.35355339,  0.35355339,  0.35355339],
                [0.49039264,  0.41573481,  0.27778512,  0.09754516,
                 -0.09754516, -0.27778512, -0.41573481, -0.49039264],
                [0.46193977,  0.19134172, -0.19134172, -0.46193977,
                 -0.46193977, -0.19134172,  0.19134172,  0.46193977],
                [0.41573481, -0.09754516, -0.49039264, -0.27778512,
                 0.27778512,  0.49039264,  0.09754516, -0.41573481],
                [0.35355339, -0.35355339, -0.35355339,  0.35355339,
                 0.35355339, -0.35355339, -0.35355339,  0.35355339],
                [0.27778512, -0.49039264,  0.09754516,  0.41573481,
                 -0.41573481, -0.09754516,  0.49039264, -0.27778512],
                [0.19134172, -0.46193977,  0.46193977, -0.19134172,
                 -0.19134172,  0.46193977, -0.46193977,  0.19134172],
                [0.09754516, -0.27778512,  0.41573481, -0.49039264,
                 0.49039264, -0.41573481,  0.27778512, -0.09754516]
            ]
        )
        Ycoef = np.asarray(
            [
                [16, 11, 10, 16, 24, 40, 51, 61],
                [12, 12, 14, 19, 26, 58, 60, 55],
                [14, 13, 16, 24, 40, 57, 69, 56],
                [14, 17, 22, 29, 51, 87, 80, 62],
                [18, 22, 37, 56, 68, 109, 103, 77],
                [24, 35, 55, 64, 81, 104, 113, 92],
                [49, 64, 78, 87, 103, 121, 120, 101],
                [72, 92, 95, 98, 112, 100, 103, 99],
            ]
        )
        Ccoef = np.asarray(
            [
                [17, 18, 24, 47, 66, 99, 99, 99],
                [18, 21, 26, 66, 99, 99, 99, 99],
                [24, 26, 56, 99, 99, 99, 99, 99],
                [47, 66, 99, 99, 99, 99, 99, 99],
                [99, 99, 99, 99, 99, 99, 99, 99],
                [99, 99, 99, 99, 99, 99, 99, 99],
                [99, 99, 99, 99, 99, 99, 99, 99],
                [99, 99, 99, 99, 99, 99, 99, 99],
            ]
        )
        zigzag_table = [
            [0,  1,  5,  6, 14, 15, 27, 28],
            [2,  4,  7, 13, 16, 26, 29, 42],
            [3,  8, 12, 17, 25, 30, 41, 43],
            [9, 11, 18, 24, 31, 40, 44, 53],
            [10, 19, 23, 32, 39, 45, 52, 54],
            [20, 22, 33, 38, 46, 51, 55, 60],
            [21, 34, 37, 47, 50, 56, 59, 61],
            [35, 36, 48, 49, 57, 58, 62, 63],
        ]
        padding_shape = [0, 0, 3]
        padding_shape[0] = img.shape[0] if img.shape[0] % 8 == 0 \
            else (img.shape[0] // 8 + 1) * 8
        padding_shape[1] = img.shape[1] if img.shape[1] % 8 == 0 \
            else (img.shape[0] // 8 + 1) * 8
        DCT_coef = np.zeros(padding_shape)
        DCT_coef[:img.shape[0], :img.shape[1], :] = img
        for c in range(3):
            for i in range(padding_shape[0]//8):
                for j in range(padding_shape[1]//8):
                    coef = factor_matrix @ DCT_coef[i *
                                                    8:(i+1)*8, j*8:(j+1)*8, c] @ factor_matrix.T
                    if c == 0:
                        DCT_coef[i*8:(i+1)*8, j*8:(j+1)*8,
                                 c] = coef / Ycoef / q
                    else:
                        DCT_coef[i*8:(i+1)*8, j*8:(j+1)*8,
                                 c] = coef / Ccoef / q
        return DCT_coef.astype(np.int32)

    # RLE编码
    def _rle_encode(self, image: np.ndarray):
        def zigzag_serialize(matrix):
            # if not matrix or not matrix[0]:
            #     return []

            rows, cols = len(matrix), len(matrix[0])
            result = []
            up = True

            row, col = 0, 0
            while row < rows and col < cols:
                result.append(matrix[row][col])
                new_row = row + (-1 if up else 1)
                new_col = col + (1 if up else -1)

                if new_row >= rows or new_row < 0 or new_col >= cols or new_col < 0:
                    if up:
                        if col + 1 < cols:
                            col += 1
                        else:
                            row += 1
                    else:
                        if row + 1 < rows:
                            row += 1
                        else:
                            col += 1
                    up = not up
                else:
                    row, col = new_row, new_col

            return result

        def rle_encode_numeric(data):
            if not data:
                return []

            encoding = []
            prev_num = data[0]
            count = 1

            for num in data[1:]:
                if num == prev_num:
                    count += 1
                else:
                    encoding.append(prev_num)
                    encoding.append(count)
                    prev_num = num
                    count = 1
            encoding.append(prev_num)
            encoding.append(count)
            return encoding

        h, w = image.shape[:2]
        res = []
        for c in range(3):
            for i in range(h // 8):
                for j in range(w // 8):
                    block = image[i*8:(i+1)*8, j*8:(j+1)*8, c]
                    zig_block = zigzag_serialize(block)
                    rle_block = rle_encode_numeric(zig_block)
                    res.extend(rle_block)
        return res


    # 哈夫曼编码
    def _huffman_encode(self, sig: np.ndarray):
        # 统计图像信息，得到每个像素值的分布概率
        symbs = [i for i in range(sig.min(), sig.max()+1)]
        symbs, probs = hist(sig, symbs)

        # 根据像素值和分布概率建立哈夫曼树
        huffman_dict = huffman(symbs, probs)
        """
        ================ CODE FORMAT ================
        huffman dict:
            +--------+---------------+---------+
            | symbol |  len of code  |   code  |
            +--------+---------------+---------+
            | 9 bits |     5 bits    |  n bits |
            +--------+---------------+---------+
        img codes:
            bits, with 0s filling behind
        =============================================
        """
        # 将哈夫曼表写入码流
        for k in huffman_dict.keys():
            # 使用255移码，将[-255, 255]的像素值放缩到[0, 511]之间
            self.outer.out(uint2bin(k+255, depth=9))
            self.outer.out(uint2bin(len(huffman_dict[k]), depth=5))
            self.outer.out(huffman_dict[k])

        # 以二进制码1 1111 1111作为EOF，分割哈夫曼表区与图像编码区
        self.outer.out(uint2bin(511, depth=9))
        # 将图像的长度写入码流
        self.outer.out(uint2bin(len(sig), depth=32))
        # 将三通道图像展开成向量，进行编码
        sig = np.reshape(sig, [-1])
        # 将每个像素按照哈夫曼编码，写入码流
        for i in tqdm(range(len(sig)), "编码图像"):
            self.outer.out(huffman_dict[sig[i]])

        return


# 解码器的建立
class Decoder:
    def __init__(self) -> None:
        pass

    def decode(self, path: str, q=2):
        self.inner = inner(path)

        # 从码流中获取图像的尺寸信息
        bits = tuple()
        for _ in range(32):
            bits = bits + self.inner.in_()
        h, w = bin2uint(bits[:16]), bin2uint(bits[16:])

        # 对码流进行哈夫曼解码
        dct_h = h if h % 8 == 0 else (h // 8 + 1) * 8
        dct_w = w if w % 8 == 0 else (w // 8 + 1) * 8
        img = self._huffman_decode(dct_h, dct_w)

        self.inner.close()

        img = self._rle_decode(img, dct_h, dct_w)

        # 对差分图像进行DCT解码
        img = self._dct_decode(img, q)
        img = Image.fromarray(img[:h, :w, :], mode="YCbCr").convert("RGB")
        return img

    # 哈夫曼解码
    def _huffman_decode(self, height: int, width: int) -> np.ndarray:
        """
        ================ CODE FORMAT ================
        huffman dict:
            +--------+---------------+---------+
            | symbol |  len of code  |   code  |
            +--------+---------------+---------+
            | 9 bits |     5 bits    |  n bits |
            +--------+---------------+---------+
        img codes:
            bits, with 0s filling behind
        =============================================
        """
        # 从码流中读取哈夫曼表
        huffman_dict = {}
        while True:
            # 9 bit码流作为像素值，检测到EOF时退出循环，开始读取图像
            symb_bits = tuple()
            for _ in range(9):
                symb_bits = symb_bits + self.inner.in_()
            symb = bin2uint(symb_bits)
            if symb == 511:
                break
            else:
                symb -= 255

            # 5 bit码流作为编码长度值
            len_bits = tuple()
            for _ in range(5):
                len_bits = len_bits + self.inner.in_()
            length = bin2uint(len_bits)

            # 根据读取的编码长度读取编码
            code = tuple()
            for _ in range(length):
                code = code + self.inner.in_()

            # 将键值对写入哈夫曼表
            huffman_dict[code] = symb

        # 读取图像的长度
        bits = tuple()
        for _ in range(32):
            bits = bits + self.inner.in_()
        img_len = bin2uint(bits)

        # 开始读取图像
        img = []
        codes = huffman_dict.keys()
        bits = tuple()
        while True:
            while True:
                bits = bits + self.inner.in_()
                # 当文件读取完毕时，self.inner会返回空值，并将self.inner.current_byte设置为-1
                # 用这种方法防止溢出
                if (self.inner.current_byte == -1):
                    bits = bits + self.inner.in_()
                    break
                if bits in codes:
                    break
            # 生成图像
            if bits in codes:
                img.append(huffman_dict[bits])
            if self.inner.current_byte == -1:
                break
            bits = tuple()
        while len(img) > img_len:
            img.pop()
        return np.array(img)

    def _rle_decode(self, img: np.ndarray, height: int, width: int):
        def rle_decode_numeric(encoded_data):
            decoded = []
            n = len(encoded_data)
            for i in range(0, n, 2):
                decoded.extend([encoded_data[i]] * encoded_data[i + 1])
            return decoded

        def zigzag_deserialize(serialized, rows, cols):
            if not serialized:
                return []

            matrix = [[0] * cols for _ in range(rows)]
            up = True

            index = 0
            row, col = 0, 0
            while row < rows and col < cols:
                matrix[row][col] = serialized[index]
                index += 1
                new_row = row + (-1 if up else 1)
                new_col = col + (1 if up else -1)

                if new_row >= rows or new_row < 0 or new_col >= cols or new_col < 0:
                    if up:
                        if col + 1 < cols:
                            col += 1
                        else:
                            row += 1
                    else:
                        if row + 1 < rows:
                            row += 1
                        else:
                            col += 1
                    up = not up
                else:
                    row, col = new_row, new_col

            return matrix

        img = rle_decode_numeric(img)
        res = np.zeros([height, width, 3], dtype=np.int16)
        idx = 0
        for c in range(3):
            for i in range(height // 8):
                for j in range(width // 8):

                    res[i * 8:(i + 1) * 8, j * 8:(j + 1) * 8, c] = zigzag_deserialize(img[idx : idx + 64], 8, 8)
                    idx += 64
        return res



    # DCT解码
    def _dct_decode(self, DCT_coef: np.ndarray, q):
        DCT_coef = DCT_coef.astype(np.int32)
        factor_matrix = np.asarray(
            [
                [0.35355339,  0.35355339,  0.35355339,  0.35355339,
                 0.35355339,  0.35355339,  0.35355339,  0.35355339],
                [0.49039264,  0.41573481,  0.27778512,  0.09754516,
                 -0.09754516, -0.27778512, -0.41573481, -0.49039264],
                [0.46193977,  0.19134172, -0.19134172, -0.46193977,
                 -0.46193977, -0.19134172,  0.19134172,  0.46193977],
                [0.41573481, -0.09754516, -0.49039264, -0.27778512,
                 0.27778512,  0.49039264,  0.09754516, -0.41573481],
                [0.35355339, -0.35355339, -0.35355339,  0.35355339,
                 0.35355339, -0.35355339, -0.35355339,  0.35355339],
                [0.27778512, -0.49039264,  0.09754516,  0.41573481,
                 -0.41573481, -0.09754516,  0.49039264, -0.27778512],
                [0.19134172, -0.46193977,  0.46193977, -0.19134172,
                 -0.19134172,  0.46193977, -0.46193977,  0.19134172],
                [0.09754516, -0.27778512,  0.41573481, -0.49039264,
                 0.49039264, -0.41573481,  0.27778512, -0.09754516]
            ]
        )
        Ycoef = np.asarray(
            [
                [16, 11, 10, 16, 24, 40, 51, 61],
                [12, 12, 14, 19, 26, 58, 60, 55],
                [14, 13, 16, 24, 40, 57, 69, 56],
                [14, 17, 22, 29, 51, 87, 80, 62],
                [18, 22, 37, 56, 68, 109, 103, 77],
                [24, 35, 55, 64, 81, 104, 113, 92],
                [49, 64, 78, 87, 103, 121, 120, 101],
                [72, 92, 95, 98, 112, 100, 103, 99],
            ]
        )
        Ccoef = np.asarray(
            [
                [17, 18, 24, 47, 66, 99, 99, 99],
                [18, 21, 26, 66, 99, 99, 99, 99],
                [24, 26, 56, 99, 99, 99, 99, 99],
                [47, 66, 99, 99, 99, 99, 99, 99],
                [99, 99, 99, 99, 99, 99, 99, 99],
                [99, 99, 99, 99, 99, 99, 99, 99],
                [99, 99, 99, 99, 99, 99, 99, 99],
                [99, 99, 99, 99, 99, 99, 99, 99],
            ]
        )
        img = np.zeros_like(DCT_coef)
        for c in range(3):
            for i in range(DCT_coef.shape[0]//8):
                for j in range(DCT_coef.shape[1]//8):
                    if c == 0:
                        coef = DCT_coef[i*8:(i+1)*8, j *
                                        8:(j+1)*8, c] * Ycoef * q
                    else:
                        coef = DCT_coef[i*8:(i+1)*8, j *
                                        8:(j+1)*8, c] * Ccoef * q
                    img[i*8:(i+1)*8, j*8:(j+1)*8,
                        c] = factor_matrix.T @ coef @ factor_matrix
        return np.clip(img, 0, 255).astype(np.uint8)
