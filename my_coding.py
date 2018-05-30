# -*- coding: utf-8 -*-
import sys
import hashlib
import math
import argparse
import numpy as np
from scipy.linalg import hadamard
import copy
from base64 import b64encode, b64decode


def to_bits(s: str) -> str:
    result = []
    for c in s:
        bits = bin(ord(c))[2:]
        bits = '00000000'[len(bits):] + bits
        result.extend([int(b) for b in bits])
    return result


def from_bits(bits: list) -> str:
    chars = []
    for b in range(len(bits) // 8):
        byte = bits[b*8:(b+1)*8]
        chars.append(chr(int(''.join([str(bit) for bit in byte]), 2)))
    return ''.join(chars)


def random_errors(v: np.array, t: int) -> np.array:
    n_tmp = len(v)
    res = copy.deepcopy(v)
    for i in range(t):
        index = np.random.randint(n_tmp)
        res[index] = (res[index] + 1) % 2
    return res


def inverse_matrix(input: np.array) -> np.array:
    A = copy.deepcopy(input)
    if A.shape[0] != A.shape[1]:
        print("wrong dimensions")
        return
    elif np.around(np.linalg.det(A)) % 2 == 0:
        print("det == 0")
        return
    else:
        I = np.zeros(shape=A.shape, dtype=int)
        for i in range(A.shape[0]):
            I[i][i] = 1

        for i in range(A.shape[0]):
            if A[i][i] != 1:
                offset = np.argmax(A[i:, i])
                A[i] = (A[i] + A[i + offset]) % 2
                I[i] = (I[i] + I[i + offset]) % 2
            for j in range(i):
                if A[j][i] == 1:
                    A[j] = (A[j] + A[i]) % 2
                    I[j] = (I[j] + I[i]) % 2

            for j in range(i+1, A.shape[0]):
                if A[j][i] == 1:
                    A[j] = (A[j] + A[i]) % 2
                    I[j] = (I[j] + I[i]) % 2
            # print(i)
            # print(A)
            # print(I)
            # print()
    return I


def hamming_matrix(code_dim: int) -> np.ndarray:
    # проверочная матрица кода Хэмминга (n = 2^code_dim - 1, k = 2^code_dim - 1 - code_dim)
    assert(code_dim < 100)
    tmp = []
    for i in range(1, 2**code_dim):
        bits = bin(i)[2:]
        bits = ('0'*code_dim)[len(bits):] + bits
        tmp.extend(bits)
    result = np.array(tmp).reshape(-1, code_dim).T
    return result  # H.T


def reed_muller_matrix(code_dim: int) -> np.ndarray:
    # порождающая матрица кода Рида-Маллера 1-го порядка (n = 2^code_dim, k = code_dim)
    assert(code_dim < 100)
    tmp = []
    for i in range(2**code_dim):
        bits = bin(i)[2:]
        bits = '1'+('0'*code_dim)[len(bits):] + bits
        tmp.extend([int(i) for i in bits])
    result = np.array(tmp).reshape(-1, code_dim+1).T
    return result  # G


def encode_reed_muller(x: np.array, G: np.array) -> np.array:
    if x.shape[0] != G.shape[0] and \
       x.shape[1] != G.shape[0]:
        print('wrong shape of vector in encoding')
        exit(1)
    return x @ G


# def adamar_matrix(size: int) -> np.array:
#     if size < 0:
#         print('incorrect size adamar matrix')
#         exit(1)
#     else:
#         H = np.array(1)
#         for i in range(size):
#             H = np.array([H, H, H, -H]).resize(2**i)
#         return H


def decode_reed_muller(y: np.array) -> np.array:
    if isinstance(y, list):
        y = np.array(y)
    y = 2*y - 1
    m = int(np.log2(y.shape[0]))

    H = hadamard(y.shape[0])
    A = (1 + H) // 2

    y_hat = H @ y
    if sum(y_hat == 0) == len(y_hat):
        pass  # syndrome = 0
    elif np.argmax(y_hat) > 0:
        y1 = A[np.argmax(y_hat)]
    else:
        y1 = 1 - A[np.argmax(y_hat)]

    x = np.zeros(shape=m+1, dtype=int)
    x[0] = y1[0]
    for i in range(m):
        x[m-i] = (x[0] + y1[2**i]) % 2
    return x


# def encrypt_block(block: np.array, P: np.array, G: np.array, M: np.array) -> np.array:
#     G_dash = M @ G @ P
#     w = block @ G_dash
#     return w


def algo1_encrypt_message(msg: str, block_length: int = 64) -> str:
    if not isinstance(msg, str):
        print('invalid argument to encrypt')
        exit(1)
    decoded_bits = to_bits(msg)
    partition = [decoded_bits[i*block_length: (i+1)*block_length] for i in range(len(decoded_bits) // block_length)]
    if len(partition[-1]) != block_length:
        partition[-1].extend(np.zeros(block_length-len(partition[-1])))

    k = int(round(np.log2(block_length) + 1))
    P = np.random.random_integers(0, 1, size=(2 ** k, 2 ** k))
    while np.around(np.linalg.det(P)) % 2 == 0:
        P = np.random.random_integers(0, 1, size=(2 ** k, 2 ** k))

    G = reed_muller_matrix(k)  # (k+1, 2 ** k)
    M = np.random.random_integers(0, 1, size=(2 ** k, 2 ** k))
    while np.around(np.linalg.det(M)) % 2 == 0:
        M = np.random.random_integers(0, 1, size=(2 ** k, 2 ** k))

    p = np.random.randint((2**k) * 1 // 4, (2**k) * 3 // 4)
    M_first, M_second = M[:p, :], M[p:, :]
    t = 2 ** (k-1)

    G_stroke = G @ M % 2
    encrypted = [algo1_encrypt(i, G_stroke, M_first, t) for i in partition]

    print(partition)


def sign_message(msg: str, block_length: int = 32) -> str:
    if isinstance(msg, str):
        msg = msg.encode()
    hashed = hashlib.sha3_256(msg)
    decoded = b64encode(hashed.digest()).decode()
    decoded_bits = to_bits(decoded)
    partition = [decoded_bits[i*block_length: (i+1)*block_length] for i in range(len(decoded_bits) // block_length)]
    # if len(partition[-1]) != block_length:
    #     partition[-1].extend(np.zeros(block_length-len(partition[-1])))
    #
    k = int(round(np.log2(block_length) + 1))
    # P = np.random.random_integers(0, 1, size=(2**k - 1, 2**k - 1));
    f = True
    while f:
        try:
            P = np.random.random_integers(0, 1, size=(2 ** k, 2 ** k));
            np.linalg.inv(P)
            f = False
        except np.linalg.LinAlgError:
            pass


    # H = hamming_matrix(t);
    G = reed_muller_matrix(k) # (k+1, 2 ** k)

    # M = np.random.random_integers(0, 1, size=(2 ** k - 1 - k, 2 ** k - 1 - k));
    f = True
    while f:
        try:
            M = np.random.random_integers(0, 1, size=(k+1, k+1));
            np.linalg.inv(M)
            f = False
        except np.linalg.LinAlgError:
            pass

    encrypted = [encrypt_block(partition[i], P, G, M) for i in range(len(partition))]
    print(partition)


def algo1_encrypt(u: np.array, G_stroke: np.array, M_first: np.array, t: int) -> np.array:
    v = u @ G_stroke % 2
    e = np.zeros(M_first.shape[0], dtype=int)
    e = random_errors(e, t)
    e_stroke = e @ M_first % 2
    w = (v + e_stroke) % 2
    return w


def algo1_decrypt(w: np.array, G: np.array, M: np.array, t: int) -> np.array:
    M_invers = inverse_matrix(M)
    w_stroke = w @ M_invers % 2
    res = decode_reed_muller(w_stroke)
    return res


def test(dim: int):
    k = dim
    G = reed_muller_matrix(k)
    M = np.random.random_integers(0, 1, size=(2 ** k, 2 ** k))
    while np.around(np.linalg.det(M)) % 2 == 0:
        M = np.random.random_integers(0, 1, size=(2 ** k, 2 ** k))
    p = np.random.randint(1, 2**k)
    M1 = M[:p, :]
    M2 = M[p:, :]
    G_stroke = G @ M % 2
    print('enter msg len {:}'.format(dim))
    msg = input()
    binary = to_bits(msg)
    print("binary msg: ",binary)
    u = binary[:dim+1]
    print("u: ",u)
    encrypted = algo1_encrypt(u, G_stroke, M1, 2**(k-1))
    print("encrypted: ", encrypted)
    decrypted = algo1_decrypt(encrypted, G, M, 2**(k-1))
    print("decrypted: ", decrypted)

# if __name__ == '__main__':
#     print("input message")
#     s = input()
#     encrypt_message(s)
#
#     print('')
