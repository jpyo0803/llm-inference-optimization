import torch
import torch.nn as nn
import custom_backend  # C++ Extension 모듈

def custom_matmul(a, b):
    # a: [M, K]
    # b: [K, N]

    # 메모리 연속성 보장
    if not a.is_contiguous():
        a = a.contiguous()
    if not b.is_contiguous():
        b = b.contiguous()

    # Custom Backend MatMul 호출
    output = custom_backend.matmul(a, b)  # [M, N]

    return output

def custom_qk(q, k):
    # q: [Batch, Head, Q Len, Head Dim]
    # k: [Batch, Head, K Len, Head Dim]

    # 메모리 연속성 보장
    if not q.is_contiguous():
        q = q.contiguous()
    if not k.is_contiguous():
        k = k.contiguous()

    # Custom Backend QK 호출
    output = custom_backend.qk(q, k)  # [Batch, Head, Q Len, K Len]

    return output

def custom_pv(p, v):
    # p: [Batch, Head, Q Len, V Len]
    # v: [Batch, Head, V Len, Head Dim]

    # 메모리 연속성 보장
    if not p.is_contiguous():
        p = p.contiguous()
    if not v.is_contiguous():
        v = v.contiguous()

    # Custom Backend PV 호출
    output = custom_backend.pv(p, v)  # [Batch, Head, Q Len, Head Dim]

    return output