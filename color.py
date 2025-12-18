#!/usr/bin/env python3
import sys
import math
from pathlib import Path
import numpy as np
from PIL import Image

# ---------------------------------------------------------
# MATRICES DCT
# ---------------------------------------------------------

def make_dct_matrix(n=8):
    C = np.zeros((n, n), dtype=np.float64)
    factor = math.pi / (2 * n)
    for u in range(n):
        alpha = math.sqrt(1/n) if u == 0 else math.sqrt(2/n)
        for x in range(n):
            C[u, x] = alpha * math.cos((2*x+1)*u*factor)
    return C

BLOCK = 8
DCT = make_dct_matrix(BLOCK)
IDCT = DCT.T     # DCT orthonormée ⇒ inverse = transpose

def dct2(b): return DCT @ b @ DCT.T
def idct2(c): return IDCT @ c @ IDCT.T

# ---------------------------------------------------------
# MATRICES DE QUANTIFICATION (JPEG standard)
# ---------------------------------------------------------

QY = np.array([
    [16,11,10,16,24,40,51,61],
    [12,12,14,19,26,58,60,55],
    [14,13,16,24,40,57,69,56],
    [14,17,22,29,51,87,80,62],
    [18,22,37,56,68,109,103,77],
    [24,35,55,64,81,104,113,92],
    [49,64,78,87,103,121,120,101],
    [72,92,95,98,112,100,103,99]
], dtype=np.float64)

QCh = np.array([
    [17,18,24,47,99,99,99,99],
    [18,21,26,66,99,99,99,99],
    [24,26,56,99,99,99,99,99],
    [47,66,99,99,99,99,99,99],
    [99,99,99,99,99,99,99,99],
    [99,99,99,99,99,99,99,99],
    [99,99,99,99,99,99,99,99],
    [99,99,99,99,99,99,99,99]
], dtype=np.float64)

# ---------------------------------------------------------
# ZIGZAG
# ---------------------------------------------------------

def zigzag_indices(n=8):
    out = []
    for s in range(2*n-1):
        if s % 2 == 0:
            for i in range(s, -1, -1):
                j = s-i
                if i<n and j<n: out.append((i,j))
        else:
            for j in range(s, -1, -1):
                i = s-j
                if i<n and j<n: out.append((i,j))
    return out

ZZ = zigzag_indices(BLOCK)

def zigzag(b): return np.array([b[i,j] for i,j in ZZ], dtype=np.int32)
def inv_zigzag(v):
    b = np.zeros((BLOCK,BLOCK),dtype=np.int32)
    for k,(i,j) in enumerate(ZZ): b[i,j]=v[k]
    return b

# ---------------------------------------------------------
# RLE
# ---------------------------------------------------------

def rle_encode(v):
    out = []
    zeros = 0
    for c in v:
        if c == 0:
            zeros += 1
        else:
            out.append((zeros, int(c)))
            zeros = 0
    return out

def rle_decode(pairs):
    v = [0]*64
    pos = 0
    for r,val in pairs:
        pos += r
        if pos>=64: break
        v[pos]=val
        pos+=1
        if pos>=64: break
    return np.array(v,dtype=np.int32)

# ---------------------------------------------------------
# BLOCKING / UNBLOCKING
# ---------------------------------------------------------

def pad(img):
    h,w = img.shape
    H = (h+7)//8 * 8
    W = (w+7)//8 * 8
    out = np.zeros((H,W),dtype=img.dtype)
    out[:h,:w]=img
    out[h:, :] = out[h-1:h, :]
    out[:, w:] = out[:, w-1:w]
    return out,h,w

def to_blocks(img):
    padded,h,w = pad(img)
    H,W = padded.shape
    by, bx = H//8, W//8
    blocks=[]
    for y in range(by):
        for x in range(bx):
            blocks.append(padded[y*8:(y+1)*8, x*8:(x+1)*8])
    return blocks, by, bx, h, w

def from_blocks(blks, by, bx, h, w):
    H = by*8; W = bx*8
    img = np.zeros((H,W),dtype=np.float64)
    k=0
    for y in range(by):
        for x in range(bx):
            img[y*8:(y+1)*8, x*8:(x+1)*8] = blks[k]
            k+=1
    return img[:h,:w]

# ---------------------------------------------------------
# RGB <-> YCbCr
# ---------------------------------------------------------

def rgb_to_ycbcr(img):
    arr = np.array(img,dtype=np.float64)
    R,G,B = arr[:,:,0], arr[:,:,1], arr[:,:,2]
    Y  =  0.299*R + 0.587*G + 0.114*B
    Cb = -0.1687*R - 0.3313*G + 0.5*B   + 128
    Cr =  0.5*R   - 0.4187*G - 0.0813*B + 128
    return Y, Cb, Cr

def ycbcr_to_rgb(Y,Cb,Cr):
    Cb-=128; Cr-=128
    R = Y + 1.402*Cr
    G = Y - 0.34414*Cb - 0.71414*Cr
    B = Y + 1.772*Cb
    return np.clip(np.dstack([R,G,B]),0,255).astype(np.uint8)

# ---------------------------------------------------------
# COMPRESSION
# ---------------------------------------------------------

def compress_color(input_path, output_cmp, q=1.0):
    img = Image.open(input_path).convert("RGB")
    Y, Cb, Cr = rgb_to_ycbcr(img)

    comps = [Y, Cb, Cr]
    Qs = [QY*q, QCh*q, QCh*q]

    out = open(output_cmp,"wb")
    out.write(b"COL1")   # magique
    out.write(img.size[0].to_bytes(2,"big"))
    out.write(img.size[1].to_bytes(2,"big"))

    for channel, Q in zip(comps,Qs):
        blocks, by, bx, h, w = to_blocks(channel)
        out.write(by.to_bytes(2,"big"))
        out.write(bx.to_bytes(2,"big"))

        for b in blocks:
            b = b - 128
            d = dct2(b)
            qblk = np.round(d / Q).astype(np.int32)
            zz = zigzag(qblk)
            rle = rle_encode(zz)
            out.write(len(rle).to_bytes(2,"big"))
            for run,val in rle:
                out.write(run.to_bytes(1,"big"))
                out.write(int(val).to_bytes(2,"big",signed=True))
    out.close()
    print("Compression OK →", output_cmp)

# ---------------------------------------------------------
# DECOMPRESSION
# ---------------------------------------------------------

def decompress_color(input_cmp, output_img, q=1.0):
    f=open(input_cmp,"rb")
    if f.read(4)!=b"COL1": raise ValueError("Wrong format")

    W = int.from_bytes(f.read(2),"big")
    H = int.from_bytes(f.read(2),"big")

    Qs=[QY*q, QCh*q, QCh*q]
    channels=[]

    for Q in Qs:
        by = int.from_bytes(f.read(2),"big")
        bx = int.from_bytes(f.read(2),"big")

        blocks=[]
        for _ in range(by*bx):
            n = int.from_bytes(f.read(2),"big")
            pairs=[]
            for _ in range(n):
                run = int.from_bytes(f.read(1),"big")
                val = int.from_bytes(f.read(2),"big",signed=True)
                pairs.append((run,val))
            zz = rle_decode(pairs)
            blk = inv_zigzag(zz)
            d = blk * Q
            b = idct2(d) + 128
            blocks.append(np.clip(b,0,255))

        chan = from_blocks(blocks,by,bx,H,W)
        channels.append(chan)

    Y,Cb,Cr=channels
    rgb = ycbcr_to_rgb(Y,Cb,Cr)
    Image.fromarray(rgb).save(output_img)
    print("Décompression OK →", output_img)

# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------

if __name__=="__main__":
    if sys.argv[1]=="compress":
        compress_color(sys.argv[2],sys.argv[3], q=1.0)
    else:
        decompress_color(sys.argv[2],sys.argv[3], q=1.0)
