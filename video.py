#!/usr/bin/env python3
import cv2
import sys
import math
from pathlib import Path
import numpy as np
from PIL import Image

# --------------------------------------------------------------------
#  PARTIE COMPRESSION D'IMAGE (RÉUTILISEE TELLE QUELLE POUR LA VIDÉO)
# --------------------------------------------------------------------

BLOCK = 8

def make_dct_matrix(n=8):
    C = np.zeros((n, n), dtype=np.float64)
    factor = math.pi / (2*n)
    for u in range(n):
        alpha = math.sqrt(1/n) if u==0 else math.sqrt(2/n)
        for x in range(n):
            C[u,x] = alpha * math.cos((2*x+1)*u*factor)
    return C

DCT = make_dct_matrix(BLOCK)
IDCT = DCT.T

def dct2(b): return DCT @ b @ DCT.T
def idct2(c): return IDCT @ c @ IDCT.T

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

# zigzag
def zigzag_indices(n=8):
    out=[]
    for s in range(2*n-1):
        if s%2==0:
            for i in range(s,-1,-1):
                j=s-i
                if i<n and j<n: out.append((i,j))
        else:
            for j in range(s,-1,-1):
                i=s-j
                if i<n and j<n: out.append((i,j))
    return out

ZZ = zigzag_indices(8)

def zigzag(b): return np.array([b[i,j] for (i,j) in ZZ], dtype=np.int32)
def inv_zigzag(v):
    b = np.zeros((8,8),dtype=np.int32)
    for k,(i,j) in enumerate(ZZ): b[i,j] = v[k]
    return b

# RLE
def rle_encode(v):
    out=[]
    z=0
    for c in v:
        if c==0:
            z+=1
        else:
            out.append((z,int(c)))
            z=0
    return out

def rle_decode(pairs):
    v=[0]*64
    pos=0
    for r,val in pairs:
        pos+=r
        if pos>=64: break
        v[pos]=val
        pos+=1
        if pos>=64: break
    return np.array(v,dtype=np.int32)

# blocking
def pad(img):
    h,w = img.shape
    H = (h+7)//8*8
    W = (w+7)//8*8
    out=np.zeros((H,W),dtype=img.dtype)
    out[:h,:w]=img
    out[h:,:] = out[h-1:h,:]
    out[:,w:] = out[:,w-1:w]
    return out,h,w

def to_blocks(img):
    padded,h,w = pad(img)
    H,W = padded.shape
    by,bx = H//8, W//8
    blocks=[]
    for y in range(by):
        for x in range(bx):
            blocks.append(padded[y*8:(y+1)*8, x*8:(x+1)*8])
    return blocks,by,bx,h,w

def from_blocks(blocks,by,bx,h,w):
    H=by*8; W=bx*8
    out=np.zeros((H,W),dtype=np.float64)
    k=0
    for y in range(by):
        for x in range(bx):
            out[y*8:(y+1)*8, x*8:(x+1)*8] = blocks[k]
            k+=1
    return out[:h,:w]

# YCbCr conversion
def rgb_to_ycbcr(rgb):
    R,G,B = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    Y  = 0.299*R + 0.587*G + 0.114*B
    Cb = -0.1687*R -0.3313*G +0.5*B +128
    Cr = 0.5*R     -0.4187*G -0.0813*B +128
    return Y,Cb,Cr

def ycbcr_to_rgb(Y,Cb,Cr):
    Cb-=128; Cr-=128
    R = Y + 1.402*Cr
    G = Y - 0.34414*Cb -0.71414*Cr
    B = Y + 1.772*Cb
    return np.clip(np.dstack([R,G,B]),0,255).astype(np.uint8)

# --------------------------------------------------------------------
#  COMPRESSION D'UNE SEULE FRAME (couleur)
# --------------------------------------------------------------------

def compress_frame(rgb, q=1.0, out_file=None):
    Y,Cb,Cr = rgb_to_ycbcr(rgb)
    comps=[Y,Cb,Cr]
    Qs=[QY*q, QCh*q, QCh*q]

    all_data = []

    for comp, Q in zip(comps,Qs):
        blocks,by,bx,h,w = to_blocks(comp)
        all_data.append((by,bx))

        for b in blocks:
            b = b - 128
            d = dct2(b)
            qblk = np.round(d / Q).astype(np.int32)
            zz = zigzag(qblk)
            rle = rle_encode(zz)
            all_data.append(rle)

    return all_data, (rgb.shape[1], rgb.shape[0])  # (W,H)

# --------------------------------------------------------------------
#  DÉCOMPRESSION D'UNE SEULE FRAME
# --------------------------------------------------------------------

def decompress_frame(data_iter, size, q=1.0):
    """ NE SERT PLUS — on ne l’utilise plus dans la version finale.
        La reconstruction est faite directement dans decompress_video().
    """
    raise NotImplementedError("Cette fonction n'est plus utilisée.")


# --------------------------------------------------------------------
#  COMPRESSION VIDÉO
# --------------------------------------------------------------------

def compress_video(input_video, output_cmpv, q=1.0):
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print("Erreur : impossible d’ouvrir", input_video)
        return

    out = open(output_cmpv,"wb")
    out.write(b"CMPV")  # magique

    # Lecture info
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FPS = int(cap.get(cv2.CAP_PROP_FPS))
    FRAMECOUNT = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # header
    out.write(W.to_bytes(2,"big"))
    out.write(H.to_bytes(2,"big"))
    out.write(FPS.to_bytes(2,"big"))
    out.write(FRAMECOUNT.to_bytes(4,"big"))

    print("Compression vidéo...")
    print(f"{FRAMECOUNT} frames, {W}x{H}, {FPS} FPS")

    frame_idx=0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        data,size = compress_frame(frame_rgb, q=q)

        # nombre d’objets stockés pour cette frame
        out.write(len(data).to_bytes(4,"big"))
        for item in data:
            if isinstance(item, tuple):
                # (by,bx)
                out.write((0).to_bytes(1,"big"))   # type 0
                out.write(item[0].to_bytes(2,"big"))
                out.write(item[1].to_bytes(2,"big"))
            else:
                # RLE
                out.write((1).to_bytes(1,"big"))
                out.write(len(item).to_bytes(2,"big"))
                for run,val in item:
                    out.write(run.to_bytes(1,"big"))
                    out.write(val.to_bytes(2,"big",signed=True))

        frame_idx+=1
        print(f"Frame {frame_idx}/{FRAMECOUNT} compressée",end="\r")

    cap.release()
    out.close()
    print("\nCompression terminée :", output_cmpv)

# --------------------------------------------------------------------
#  DÉCOMPRESSION VIDÉO
# --------------------------------------------------------------------

def decompress_video(input_cmpv, output_video, q=1.0):
    f = open(input_cmpv, "rb")

    # Vérification magic
    if f.read(4) != b"CMPV":
        raise ValueError("Format incorrect (magic CMPV manquant)")

    # Header
    W   = int.from_bytes(f.read(2), "big")
    H   = int.from_bytes(f.read(2), "big")
    FPS = int.from_bytes(f.read(2), "big")
    FRAMECOUNT = int.from_bytes(f.read(4), "big")

    print(f"Décompression : {FRAMECOUNT} frames {W}x{H} ({FPS}fps)")

    # --- Création AVI MJPEG (compatible Linux/OpenCV) ---
    # Remplacer .mp4 -> .avi automatiquement
    if output_video.endswith(".mp4"):
        output_avi = output_video[:-4] + ".avi"
    else:
        output_avi = output_video + ".avi"

    print("Écriture AVI intermédiaire :", output_avi)

    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    vw = cv2.VideoWriter(output_avi, fourcc, FPS, (W, H))

    if not vw.isOpened():
        raise RuntimeError("Impossible d'ouvrir VideoWriter pour " + output_avi)

    # --- Décompression frame par frame ---
    for i in range(FRAMECOUNT):

        # Nombre d'items encodés dans cette frame
        n_items = int.from_bytes(f.read(4), "big")
        items = []

        # Lecture items (types: 0 = (by,bx), 1 = RLE)
        for _ in range(n_items):
            typ = int.from_bytes(f.read(1), "big")

            if typ == 0:
                by = int.from_bytes(f.read(2), "big")
                bx = int.from_bytes(f.read(2), "big")
                items.append((by, bx))

            else:
                nb = int.from_bytes(f.read(2), "big")
                rle = []
                for _ in range(nb):
                    run = int.from_bytes(f.read(1), "big")
                    val = int.from_bytes(f.read(2), "big", signed=True)
                    rle.append((run, val))
                items.append(rle)

        # =====================
        # Découper Y / Cb / Cr
        # =====================

        # --- Y ---
        byY, bxY = items[0]
        countY = byY * bxY
        rleY = items[1 : 1 + countY]

        # --- Cb ---
        offset = 1 + countY
        byCb, bxCb = items[offset]
        countCb = byCb * bxCb
        rleCb = items[offset + 1 : offset + 1 + countCb]

        # --- Cr ---
        offset = offset + 1 + countCb
        byCr, bxCr = items[offset]
        countCr = byCr * bxCr
        rleCr = items[offset + 1 : offset + 1 + countCr]

        # ======================================================
        # Reconstruction d'une composante (déquant + IDCT + bloc)
        # ======================================================

        def rebuild_channel(by, bx, rle_list, Q):
            blocks_quant = []
            data_iter = iter(rle_list)

            # Lire blocs quantifiés
            for _ in range(by * bx):
                rle = next(data_iter)
                zz = rle_decode(rle)
                blk_q = inv_zigzag(zz).astype(np.float64)
                blocks_quant.append(blk_q)

            # Déquantification + IDCT
            blocks = []
            for blk in blocks_quant:
                d = blk * Q
                b = idct2(d) + 128
                blocks.append(np.clip(b, 0, 255))

            # Reconstruction image
            return from_blocks(blocks, by, bx, H, W)

        # Matrices de quantification
        Qy  = QY  * q
        Qcb = QCh * q
        Qcr = QCh * q

        # Reconstruction Y, Cb, Cr
        Y  = rebuild_channel(byY,  bxY,  rleY,  Qy)
        Cb = rebuild_channel(byCb, bxCb, rleCb, Qcb)
        Cr = rebuild_channel(byCr, bxCr, rleCr, Qcr)

        # RGB final
        frame = ycbcr_to_rgb(Y, Cb, Cr)

        # Écriture vidéo (OpenCV attend BGR)
        vw.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        print(f"Frame {i+1}/{FRAMECOUNT} décompressée", end="\r")

    vw.release()
    print("\nDécompression terminée : ", output_avi)



# --------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------

if __name__=="__main__":
    if len(sys.argv)<4:
        print("Usage :")
        print("  Compression   : python3 video_cmp.py compress   input.mp4 out.cmpv")
        print("  Décompression : python3 video_cmp.py decompress input.cmpv out_frames")
        sys.exit(1)

    mode=sys.argv[1]

    if mode=="compress":
        compress_video(sys.argv[2], sys.argv[3], q=1.0)
    else:
        decompress_video(sys.argv[2], sys.argv[3], q=1.0)
