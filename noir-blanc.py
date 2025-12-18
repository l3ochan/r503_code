#!/usr/bin/env python3
"""
Compression d'une image en niveaux de gris suivant le schéma du cours JPEG :

Etapes implémentées :
  1) Découpage en blocs 8x8 (segmentation)
  2) DCT 2D sur chaque bloc
  3) Quantification non uniforme
  4) Lecture en zigzag
  5) Codage RLE (codage entropique simple)

On stocke le résultat dans un fichier binaire .cmp.
On peut ensuite décompresser vers une image JPG (en niveaux de gris).

Usage :
    python mini_jpeg.py compress  input.gif  sortie.cmp
    python mini_jpeg.py decompress  entree.cmp  sortie.jpg
"""

import sys
import math
from pathlib import Path

import numpy as np
from PIL import Image

BLOCK_SIZE = 8


# ---------------------------------------------------------
#  MATRICE DE QUANTIFICATION (luminance JPEG classique)
# ---------------------------------------------------------

Q_LUMA = np.array([
    [16, 11, 10, 16, 24,  40,  51,  61],
    [12, 12, 14, 19, 26,  58,  60,  55],
    [14, 13, 16, 24, 40,  57,  69,  56],
    [14, 17, 22, 29, 51,  87,  80,  62],
    [18, 22, 37, 56, 68, 109, 103,  77],
    [24, 35, 55, 64, 81, 104, 113,  92],
    [49, 64, 78, 87,103, 121, 120, 101],
    [72, 92, 95, 98,112, 100, 103,  99],
], dtype=np.float64)


# ---------------------------------------------------------
#  MATRICE DCT 8x8 (DCT-II orthonormée)
# ---------------------------------------------------------

def make_dct_matrix(n=8):
    C = np.zeros((n, n), dtype=np.float64)
    factor = math.pi / (2 * n)
    for u in range(n):
        if u == 0:
            alpha = math.sqrt(1 / n)
        else:
            alpha = math.sqrt(2 / n)
        for x in range(n):
            C[u, x] = alpha * math.cos((2 * x + 1) * u * factor)
    return C


DCT_MAT = make_dct_matrix(BLOCK_SIZE)
IDCT_MAT = DCT_MAT.T  # pour une DCT orthonormée, l'inverse est la transpose


def dct2(block):
    """
    DCT 2D : C * block * C^T
    block : matrice 8x8 (float)
    """
    return DCT_MAT @ block @ DCT_MAT.T


def idct2(coeffs):
    """
    DCT inverse 2D : C^T * coeffs * C
    """
    return IDCT_MAT @ coeffs @ IDCT_MAT.T


# ---------------------------------------------------------
#  ZIGZAG
# ---------------------------------------------------------

def build_zigzag_indices(n=8):
    """
    Construit l'ordre de parcours en zigzag pour une matrice n x n.
    Retourne une liste de (i, j) de longueur n*n.
    """
    indices = []
    for s in range(2 * n - 1):
        if s % 2 == 0:
            # diagonale "vers le haut"
            for i in range(s, -1, -1):
                j = s - i
                if i < n and j < n:
                    indices.append((i, j))
        else:
            # diagonale "vers le bas"
            for j in range(s, -1, -1):
                i = s - j
                if i < n and j < n:
                    indices.append((i, j))
    return indices


ZIGZAG_INDICES = build_zigzag_indices(BLOCK_SIZE)


def zigzag_scan(block):
    """
    Transforme un bloc 8x8 en vecteur 1D de 64 coefficients
    en suivant l'ordre zigzag.
    """
    return np.array([block[i, j] for (i, j) in ZIGZAG_INDICES], dtype=np.int32)


def inverse_zigzag(vector):
    """
    Transforme un vecteur 1D de 64 coefficients en bloc 8x8
    en inversant l'ordre zigzag.
    """
    block = np.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=np.int32)
    for k, (i, j) in enumerate(ZIGZAG_INDICES):
        block[i, j] = vector[k]
    return block


# ---------------------------------------------------------
#  RLE (Run-Length Encoding) sur la suite des coefficients
# ---------------------------------------------------------

def rle_encode(coeffs):
    """
    Encode la liste de 64 coefficients quantifiés avec un RLE de type JPEG :
    - On compte les zéros successifs avant chaque coefficient non nul.
    - On ne stocke pas explicitement les zéros de fin de bloc.
    coeffs : array 1D de 64 entiers

    Retourne une liste de (run_zeros, value_nonzero).
    """
    result = []
    zeros = 0
    for c in coeffs:
        if c == 0:
            zeros += 1
        else:
            result.append((zeros, int(c)))
            zeros = 0
    # Les zéros de fin ne sont pas encodés (EOB implicite)
    return result


def rle_decode(pairs):
    """
    Décodage du RLE défini ci-dessus vers un vecteur de 64 coefficients.
    pairs : liste de (run_zeros, value_nonzero)
    """
    coeffs = [0] * 64
    pos = 0
    for run, val in pairs:
        pos += run
        if pos >= 64:
            break  # Sécurité
        coeffs[pos] = val
        pos += 1
        if pos >= 64:
            break
    # Les positions restantes sont déjà à 0
    return np.array(coeffs, dtype=np.int32)


# ---------------------------------------------------------
#  DIVISION EN BLOCS ET RECONSTRUCTION
# ---------------------------------------------------------

def pad_image_to_block(img_array, block_size=6):
    """
    Ajoute du padding (répétition du dernier pixel) pour que
    largeur et hauteur soient multiples de block_size.
    """
    h, w = img_array.shape
    new_h = ((h + block_size - 1) // block_size) * block_size
    new_w = ((w + block_size - 1) // block_size) * block_size

    padded = np.zeros((new_h, new_w), dtype=img_array.dtype)
    padded[:h, :w] = img_array

    # Répéter la dernière ligne / colonne pour remplir
    if new_h > h:
        padded[h:new_h, :w] = img_array[h-1:h, :]
    if new_w > w:
        padded[:, w:new_w] = padded[:, w-1:w]

    return padded, h, w


def image_to_blocks(img_array, block_size=8):
    """
    Découpe une image 2D (numpy) en blocs 8x8.
    Retourne :
        blocks : tableau (nb_blocks, 8, 8)
        blocks_y, blocks_x : nombre de blocs en hauteur et largeur
    """
    padded, h, w = pad_image_to_block(img_array, block_size)
    H, W = padded.shape
    blocks_y = H // block_size
    blocks_x = W // block_size

    blocks = []
    for by in range(blocks_y):
        for bx in range(blocks_x):
            y0 = by * block_size
            x0 = bx * block_size
            block = padded[y0:y0+block_size, x0:x0+block_size]
            blocks.append(block)
    blocks = np.stack(blocks, axis=0)
    return blocks, blocks_y, blocks_x, h, w


def blocks_to_image(blocks, blocks_y, blocks_x, orig_h, orig_w, block_size=8):
    """
    Recompose l'image à partir des blocs 8x8.
    """
    H = blocks_y * block_size
    W = blocks_x * block_size
    img = np.zeros((H, W), dtype=np.float64)

    idx = 0
    for by in range(blocks_y):
        for bx in range(blocks_x):
            y0 = by * block_size
            x0 = bx * block_size
            img[y0:y0+block_size, x0:x0+block_size] = blocks[idx]
            idx += 1

    # On coupe le padding pour revenir à la taille d'origine
    img = img[:orig_h, :orig_w]
    return img


# ---------------------------------------------------------
#  COMPRESSION
# ---------------------------------------------------------

def compress_image(input_path, output_path, quality_factor=6):
    """
    Compression "mini-JPEG" d'une image GIF (ou autre) en niveaux de gris.

    - Lecture de l'image
    - Conversion en gris
    - Découpage en blocs 8x8
    - Décalage des niveaux (valeurs -128..+127)
    - DCT 2D
    - Quantification
    - Zigzag
    - RLE
    - Ecriture d'un fichier binaire .cmp
    """
    # 1. Lecture & gris
    img = Image.open(input_path).convert("L")  # niveaux de gris
    img_arr = np.array(img, dtype=np.float64)

    # Segmentation + padding
    blocks, blocks_y, blocks_x, orig_h, orig_w = image_to_blocks(img_arr, BLOCK_SIZE)

    # Prépare la matrice de quantification (on peut ajuster avec quality_factor)
    Q = Q_LUMA * quality_factor

    # Fichier de sortie
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "wb") as f:
        # HEADER :
        # 4 octets : magique "CMP1"
        # 2 octets : largeur originale
        # 2 octets : hauteur originale
        # 2 octets : blocks_x
        # 2 octets : blocks_y
        f.write(b"CMP1")
        f.write(orig_w.to_bytes(2, "big"))
        f.write(orig_h.to_bytes(2, "big"))
        f.write(blocks_x.to_bytes(2, "big"))
        f.write(blocks_y.to_bytes(2, "big"))

        # Pour chaque bloc :
        for b in blocks:
            # Décalage de -128 (comme dans le cours, on passe à [-128,127]) :contentReference[oaicite:6]{index=6}
            shifted = b - 128.0

            # DCT 2D
            dct_block = dct2(shifted)

            # Quantification
            q_block = np.round(dct_block / Q).astype(np.int32)
            
            

            # Zigzag
            zz = zigzag_scan(q_block)

            # RLE
            rle = rle_encode(zz)

            # Ecriture du bloc :
            # 2 octets : nombre de paires
            # Puis, pour chaque paire :
            #   1 octet : run (0..63)
            #   2 octets : valeur (int16)
            n_pairs = len(rle)
            f.write(n_pairs.to_bytes(2, "big"))
            for run, val in rle:
                if run > 255:
                    # On n'est jamais censé dépasser 63 avec des blocs 8x8,
                    # mais on sécurise.
                    run = 255
                f.write(run.to_bytes(1, "big", signed=False))
                f.write(int(val).to_bytes(2, "big", signed=True))

    print(f"Compression terminée : {output_path}")
    print(f"Dimensions originales : {orig_w} x {orig_h} px")
    print(f"Nombre de blocs : {blocks_x} x {blocks_y} = {blocks_x * blocks_y}")


# ---------------------------------------------------------
#  DECOMPRESSION
# ---------------------------------------------------------

def decompress_image(input_path, output_jpg_path, quality_factor=6):
    """
    Décompression du format .cmp défini dans compress_image,
    puis sauvegarde en JPG (niveaux de gris).
    """
    with open(input_path, "rb") as f:
        magic = f.read(4)
        if magic != b"CMP1":
            raise ValueError("Format de fichier inconnu (magic != CMP1)")

        orig_w = int.from_bytes(f.read(2), "big")
        orig_h = int.from_bytes(f.read(2), "big")
        blocks_x = int.from_bytes(f.read(2), "big")
        blocks_y = int.from_bytes(f.read(2), "big")

        nb_blocks = blocks_x * blocks_y

        Q = Q_LUMA * quality_factor

        blocks = []

        for _ in range(nb_blocks):
            # Lire le nombre de paires RLE
            n_pairs = int.from_bytes(f.read(2), "big")
            pairs = []
            for _ in range(n_pairs):
                run = int.from_bytes(f.read(1), "big", signed=False)
                val = int.from_bytes(f.read(2), "big", signed=True)
                pairs.append((run, val))

            # RLE inverse → zigzag inverse → bloc quantifié
            zz = rle_decode(pairs)
            q_block = inverse_zigzag(zz)

            # Déquantification
            dct_block = q_block * Q
            
            # IDCT
            spatial = idct2(dct_block)

            # Re-décalage +128 et clamp
            block = spatial + 128.0
            block = np.clip(block, 0, 255)
            blocks.append(block)

    blocks = np.stack(blocks, axis=0)

    # Reconstruction de l'image
    img_arr = blocks_to_image(blocks, blocks_y, blocks_x, orig_h, orig_w, BLOCK_SIZE)
    img_arr = np.clip(img_arr, 0, 255).astype(np.uint8)

    img = Image.fromarray(img_arr, mode="L")
    output_jpg_path = Path(output_jpg_path)
    output_jpg_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_jpg_path, format="JPEG", quality=90)

    print(f"Décompression terminée, image sauvegardée en : {output_jpg_path}")


# ---------------------------------------------------------
#  CLI
# ---------------------------------------------------------

def main():
    if len(sys.argv) < 4:
        print("Usage :")
        print("  Compression   : python mini_jpeg.py compress   entree.gif  sortie.cmp")
        print("  Décompression : python mini_jpeg.py decompress entree.cmp  sortie.jpg")
        sys.exit(1)

    mode = sys.argv[1].lower()
    in_path = sys.argv[2]
    out_path = sys.argv[3]

    if mode == "compress":
        compress_image(in_path, out_path)
    elif mode == "decompress":
        decompress_image(in_path, out_path)
    else:
        print("Mode inconnu :", mode)
        sys.exit(1)


if __name__ == "__main__":
    main()
