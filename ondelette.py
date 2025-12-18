#!/bin/env/python3

import numpy as np
import cv2
import pywt

def dwt2(img, wave='haar', level=1):
    coeffs = pywt.wavedec2(img, wavelet=wave, level=level)
    arr, coeff_slices = pywt.coeffs_to_array(coeffs)
    return arr, coeff_slices

def idwt2(arr, coeff_slices, wave='haar'):
    coeffs = pywt.array_to_coeffs(arr, coeff_slices, output_format='wavedec2')
    return pywt.waverec2(coeffs, wavelet=wave)

def quantize(arr, q):
    return np.round(arr / q).astype(np.int32)

def dequantize(arr, q):
    return arr * q

def rle_encode(arr_flat):
    result = []
    count = 1
    prev = arr_flat[0]
    for val in arr_flat[1:]:
        if val == prev:
            count += 1
        else:
            result.append((prev, count))
            prev = val
            count = 1
    result.append((prev, count))
    return result

def rle_decode(sequence):
    arr = []
    for val, count in sequence:
        arr.extend([val] * count)
    return np.array(arr, dtype=np.int32)

def compress_image(path, wave='haar', level=2, q=10):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.float64)

    if img.ndim == 2:
        arr, slices = dwt2(img, wave=wave, level=level)
        q_arr = quantize(arr, q)
        rle = rle_encode(q_arr.flatten())
        return ('gray', img.shape, slices, rle, q, wave, level)

    else:
        h, w, c = img.shape
        data = ['color', (h, w), q, wave, level]
        for i in range(3):
            arr, slices = dwt2(img[:, :, i], wave=wave, level=level)
            q_arr = quantize(arr, q)
            rle = rle_encode(q_arr.flatten())
            data.append((slices, rle))
        return tuple(data)

def decompress_image(data):
    if data[0] == 'gray':
        _, shape, slices, rle, q, wave, level = data
        flat = rle_decode(rle)
        arr = flat.reshape(pywt.coeffs_to_array(pywt.wavedec2(np.zeros(shape), wavelet=wave, level=level))[0].shape)
        arr = dequantize(arr, q)
        rec = idwt2(arr, slices, wave)
        return np.clip(rec, 0, 255).astype(np.uint8)

    else:
        _, (h, w), q, wave, level = data[:4]
        rec = np.zeros((h, w, 3), dtype=np.float64)
        for i in range(3):
            slices, rle = data[4 + i]
            flat = rle_decode(rle)
            arr = flat.reshape(pywt.coeffs_to_array(pywt.wavedec2(np.zeros((h, w)), wavelet=wave, level=level))[0].shape)
            arr = dequantize(arr, q)
            rec[:, :, i] = idwt2(arr, slices, wave)
        return np.clip(rec, 0, 255).astype(np.uint8)
