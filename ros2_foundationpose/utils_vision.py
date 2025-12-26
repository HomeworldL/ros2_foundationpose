#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import cv2
import supervision as sv
from PIL import Image
from scipy.ndimage import binary_dilation
import matplotlib.pyplot as plt
import os

np.random.seed(200)
_palette = ((np.random.random((3*255))*0.7+0.3)*255).astype(np.uint8).tolist()
_palette = [0,0,0]+_palette

DAVIS_PALETTE = b"\x00\x00\x00\x80\x00\x00\x00\x80\x00\x80\x80\x00\x00\x00\x80\x80\x00\x80\x00\x80\x80\x80\x80\x80@\x00\x00\xc0\x00\x00@\x80\x00\xc0\x80\x00@\x00\x80\xc0\x00\x80@\x80\x80\xc0\x80\x80\x00@\x00\x80@\x00\x00\xc0\x00\x80\xc0\x00\x00@\x80\x80@\x80\x00\xc0\x80\x80\xc0\x80@@\x00\xc0@\x00@\xc0\x00\xc0\xc0\x00@@\x80\xc0@\x80@\xc0\x80\xc0\xc0\x80\x00\x00@\x80\x00@\x00\x80@\x80\x80@\x00\x00\xc0\x80\x00\xc0\x00\x80\xc0\x80\x80\xc0@\x00@\xc0\x00@@\x80@\xc0\x80@@\x00\xc0\xc0\x00\xc0@\x80\xc0\xc0\x80\xc0\x00@@\x80@@\x00\xc0@\x80\xc0@\x00@\xc0\x80@\xc0\x00\xc0\xc0\x80\xc0\xc0@@@\xc0@@@\xc0@\xc0\xc0@@@\xc0\xc0@\xc0@\xc0\xc0\xc0\xc0\xc0 \x00\x00\xa0\x00\x00 \x80\x00\xa0\x80\x00 \x00\x80\xa0\x00\x80 \x80\x80\xa0\x80\x80`\x00\x00\xe0\x00\x00`\x80\x00\xe0\x80\x00`\x00\x80\xe0\x00\x80`\x80\x80\xe0\x80\x80 @\x00\xa0@\x00 \xc0\x00\xa0\xc0\x00 @\x80\xa0@\x80 \xc0\x80\xa0\xc0\x80`@\x00\xe0@\x00`\xc0\x00\xe0\xc0\x00`@\x80\xe0@\x80`\xc0\x80\xe0\xc0\x80 \x00@\xa0\x00@ \x80@\xa0\x80@ \x00\xc0\xa0\x00\xc0 \x80\xc0\xa0\x80\xc0`\x00@\xe0\x00@`\x80@\xe0\x80@`\x00\xc0\xe0\x00\xc0`\x80\xc0\xe0\x80\xc0 @@\xa0@@ \xc0@\xa0\xc0@ @\xc0\xa0@\xc0 \xc0\xc0\xa0\xc0\xc0`@@\xe0@@`\xc0@\xe0\xc0@`@\xc0\xe0@\xc0`\xc0\xc0\xe0\xc0\xc0\x00 \x00\x80 \x00\x00\xa0\x00\x80\xa0\x00\x00 \x80\x80 \x80\x00\xa0\x80\x80\xa0\x80@ \x00\xc0 \x00@\xa0\x00\xc0\xa0\x00@ \x80\xc0 \x80@\xa0\x80\xc0\xa0\x80\x00`\x00\x80`\x00\x00\xe0\x00\x80\xe0\x00\x00`\x80\x80`\x80\x00\xe0\x80\x80\xe0\x80@`\x00\xc0`\x00@\xe0\x00\xc0\xe0\x00@`\x80\xc0`\x80@\xe0\x80\xc0\xe0\x80\x00 @\x80 @\x00\xa0@\x80\xa0@\x00 \xc0\x80 \xc0\x00\xa0\xc0\x80\xa0\xc0@ @\xc0 @@\xa0@\xc0\xa0@@ \xc0\xc0 \xc0@\xa0\xc0\xc0\xa0\xc0\x00`@\x80`@\x00\xe0@\x80\xe0@\x00`\xc0\x80`\xc0\x00\xe0\xc0\x80\xe0\xc0@`@\xc0`@@\xe0@\xc0\xe0@@`\xc0\xc0`\xc0@\xe0\xc0\xc0\xe0\xc0  \x00\xa0 \x00 \xa0\x00\xa0\xa0\x00  \x80\xa0 \x80 \xa0\x80\xa0\xa0\x80` \x00\xe0 \x00`\xa0\x00\xe0\xa0\x00` \x80\xe0 \x80`\xa0\x80\xe0\xa0\x80 `\x00\xa0`\x00 \xe0\x00\xa0\xe0\x00 `\x80\xa0`\x80 \xe0\x80\xa0\xe0\x80``\x00\xe0`\x00`\xe0\x00\xe0\xe0\x00``\x80\xe0`\x80`\xe0\x80\xe0\xe0\x80  @\xa0 @ \xa0@\xa0\xa0@  \xc0\xa0 \xc0 \xa0\xc0\xa0\xa0\xc0` @\xe0 @`\xa0@\xe0\xa0@` \xc0\xe0 \xc0`\xa0\xc0\xe0\xa0\xc0 `@\xa0`@ \xe0@\xa0\xe0@ `\xc0\xa0`\xc0 \xe0\xc0\xa0\xe0\xc0``@\xe0`@`\xe0@\xe0\xe0@``\xc0\xe0`\xc0`\xe0\xc0\xe0\xe0\xc0"
COLOR_MAP = np.frombuffer(DAVIS_PALETTE, dtype=np.uint8).reshape(-1, 3).copy()
# scales for better visualization
COLOR_MAP = (COLOR_MAP.astype(np.float32) * 1.5).clip(0, 255).astype(np.uint8)

TAG_CHAR = np.array([202021.25], np.float32)

def draw_point(image: np.ndarray, points, labels, radius=5) -> np.ndarray:
    if points == []:
        return image
    for i, point in enumerate(points):
        if labels[i] == 1:
            color = [0, 0, 255]
        else:
            color = [255, 0, 0]
        result_image = cv2.circle(image, tuple(point), radius, color, -1)
    return result_image.astype(np.uint8)

def draw_bbox(image: np.ndarray, bbox, label) -> np.ndarray:
    if bbox is None:
        return image
    detections = sv.Detections(
        xyxy=np.array([bbox]),
        mask=None,
    )
    box_annotator = sv.BoxAnnotator()
    result_image = box_annotator.annotate(scene=image, detections=detections, labels=[label])
    return result_image.astype(np.uint8)

def draw_prompt(image: np.ndarray, points, prompt_labels, bbox, label) -> np.ndarray:
    result_image = draw_point(image, points, prompt_labels)
    result_image = draw_bbox(image, bbox, label)
    return result_image.astype(np.uint8)

def overlay_davis(image: np.ndarray, mask: np.ndarray, alpha: float = 0.5, fade: bool = False):
    """Overlay segmentation on top of RGB image. from davis official"""
    im_overlay = image.copy()

    colored_mask = COLOR_MAP[mask]
    foreground = image * alpha + (1 - alpha) * colored_mask
    binary_mask = mask > 0
    # Compose image
    im_overlay[binary_mask] = foreground[binary_mask]
    if fade:
        im_overlay[~binary_mask] = im_overlay[~binary_mask] * 0.6
    return im_overlay.astype(image.dtype)

def colorize_mask(pred_mask):
    save_mask = Image.fromarray(pred_mask.astype(np.uint8))
    save_mask = save_mask.convert(mode='P')
    save_mask.putpalette(_palette)
    save_mask = save_mask.convert(mode='RGB')
    return np.array(save_mask)

def draw_mask(img, mask, alpha=0.5, id_countour=False):
    img_mask = np.zeros_like(img)
    img_mask = img
    if id_countour:
        # very slow ~ 1s per image
        obj_ids = np.unique(mask)
        obj_ids = obj_ids[obj_ids!=0]

        for id in obj_ids:
            # Overlay color on  binary mask
            if id <= 255:
                color = _palette[id*3:id*3+3]
            else:
                color = [0,0,0]
            foreground = img * (1-alpha) + np.ones_like(img) * alpha * np.array(color)
            binary_mask = (mask == id)

            # Compose image
            img_mask[binary_mask] = foreground[binary_mask]

            countours = binary_dilation(binary_mask,iterations=1) ^ binary_mask
            img_mask[countours, :] = 0
    else:
        binary_mask = (mask!=0)
        countours = binary_dilation(binary_mask,iterations=1) ^ binary_mask
        foreground = img*(1-alpha)+colorize_mask(mask)*alpha
        img_mask[binary_mask] = foreground[binary_mask]
        img_mask[countours,:] = 0
        
    return img_mask.astype(img.dtype)

def overlay_masks(masks):
    mask = np.zeros_like(masks[0]).astype(np.uint8)  # [H, W]
    for i, m in enumerate(masks):
        mask = np.clip(
            mask + m * (i + 1),
            0,
            i + 1,
        )
        
    return mask

def apply_mask(image, mask):
    """
    Apply a mask to an image and return the segmented image.
    
    Args:
        image (numpy.ndarray): The original image.
        mask (numpy.ndarray): The mask to be applied.
    
    Returns:
        numpy.ndarray: The segmented image.
    """
    segmented_image = image * mask[:, :, np.newaxis]
    
    return segmented_image

def compute_overlap(mask1, mask2):
    # 计算交集
    intersection = np.logical_and(mask1, mask2)
    
    # 计算交集像素数
    intersection_pixels = np.count_nonzero(intersection)
    
    # 计算掩码1的像素总数
    mask1_pixels = np.count_nonzero(mask1)
    
    # 计算掩码2的像素总数
    mask2_pixels = np.count_nonzero(mask2)
    
    # 计算重叠率
    overlap_ratio = intersection_pixels / max(mask1_pixels, mask2_pixels)
    
    return overlap_ratio

def readFlow(fn):
    """ Read .flo file in Middlebury format"""
    # Code adapted from:
    # http://stackoverflow.com/questions/28013200/reading-middlebury-flow-files-with-python-bytes-array-numpy

    # WARNING: this will work on little-endian architectures (eg Intel x86) only!
    # print 'fn = %s'%(fn)
    with open(fn, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        if 202021.25 != magic:
            print('Magic number incorrect. Invalid .flo file')
            return None
        else:
            w = np.fromfile(f, np.int32, count=1)
            h = np.fromfile(f, np.int32, count=1)
            # print 'Reading %d x %d flo file\n' % (w, h)
            data = np.fromfile(f, np.float32, count=2*int(w)*int(h))
            # Reshape data into 3D array (columns, rows, bands)
            # The reshape here is for visualization, the original code is (w,h,2)
            return np.resize(data, (int(h), int(w), 2))

def writeFlow(filename,uv,v=None):
    """ Write optical flow to file.
    
    If v is None, uv is assumed to contain both u and v channels,
    stacked in depth.
    Original code by Deqing Sun, adapted from Daniel Scharstein.
    """
    nBands = 2

    if v is None:
        assert(uv.ndim == 3)
        assert(uv.shape[2] == 2)
        u = uv[:,:,0]
        v = uv[:,:,1]
    else:
        u = uv

    assert(u.shape == v.shape)
    height,width = u.shape
    f = open(filename,'wb')
    # write the header
    f.write(TAG_CHAR)
    np.array(width).astype(np.int32).tofile(f)
    np.array(height).astype(np.int32).tofile(f)
    # arrange into matrix form
    tmp = np.zeros((height, width*nBands))
    tmp[:,np.arange(width)*2] = u
    tmp[:,np.arange(width)*2 + 1] = v
    tmp.astype(np.float32).tofile(f)
    f.close()


# ref: https://github.com/sampepose/flownet2-tf/
# blob/18f87081db44939414fc4a48834f9e0da3e69f4c/src/flowlib.py#L240
def visulize_flow_file(flow_filename, save_dir=None):
	flow_data = readFlow(flow_filename)
	img = flow2img(flow_data)
	# plt.imshow(img)
	# plt.show()
	if save_dir:
		idx = flow_filename.rfind("/") + 1
		plt.imsave(os.path.join(save_dir, "%s-vis.png" % flow_filename[idx:-4]), img)


def flow2img(flow_data):
	"""
	convert optical flow into color image
	:param flow_data:
	:return: color image
	"""
	# print(flow_data.shape)
	# print(type(flow_data))
	u = flow_data[:, :, 0]
	v = flow_data[:, :, 1]

	UNKNOW_FLOW_THRESHOLD = 1e7
	pr1 = abs(u) > UNKNOW_FLOW_THRESHOLD
	pr2 = abs(v) > UNKNOW_FLOW_THRESHOLD
	idx_unknown = (pr1 | pr2)
	u[idx_unknown] = v[idx_unknown] = 0

	# get max value in each direction
	maxu = -999.
	maxv = -999.
	minu = 999.
	minv = 999.
	maxu = max(maxu, np.max(u))
	maxv = max(maxv, np.max(v))
	minu = min(minu, np.min(u))
	minv = min(minv, np.min(v))

	rad = np.sqrt(u ** 2 + v ** 2)
	maxrad = max(-1, np.max(rad))
	u = u / maxrad + np.finfo(float).eps
	v = v / maxrad + np.finfo(float).eps

	img = compute_color(u, v)

	idx = np.repeat(idx_unknown[:, :, np.newaxis], 3, axis=2)
	img[idx] = 0

	return np.uint8(img)


def compute_color(u, v):
	"""
	compute optical flow color map
	:param u: horizontal optical flow
	:param v: vertical optical flow
	:return:
	"""

	height, width = u.shape
	img = np.zeros((height, width, 3))

	NAN_idx = np.isnan(u) | np.isnan(v)
	u[NAN_idx] = v[NAN_idx] = 0

	colorwheel = make_color_wheel()
	ncols = np.size(colorwheel, 0)

	rad = np.sqrt(u ** 2 + v ** 2)

	a = np.arctan2(-v, -u) / np.pi

	fk = (a + 1) / 2 * (ncols - 1) + 1

	k0 = np.floor(fk).astype(int)

	k1 = k0 + 1
	k1[k1 == ncols + 1] = 1
	f = fk - k0

	for i in range(0, np.size(colorwheel, 1)):
		tmp = colorwheel[:, i]
		col0 = tmp[k0 - 1] / 255
		col1 = tmp[k1 - 1] / 255
		col = (1 - f) * col0 + f * col1

		idx = rad <= 1
		col[idx] = 1 - rad[idx] * (1 - col[idx])
		notidx = np.logical_not(idx)

		col[notidx] *= 0.75
		img[:, :, i] = np.uint8(np.floor(255 * col * (1 - NAN_idx)))

	return img


def make_color_wheel():
	"""
	Generate color wheel according Middlebury color code
	:return: Color wheel
	"""
	RY = 15
	YG = 6
	GC = 4
	CB = 11
	BM = 13
	MR = 6

	ncols = RY + YG + GC + CB + BM + MR

	colorwheel = np.zeros([ncols, 3])

	col = 0

	# RY
	colorwheel[0:RY, 0] = 255
	colorwheel[0:RY, 1] = np.transpose(np.floor(255 * np.arange(0, RY) / RY))
	col += RY

	# YG
	colorwheel[col:col + YG, 0] = 255 - np.transpose(np.floor(255 * np.arange(0, YG) / YG))
	colorwheel[col:col + YG, 1] = 255
	col += YG

	# GC
	colorwheel[col:col + GC, 1] = 255
	colorwheel[col:col + GC, 2] = np.transpose(np.floor(255 * np.arange(0, GC) / GC))
	col += GC

	# CB
	colorwheel[col:col + CB, 1] = 255 - np.transpose(np.floor(255 * np.arange(0, CB) / CB))
	colorwheel[col:col + CB, 2] = 255
	col += CB

	# BM
	colorwheel[col:col + BM, 2] = 255
	colorwheel[col:col + BM, 0] = np.transpose(np.floor(255 * np.arange(0, BM) / BM))
	col += + BM

	# MR
	colorwheel[col:col + MR, 2] = 255 - np.transpose(np.floor(255 * np.arange(0, MR) / MR))
	colorwheel[col:col + MR, 0] = 255

	return colorwheel



