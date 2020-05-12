#!/usr/bin/env python

import numpy as np
import cv2

def showMask(img_obj)
    img_ori = img.copy()
    gtmasks = img_obj['gtmasks']
    n = len(gtmasks)
    print(img.shape)
    for i, mobj in enumerate(gtmasks):
        if not (type(mobj['mask']) is list):
            print("Pass a RLE mask")
            continue
        else:
            pts = np.round(np.asarray(mobj['mask'][0]))
            pts = pts.reshape(pts.shape[0] // 2, 2)
            pts = np.int32(pts)
            color = np.uint8(np.random.rand(3) * 255).tolist()
            cv2.fillPoly(img, [pts], color)
    cv2.addWeighted(img, 0.5, img_ori, 0.5, 0, img)
    cv2.imshow("Mask", img)
    cv2.waitKey(0) 


img = cv2.imread(img_obj['../../sample_1.jpg'])