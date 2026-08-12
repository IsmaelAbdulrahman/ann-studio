# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 14: Modern CNN architectures & computer vision
# Section: Exercises
# Code example 5 of 5 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "vision"
# ====================================================================

import numpy as np
def iou(a, b):                       # boxes as (x1, y1, x2, y2)
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / union
def nms(boxes, scores, thr=0.5):
    order = list(np.argsort(scores)[::-1])       # indices, highest score first
    keep = []
    while order:
        i = int(order.pop(0)); keep.append(i)    # keep the top box ...
        order = [j for j in order if iou(boxes[i], boxes[j]) <= thr]  # ... drop its overlaps
    return keep
boxes  = np.array([[10,10,60,60],[12,12,58,58],[70,70,110,110]], float)
scores = np.array([0.90, 0.80, 0.70])
print("IoU(box0, box1) =", round(iou(boxes[0], boxes[1]), 3))    # 0.846 (near-duplicate)
print("IoU(box0, box2) =", round(iou(boxes[0], boxes[2]), 3))    # 0.0  (disjoint)
print("NMS keeps:", nms(boxes, scores, thr=0.5))                 # [0, 2]
