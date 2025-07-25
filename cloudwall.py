# cloudwall.py
import cv2
import numpy as np

def low_cloud_ratio(img, roi, gray_th):
    """
    img: BGR image (Himawari PNG)
    roi: [x, y, w, h] 在整幅图上的矩形区域
    gray_th: 灰度阈值；Band13 常见配色高云亮、低云暗 => 低云灰度较低，故用 < gray_th
    """
    x, y, w, h = roi
    crop = img[y:y+h, x:x+w]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    mask = gray < gray_th
    return float(mask.mean())

def trend_alert(ratios, warn_ratio=0.12):
    """
    ratios: 最近N帧低云占比
    返回 0 / 1 / 2
      0: 正常
      1: 关注（占比超过阈值但增幅不大）
      2: 预警（占比高且增长明显）
    """
    if len(ratios) < 2:
        return 0
    incr = ratios[-1] - ratios[0]
    if ratios[-1] >= warn_ratio and incr > 0.03:
        return 2
    if ratios[-1] >= warn_ratio:
        return 1
    return 0
