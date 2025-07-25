# cloudwall.py
import cv2
import numpy as np

def low_cloud_ratio(img, roi, gray_th):
    """
    img: BGR image
    roi: [x, y, w, h] 在整幅图上的矩形区域
    gray_th: 灰度阈值（根据实际配色调整 > 或 <）
    """
    x, y, w, h = roi
    crop = img[y:y+h, x:x+w]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    # 注意：Himawari Band13 常见配色是高云亮，低云暗 => 低云灰度较低
    mask = gray < gray_th
    return float(mask.mean())

def trend_alert(ratios, warn_ratio=0.12):
    """
    ratios: 最近N帧低云占比
    返回：0=无墙, 1=关注, 2=预警
    """
    if len(ratios) < 2:
        return 0
    incr = ratios[-1] - ratios[0]
    if ratios[-1] >= warn_ratio and incr > 0.03:
        return 2
    if ratios[-1] >= warn_ratio:
        return 1
    return 0
