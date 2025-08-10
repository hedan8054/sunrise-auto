# -*- coding: utf-8 -*-
import yaml, html
from typing import Dict, List, Any

def _cmp(val, expr):
    if isinstance(expr, list) and len(expr)==2:
        lo, hi = expr; return (val is not None) and (lo <= val <= hi)
    if isinstance(expr, str):
        expr = expr.strip()
        if expr.startswith(">="): return val is not None and val >= float(expr[2:])
        if expr.startswith("<="): return val is not None and val <= float(expr[2:])
        if expr.startswith(">"):  return val is not None and val >  float(expr[1:])
        if expr.startswith("<"):  return val is not None and val <  float(expr[1:])
    return False

def _ok_one(metric: Dict[str, float], cond: Dict[str, Any]) -> bool:
    if "AND" in cond:
        return all(_ok_one(metric, c) for c in cond["AND"])
    if "ANY" in cond:
        return any(_ok_one(metric, c) for c in cond["ANY"])
    if "NOT" in cond:
        return not _ok_one(metric, cond["NOT"])
    # leaf
    for k, expr in cond.items():
        if k not in metric: 
            return False
        if not _cmp(metric[k], expr):
            return False
    return True

def load_labels(path="scene_labels.yaml") -> List[Dict[str, Any]]:
    y = yaml.safe_load(open(path, "r", encoding="utf-8"))
    labels = y["labels"]
    # 统一加默认字段
    for l in labels:
        l.setdefault("score", 50)
        l.setdefault("min_duration_h", 1)
    # 高分在前
    labels.sort(key=lambda x: x["score"], reverse=True)
    return labels

def pick_labels(metric_hourly: List[Dict[str, float]], labels: List[Dict[str, Any]], topk=6):
    """
    metric_hourly: 若干个小时的数值（通常只传目标时刻附近1-2小时）
      每个元素需要字段：low, mid, high, cloud_base_m, vis_km, wind, precip, dp
    返回：按权重+命中时长排序的标签列表
    """
    hits = []
    for lab in labels:
        need = lab["min_duration_h"]
        ok_cnt = sum(1 for m in metric_hourly if _ok_one(m, lab))
        if ok_cnt >= need:
            hits.append((lab["key"], lab["score"], ok_cnt))
    hits.sort(key=lambda x: (x[1], x[2]), reverse=True)
    return [h[0] for h in hits[:topk]]

def to_xhs_caption(event_name: str, time_str: str, place: str, score5: float,
                   metrics_show: Dict[str, Any], tags: List[str]) -> str:
    """
    生成小红书风格文案（含数据+场景标签）
    metrics_show: 用于展示的最终值（和你报告里的相同单位）
    """
    head = f"{event_name}预报｜{place}｜{time_str}\n"
    badge = f"场景标签：#{' #'.join(tags)}\n"
    scoreline = f"体感评分：{score5:.1f}/5\n"
    facts = (
        f"低云：{metrics_show.get('low', 'NA')}%｜"
        f"中/高云：{metrics_show.get('mh', 'NA')}%｜"
        f"云底：{metrics_show.get('cloud_base_m', '未知')}m｜\n"
        f"能见度：{metrics_show.get('vis_km','NA')}km｜"
        f"风：{metrics_show.get('wind','NA')}m/s｜"
        f"降雨：{metrics_show.get('precip','NA')}mm/h｜"
        f"露点差：{metrics_show.get('dp','NA')}℃\n"
    )
    tips = "拍摄建议：提前踩点，预留低云遮挡的备用机位；注意风和设备防潮。"
    return head + badge + scoreline + facts + tips
