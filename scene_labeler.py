# -*- coding: utf-8 -*-
"""
scene_labeler.py
输入：观测/预报指标（低/中/高云%、云底m、能见度km、风、降雨、露点差、低云墙风险等）
输出：场景标签（<=6个） + 一段带画面感的中文文案
"""

from dataclasses import dataclass
from typing import List, Tuple

# ---------------- 阈值配置（可按地区微调） ----------------
THRESH = {
    "low": {"low": 20, "mid": 40, "high": 60},                # 低云占比
    "mh":  {"few": 20, "ok_lo": 20, "ok_hi": 60, "much": 80}, # 中/高云总量
    "base": {"low": 500, "mid": 1000},                        # 云底(m)
    "vis": {"ok": 8, "good": 15},                             # 能见度(km)
    "wind": {"calm": 2, "breeze": 5, "windy": 8},             # 风速(m/s)
    "rain": {"drizzle": 0.1, "light": 1.0},                   # 降雨量(mm)
    "dp": {"foggy": 1.0, "moist": 3.0},                       # 露点差(℃)
}

@dataclass
class Inputs:
    event: str            # "sunrise" 或 "sunset"
    time_str: str         # "08月08日 18:59"
    place: str
    lat: float
    lon: float
    # 核心要素（与你的 bot 对齐）
    low: float
    mid: float
    high: float
    cloud_base_m: float|None
    vis_km: float
    wind_ms: float
    precip_mm: float
    dp_c: float
    lc_risk_simple: int|None = None     # 0/1/2
    lc_risk_multi:  int|None = None     # 0/1/2

# ---------------- 内部工具 ----------------
def choose(tags: List[str], n: int) -> List[str]:
    return tags[:max(0, min(n, len(tags)))]

def infer_cloud_family(low, mid, high) -> str:
    """粗分云系：cirrus/altocumulus/stratiform/none"""
    tot_mh = (mid or 0) + (high or 0)
    if tot_mh < THRESH["mh"]["few"]:
        return "none"
    if (high or 0) >= (mid or 0) + 10:
        return "cirrus"          # 卷云系
    if (mid or 0) >= (high or 0) + 10:
        return "altocumulus"     # 高积系
    return "stratiform"          # 层状/高层

def iridescence_possible(inp: Inputs) -> bool:
    """彩云（云彩虹）启发式：中高云25~70%、能见度高、无降雨"""
    tot_mh = (inp.mid or 0) + (inp.high or 0)
    good_cloud = 25 <= tot_mh <= 70
    good_vis   = inp.vis_km >= THRESH["vis"]["good"]
    no_rain    = inp.precip_mm < THRESH["rain"]["drizzle"]
    return good_cloud and good_vis and no_rain

# ---------------- 场景推断 ----------------
def low_cloud_scene(inp: Inputs, tags: List[str]):
    l = inp.low
    if l is None:
        return
    if l < THRESH["low"]["low"]:
        # 基本通透；若局部有带状云会是“镶边/破口”，但不强加
        return
    if l < THRESH["low"]["mid"]:
        tags += choose(["低云散片", "低云镶边"], 2)
    elif l < THRESH["low"]["high"]:
        tags += choose(["低云挂帘", "低云破口", "低云拖尾"], 2)
    else:
        tags += choose(["低云墙", "低云翻涌", "低云窗"], 2)

def mid_high_scene(inp: Inputs, tags: List[str]):
    fam = infer_cloud_family(inp.low, inp.mid, inp.high)
    tot = (inp.mid or 0) + (inp.high or 0)
    if tot < THRESH["mh"]["few"]:
        return
    if fam == "cirrus":
        tags += choose(["卷云羽毛", "卷云丝带", "卷云拖尾", "卷云扇形放射"], 2)
    elif fam == "altocumulus":
        tags += choose(["高积云棉花糖", "高积云点点星空", "高积云波浪纹"], 2)
    else:
        tags += choose(["高层云金色铺路", "层积云镶金边", "层积云破口透光"], 2)
    if tot >= THRESH["mh"]["much"]:
        tags.append("霞光满天")

def light_effects(inp: Inputs, tags: List[str]):
    base = inp.cloud_base_m or 800
    has_break = (inp.low or 0) >= THRESH["low"]["mid"]
    if base > 500 and has_break and inp.vis_km >= THRESH["vis"]["ok"]:
        tags.append("耶稣光")
    if inp.wind_ms <= THRESH["wind"]["breeze"] and inp.vis_km >= THRESH["vis"]["good"]:
        tags.append("金边透光")

def land_scene(inp: Inputs, tags: List[str]):
    if inp.place and any(k in inp.place for k in ["湾", "海", "岛", "港", "滩", "礁"]):
        tags += choose(["海平线直落", "岛屿剪影", "沙滩倒影"], 2)

def wx_scene(inp: Inputs, tags: List[str]):
    tot_mh = (inp.mid or 0) + (inp.high or 0)
    if tot_mh < 10 and (inp.low or 0) < 10:
        tags.append("晴空日出" if inp.event == "sunrise" else "晴空落日")
    if inp.dp_c < THRESH["dp"]["foggy"] and inp.vis_km < THRESH["vis"]["ok"]:
        tags.append("平流雾")
    if inp.precip_mm >= THRESH["rain"]["light"]:
        tags.append("雨幕天边")
    if inp.lc_risk_multi == 2:
        tags.append("低云墙")

def rank_and_dedup(tags: List[str]) -> List[str]:
    seen, out = set(), []
    priority = ["彩云", "耶稣光", "火烧", "镶边", "破口", "低云墙", "金边", "霞光"]
    tags_sorted = sorted(tags, key=lambda t: -sum(k in t for k in priority))
    for t in tags_sorted:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out[:6]   # 最多 6 个标签

def make_copy(inp: Inputs, tags: List[str]) -> str:
    feel = []
    # 低云
    if inp.low is not None:
        if inp.low < THRESH["low"]["low"]:
            feel.append("地平线基本通透")
        elif inp.low < THRESH["low"]["mid"]:
            feel.append("低云稀疏，可能出现镶边或小破口")
        elif inp.low < THRESH["low"]["high"]:
            feel.append("低云偏多，边缘透光")
        else:
            feel.append("一堵低云墙可能挡住首轮日光")
    # 中高云
    tot = (inp.mid or 0) + (inp.high or 0)
    if 20 <= tot <= 60:
        feel.append("中高云适中，具备“云接光”的舞台")
    elif tot < 20:
        feel.append("天空较干净，色彩偏简洁")
    else:
        feel.append("云量较多，层次偏厚")
    # 彩云机会
    if iridescence_possible(inp):
        feel.append("薄云角度合适，存在彩云机会")
    # 能见度/风
    if inp.vis_km >= THRESH["vis"]["good"]:
        feel.append("远景通透")
    if inp.wind_ms <= THRESH["wind"]["breeze"]:
        feel.append("海面反光会更干净")
    # 风险提示
    if inp.lc_risk_multi == 2:
        feel.append("多点模型提示低云墙预警")
    elif inp.lc_risk_multi == 1:
        feel.append("低云墙需关注")

    scenic = "；".join(feel)
    when = "日出" if inp.event == "sunrise" else "日落"
    return (
        f"【场景标签】{('、'.join(tags) if tags else '清爽纯色天')}\n"
        f"{when}：{inp.time_str}｜地点：{inp.place}\n"
        f"{scenic}。"
        f"\n小贴士：能见度{inp.vis_km:.1f}km，风{inp.wind_ms:.1f}m/s，降雨{inp.precip_mm:.1f}mm，露点差{inp.dp_c:.1f}℃。"
    )

def generate_scene(inp: Inputs) -> Tuple[List[str], str]:
    tags: List[str] = []
    low_cloud_scene(inp, tags)
    mid_high_scene(inp, tags)
    if iridescence_possible(inp):
        fam = infer_cloud_family(inp.low, inp.mid, inp.high)
        if fam == "cirrus":
            tags.append("卷云镶彩")
        elif fam == "altocumulus":
            tags.append("高积云镶彩")
        else:
            tags.append("彩边云墙")
        tags.append("彩云冠")
    light_effects(inp, tags)
    land_scene(inp, tags)
    wx_scene(inp, tags)

    tags = rank_and_dedup(tags)
    copy = make_copy(inp, tags)
    return tags, copy
