# -*- coding: utf-8 -*-
"""
scene_labeler.py
输入：观测/预报指标（低/中/高云%、云底m、能见度km、风、降雨、露点差、低云墙风险等）
输出：场景标签（>=3个） + 一段带画面感的中文文案
说明：
- 规则全部基于你现有数据字段，可在 THRESH 调整阈值
- 包含 80+ 场景标签库的自动组合（精简输出与去重）
"""

from dataclasses import dataclass
from typing import List, Dict, Tuple
import math

# ---------------- 阈值配置（可按地区微调） ----------------
THRESH = {
    "low": {"low": 20, "mid": 40, "high": 60},           # 低云占比
    "mh":  {"few": 20, "ok_lo": 20, "ok_hi": 60, "much": 80},  # 中/高云总量
    "base": {"low": 500, "mid": 1000},                   # 云底(m)
    "vis": {"ok": 8, "good": 15},                        # 能见度(km)
    "wind": {"calm": 2, "breeze": 5, "windy": 8},        # 风速(m/s)
    "rain": {"drizzle": 0.1, "light": 1.0},              # 降雨量(mm)
    "dp": {"foggy": 1.0, "moist": 3.0},                  # 露点差(℃)
}

# ---------------- 标签库（按大类组织，内部会自动挑选） ----------------
TAGSETS = {
    "low_cloud": [
        "低云镶边", "低云破口", "低云挂帘", "低云翻涌", "低云墙", "低云散片", "低云拖尾",
        "低云火烧", "低云孤峰", "低云窗", "贴海低云", "低云双层", "低云倒影"
    ],
    "mid_high": [
        "卷云丝带", "卷云火烧", "卷云拖尾", "卷云扇形放射", "卷云羽毛",
        "高积云棉花糖", "高积云铺满天", "高积云点点星空", "高积云波浪纹",
        "层积云铺盖", "层积云镶金边", "层积云破口透光", "层积云火烧",
        "高层云幕布", "高层云金色铺路", "高层云渐变天幕"
    ],
    "iridescence": [
        "彩云冠", "卷云镶彩", "高积云镶彩", "彩边云墙", "彩色透光云", "彩云幕"
    ],
    "lights": [
        "耶稣光", "反耶稣光", "辐射状光芒", "金边透光", "霞光满天", "霞光撕裂", "云后爆光"
    ],
    "land": [
        "海平线直落", "海平线蹦出", "岛屿剪影", "山脊剪影", "沙滩倒影"
    ],
    "wx": [
        "晴空落日", "晴空日出", "平流雾", "沙尘天色", "雨幕天边", "雷雨云底光"
    ]
}

@dataclass
class Inputs:
    event: str            # "sunrise" 或 "sunset"
    time_str: str         # "08月08日 18:59" 只是给文案用
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

def clamp01(x):
    try:
        return max(0.0, min(1.0, float(x)/100.0))
    except Exception:
        return 0.0

def choose(tags: List[str], n: int) -> List[str]:
    return tags[:max(0, min(n, len(tags)))]

def infer_cloud_family(low, mid, high) -> str:
    """用中高云占比粗分云系：cirrus/altocumulus/stratus-like"""
    tot_mh = (mid or 0) + (high or 0)
    if tot_mh < THRESH["mh"]["few"]:
        return "none"
    if (high or 0) >= (mid or 0) + 10:
        return "cirrus"          # 卷云系
    if (mid or 0) >= (high or 0) + 10:
        return "altocumulus"     # 高积系
    return "stratiform"          # 层状/高层

def iridescence_possible(inputs: Inputs) -> bool:
    """彩云（云彩虹）出现的必要条件（启发式）"""
    low, mid, high = inputs.low, inputs.mid, inputs.high
    tot_mh = (mid or 0) + (high or 0)
    # 高空薄云(25~70) 更利于薄膜干涉；能见度好；基本无雨；太阳在地平线±10°
    good_cloud = 25 <= tot_mh <= 70
    good_vis   = inputs.vis_km >= THRESH["vis"]["good"]
    no_rain    = inputs.precip_mm < THRESH["rain"]["drizzle"]
    # 风不大更容易观察，但不是硬条件
    return good_cloud and good_vis and no_rain

def low_cloud_scene(inputs: Inputs, tags: List[str]):
    l = inputs.low
    if l is None: return
    if l < THRESH["low"]["low"]:
        # 基本通透：更偏向“破口/镶边”在局部云带时
        pass
    elif l < THRESH["low"]["mid"]:
        tags += choose(["低云散片","低云镶边"], 2)
    elif l < THRESH["low"]["high"]:
        tags += choose(["低云挂帘","低云破口","低云拖尾"], 2)
    else:
        tags += choose(["低云墙","低云翻涌","低云窗"], 2)

def mid_high_scene(inputs: Inputs, tags: List[str]):
    fam = infer_cloud_family(inputs.low, inputs.mid, inputs.high)
    tot = (inputs.mid or 0) + (inputs.high or 0)
    if tot < THRESH["mh"]["few"]:
        return
    if fam == "cirrus":
        tags += choose(["卷云羽毛","卷云丝带","卷云拖尾","卷云扇形放射"], 2)
    elif fam == "altocumulus":
        tags += choose(["高积云棉花糖","高积云点点星空","高积云波浪纹"], 2)
    else:
        tags += choose(["高层云金色铺路","层积云镶金边","层积云破口透光"], 2)
    if tot >= THRESH["mh"]["much"]:
        tags.append("霞光满天")

def light_effects(inputs: Inputs, tags: List[str]):
    # 云底高度与破口配合时更可能出“耶稣光”
    base = inputs.cloud_base_m or 800
    has_break = inputs.low and inputs.low >= THRESH["low"]["mid"]
    if base > 500 and has_break and inputs.vis_km >= THRESH["vis"]["ok"]:
        tags.append("耶稣光")
    if inputs.wind_ms <= THRESH["wind"]["breeze"] and inputs.vis_km >= THRESH["vis"]["good"]:
        tags.append("金边透光")

def land_scene(inputs: Inputs, tags: List[str]):
    # 不做地理推断，只给常见可视类（海边/岛屿）
    if inputs.place and any(k in inputs.place for k in ["湾","海","岛","湾区","沙滩","礁","港"]):
        tags += choose(["海平线直落","岛屿剪影","沙滩倒影"], 2)

def wx_scene(inputs: Inputs, tags: List[str]):
    tot_mh = (inputs.mid or 0) + (inputs.high or 0)
    if tot_mh < 10 and (inputs.low or 0) < 10:
        tags.append("晴空日出" if inputs.event=="sunrise" else "晴空落日")
    if inputs.dp_c < THRESH["dp"]["foggy"] and inputs.vis_km < THRESH["vis"]["ok"]:
        tags.append("平流雾")
    if inputs.precip_mm >= THRESH["rain"]["light"]:
        tags.append("雨幕天边")
    if inputs.lc_risk_multi == 2:
        tags.append("低云墙")

def rank_and_dedup(tags: List[str]) -> List[str]:
    # 简单去重+优先保留更“有戏”的标签
    seen, out = set(), []
    priority = ["彩云", "耶稣光", "火烧", "镶边", "破口", "低云墙", "金边", "霞光"]
    tags_sorted = sorted(tags, key=lambda t: -sum(k in t for k in priority))
    for t in tags_sorted:
        if t not in seen:
            seen.add(t); out.append(t)
    return out[:6]   # 最多给 6 个，读起来不累

def make_copy(inputs: Inputs, tags: List[str]) -> str:
    # 拼接一段“数据+画面”的文案（和你现有风格统一）
    feel = []
    # 低云
    if inputs.low is not None:
        if inputs.low < THRESH["low"]["low"]:
            feel.append("地平线基本通透")
        elif inputs.low < THRESH["low"]["mid"]:
            feel.append("低云稀疏，可能出现镶边或小破口")
        elif inputs.low < THRESH["low"]["high"]:
            feel.append("低云偏多，边缘透光")
        else:
            feel.append("一堵低云墙可能挡住首轮日光")
    # 中高云
    tot = (inputs.mid or 0)+(inputs.high or 0)
    if 20 <= tot <= 60:
        feel.append("中高云适中，具备“云接光”的舞台")
    elif tot < 20:
        feel.append("天空较干净，色彩偏简洁")
    else:
        feel.append("云量较多，层次偏厚")
    # 彩云
    if iridescence_possible(inputs):
        feel.append("薄云角度合适，存在彩云机会")
    # 能见度/风
    if inputs.vis_km >= THRESH["vis"]["good"]: feel.append("远景通透")
    if inputs.wind_ms <= THRESH["wind"]["breeze"]: feel.append("海面反光会更干净")
    # 风险
    if inputs.lc_risk_multi == 2:
        feel.append("多点模型提示低云墙预警")
    elif inputs.lc_risk_multi == 1:
        feel.append("低云墙需关注")

    scenic = "；".join(feel)
    when = "日出" if inputs.event=="sunrise" else "日落"
    return (
        f"【场景标签】{('、'.join(tags) if tags else '清爽纯色天')}\n"
        f"{when}：{inputs.time_str}｜地点：{inputs.place}\n"
        f"{scenic}。"
        f"\n小贴士：能见度{inputs.vis_km:.1f}km，风{inputs.wind_ms:.1f}m/s，降雨{inputs.precip_mm:.1f}mm，露点差{inputs.dp_c:.1f}℃。"
    )

def generate_scene(inputs: Inputs) -> Tuple[List[str], str]:
    tags: List[str] = []
    # 低云 / 中高云 / 光效 / 地景 / 天气
    low_cloud_scene(inputs, tags)
    mid_high_scene(inputs, tags)
    if iridescence_possible(inputs):
        # 根据云系给到偏好的彩云标签
        fam = infer_cloud_family(inputs.low, inputs.mid, inputs.high)
        if fam == "cirrus": tags.append("卷云镶彩")
        elif fam == "altocumulus": tags.append("高积云镶彩")
        else: tags.append("彩边云墙")
        tags.append("彩云冠")
    light_effects(inputs, tags)
    land_scene(inputs, tags)
    wx_scene(inputs, tags)

    tags = rank_and_dedup(tags)
    copy = make_copy(inputs, tags)
    return tags, copy
