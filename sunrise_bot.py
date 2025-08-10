#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sunrise_bot.py  —  无卫星版 + 扇区云量/碎片度/岸线风向修正

模式：
  python sunrise_bot.py forecast    # 下午预测（日出评分+文案）
  python sunrise_bot.py nightcheck  # 22:00 夜检（记录多点模型风险）
  python sunrise_bot.py lastcheck   # 03:30 凌晨检（记录多点模型风险）

要点：
- 仅用 open-meteo 数值 + METAR(可选)；无卫星依赖
- “日出方向扇区”抽样：bearing±{10°,20°} × {20,40,60 km}
- 碎片度（proxy）：同一扇区多点低云覆盖率的标准差（std%）
- 岸线风向修正：当风与岸线法向夹角小（上岸/离岸）时，适度减轻低云惩罚
- 低云项“天窗加分”：在 40–70% 且（碎片度高 或 扇区低云 < 全场低云×0.85 或 岸线风向有利）时，提高 1 分（上限 2 分）
- 额外加分：中/高云 35–60 且能见度>20km → +0.5；云底>800m → +0.5
- 多点“低云墙预警”：沿日出方位多距离抽样，给 0/1/2 级
"""

import os, sys, json, yaml, math, datetime as dt
import requests, pandas as pd, numpy as np
import pytz
import warnings, urllib3
warnings.filterwarnings("ignore", message="Unverified HTTPS request")
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ----------------- 全局配置 -----------------
TZ = pytz.timezone("Asia/Shanghai")
CONFIG = yaml.safe_load(open("config.yaml", "r", encoding="utf-8"))

LAT = CONFIG["location"]["lat"]
LON = CONFIG["location"]["lon"]
LOC_NAME = CONFIG["location"]["name"]

# 可选：岸线法向（度，指向海的一侧；用于判断上/离岸风），没有就用 90°
COAST_NORMAL_DEG = CONFIG.get("coast_normal_deg", 90.0)

# 云底经验系数（m/℃），也可写在 config.yaml 的 cloudbase_coef_m_per_C
CB_COEF = float(CONFIG.get("cloudbase_coef_m_per_C", 125.0))

# 扇区采样配置
SECTOR_OFFSETS_DEG = CONFIG.get("sector_offsets_deg", [-20, -10, 0, 10, 20])
SECTOR_DISTS_KM    = CONFIG.get("sector_dists_km", [20, 40, 60])

# ----------------- 基础工具 -----------------
def now():
    return dt.datetime.now(TZ)

def ensure_dirs():
    os.makedirs("logs", exist_ok=True)
    os.makedirs("out", exist_ok=True)

def log_csv(path_tpl: str, row: dict):
    ensure_dirs()
    path = now().strftime(path_tpl)
    header = not os.path.exists(path)
    pd.DataFrame([row]).to_csv(path, mode="a", header=header, index=False, encoding="utf-8-sig")

def save_report(name: str, text: str):
    ensure_dirs()
    fname = now().strftime(f"out/{name}_%Y-%m-%d_%H%M.txt")
    with open(fname, "w", encoding="utf-8") as f:
        f.write(text)

# ----------------- 天文时间 -----------------
def sunrise_time():
    """获取明日日出（精确时间、整点）"""
    try:
        js = requests.get(
            f"https://api.sunrise-sunset.org/json?lat={LAT}&lng={LON}&date=tomorrow&formatted=0",
            timeout=30
        ).json()
        t = dt.datetime.fromisoformat(js["results"]["sunrise"]).astimezone(TZ)
    except Exception:
        t = now().replace(hour=6, minute=0, second=0, microsecond=0) + dt.timedelta(days=1)
    return t, t.replace(minute=0, second=0, microsecond=0)

# ----------------- 气象数据 -----------------
def open_meteo(lat=None, lon=None):
    """open-meteo 每小时数据，增加 winddirection_10m 便于岸线修正"""
    lat = LAT if lat is None else lat
    lon = LON if lon is None else lon
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        "&hourly=cloudcover_low,cloudcover_mid,cloudcover_high,visibility,"
        "temperature_2m,dewpoint_2m,windspeed_10m,winddirection_10m,precipitation"
        "&forecast_days=2&timezone=Asia%2FShanghai"
    )
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        data = r.json()
        if "hourly" not in data or "time" not in data["hourly"]:
            print("[ERR] open-meteo hourly 字段缺失：", data)
            return None
        return data
    except Exception as e:
        print("[ERR] open-meteo 请求失败：", e)
        return None

def metar(code="ZGSZ"):
    """NOAA METAR 文本（失败返回空字符串）"""
    try:
        url = f"https://tgftp.nws.noaa.gov/data/observations/metar/stations/{code}.TXT"
        txt = requests.get(url, timeout=20).text.strip().splitlines()[-1]
        return txt
    except Exception as e:
        print("[WARN] METAR 获取失败：", e)
        return ""

def parse_cloud_base(metar_txt):
    """解析云底高度(m)，BKN/OVC 第一层"""
    import re
    m = re.findall(r'(BKN|OVC)(\d{3})', metar_txt or "")
    if not m:
        return None
    ft = int(m[0][1]) * 100
    return ft * 0.3048

# ----------------- 地理工具 -----------------
def offset_latlon(lat, lon, bearing_deg, dist_km):
    R = 6371.0
    brng = math.radians(bearing_deg)
    lat1 = math.radians(lat)
    lon1 = math.radians(lon)
    lat2 = math.asin(math.sin(lat1)*math.cos(dist_km/R) +
                     math.cos(lat1)*math.sin(dist_km/R)*math.cos(brng))
    lon2 = lon1 + math.atan2(math.sin(brng)*math.sin(dist_km/R)*math.cos(lat1),
                             math.cos(lat1)*math.cos(dist_km/R)-math.sin(lat1)*math.sin(lat2))
    return math.degrees(lat2), math.degrees(lon2)

def ang_diff(a, b):
    """两角度差的绝对值（0~180）"""
    d = abs((a - b + 180) % 360 - 180)
    return d

# ----------------- 扇区抽样与“天窗因子” -----------------
def pick_hour_idx(om, target_dt):
    hrs = om["hourly"]["time"]
    tgt = target_dt.strftime("%Y-%m-%dT%H:00")
    if tgt in hrs:
        return hrs.index(tgt), "exact_hour"
    # 取最近小时（防止整点缺失）
    idx = min(range(len(hrs)), key=lambda i: abs(dt.datetime.fromisoformat(hrs[i]) - target_dt))
    return idx, "nearest"

def sector_samples(target_hour, center_bearing_deg):
    """
    沿“日出方向扇区”抽样，返回样本列表：
    [(dist_km, bearing_deg, low%, base_m or None, src='om:exact_hour|nearest'), ...]
    """
    samples = []
    for d in SECTOR_DISTS_KM:
        for off in SECTOR_OFFSETS_DEG:
            plat, plon = offset_latlon(LAT, LON, center_bearing_deg + off, d)
            om = open_meteo(plat, plon)
            if om is None:
                samples.append((d, center_bearing_deg+off, None, None, "om:fail"))
                continue
            idx, tag = pick_hour_idx(om, target_hour)
            low = om["hourly"]["cloudcover_low"][idx]
            base_m = None  # open-meteo 无云底字段，此处留空（用站点估算/或 METAR）
            samples.append((d, center_bearing_deg+off, low, base_m, f"om:{tag}"))
    return samples

def sector_metrics(samples):
    """从样本提炼指标：扇区低云均值、碎片度(std)、有效样本数"""
    lows = [x[2] for x in samples if isinstance(x[2], (int, float))]
    if not lows:
        return None, None, 0
    return float(np.mean(lows)), float(np.std(lows)), len(lows)

def shoreline_wind_bonus(wind_dir_deg, coast_normal_deg=COAST_NORMAL_DEG):
    """
    风向与岸线法向夹角小（<=30°）→ 上/离岸明显，返回 True
    需要 winddirection_10m（气象学风向：来自方向，0/360=北）
    """
    if wind_dir_deg is None:
        return False
    return ang_diff(wind_dir_deg, coast_normal_deg) <= 30.0

# ----------------- 风向获取 -----------------
def wind_dir_at(om, target_hour):
    try:
        idx, _ = pick_hour_idx(om, target_hour)
        wd = om["hourly"].get("winddirection_10m", [None]*len(om["hourly"]["time"]))[idx]
        return wd
    except Exception:
        return None

# ----------------- “低云墙预警”（多点规则） -----------------
def cloudwall_risk_from_samples(samples):
    """
    规则：
      - 任意样本 low>=50% → 计为“厚低云段”；若此类样本比例>=50% → 预警(2)
      - 若存在样本 low>=30% → 关注(1)
      - 否则 正常(0)
    """
    if not samples:
        return 1, "关注(模型)（samples: none）"
    lows = [l for _,_,l,_,_ in samples if isinstance(l, (int,float))]
    if not lows:
        return 1, "关注(模型)（samples: no_valid_low）"
    hi = sum(1 for l in lows if l >= 50)
    mid = sum(1 for l in lows if l >= 30)
    if hi >= len(lows)*0.5:
        score = 2; tag = "预警(模型)"
    elif mid >= 1:
        score = 1; tag = "关注(模型)"
    else:
        score = 0; tag = "正常(模型)"
    detail = " | ".join([f"{int(d)}km:{(str(l)+'%') if l is not None else 'NA%'} [{src}]"
                         for d,_,l,_,src in samples])
    return score, f"{tag}（samples: {detail}）"

# ----------------- 评分逻辑 -----------------
def score_value(v, bounds):
    if v is None:
        return 1
    if len(bounds) == 3:
        lo, mid, hi = bounds
        if lo <= v <= mid:
            return 2
        if (mid < v <= hi) or (0 <= v < lo):
            return 1
        return 0
    else:
        lo, hi = bounds
        if v >= hi:
            return 2
        if v >= lo:
            return 1
        return 0

def calc_score(vals, cloud_base_m, cfg):
    detail, total = [], 0

    # 低云（基础分）
    lc = vals["low"]
    pt_low = score_value(lc, cfg["low_cloud"])
    detail.append(("低云%", lc, pt_low)); total += pt_low

    # 中/高云
    mh = max(vals["mid"], vals["high"])
    pt_mh = score_value(mh, cfg["mid_high_cloud"])
    detail.append(("中/高云%", mh, pt_mh)); total += pt_mh

    # 云底
    if cloud_base_m is None:
        pt_cb, val_cb = 1, -1
    else:
        lo, hi = cfg["cloud_base_m"]
        pt_cb = 2 if cloud_base_m > hi else 1 if cloud_base_m > lo else 0
        val_cb = cloud_base_m
    detail.append(("云底高度m", val_cb, pt_cb)); total += pt_cb

    # 能见度
    vis_km = (vals["vis"] or 0) / 1000.0
    lo, hi = cfg["visibility_km"]
    pt_vis = 2 if vis_km >= hi else 1 if vis_km >= lo else 0
    detail.append(("能见度km", vis_km, pt_vis)); total += pt_vis

    # 风速
    w = vals["wind"]
    lo1, lo2, hi2, hi3 = cfg["wind_ms"]
    if lo2 <= w <= hi2:
        pt_w = 2
    elif lo1 <= w < lo2 or hi2 < w <= hi3:
        pt_w = 1
    else:
        pt_w = 0
    detail.append(("风速m/s", w, pt_w)); total += pt_w

    # 露点差
    dp = vals["t"] - vals["td"]
    lo, hi = cfg["dewpoint_diff"]
    pt_dp = 2 if dp >= hi else 1 if dp >= lo else 0
    detail.append(("露点差°C", dp, pt_dp)); total += pt_dp

    # 降雨量
    p = vals.get("precip", 0)
    lo, hi = cfg["precip_mm"]
    pt_p = 2 if p < lo else 1 if p < hi else 0
    detail.append(("降雨量mm", p, pt_p)); total += pt_p

    return total, detail

# —— “天窗加分”与额外加分（低改动接入）——
def apply_window_bonuses(total, det, *,
                         sector_low_avg, sector_std, global_low,
                         vis_km, mid_high, cloud_base_m,
                         wind_dir_deg, shore_bonus_flag):
    """
    规则汇总：
      A) 低云 40–70% 且 [碎片度高(>=12) 或 扇区低云 < 全场低云×0.85 或 岸线风向有利] → 低云项 +1 分（封顶2）
      B) 中/高云 35–60 且 能见度>20 → 总分 +0.5
      C) 云底 > 800m → 总分 +0.5
    """
    # 从 det 拿到低云项，做替换
    idx_low = next(i for i,(n,_,_) in enumerate(det) if n=="低云%")
    low_name, low_val, low_pt = det[idx_low]

    reason = []

    if (low_val is not None and 40 <= low_val <= 70):
        cond_frag = (sector_std is not None and sector_std >= 12.0)
        cond_sector = (sector_low_avg is not None and global_low is not None and sector_low_avg < global_low*0.85)
        cond_shore = bool(shore_bonus_flag)
        if cond_frag or cond_sector or cond_shore:
            new_low_pt = min(2, low_pt + 1)
            if new_low_pt != low_pt:
                reason.append("低云40–70%且(碎片度高/扇区更低/岸线风向有利)")
                total += (new_low_pt - low_pt)
                det[idx_low] = (low_name, low_val, new_low_pt)

    # B) 染色潜力加分
    if (mid_high is not None and 35 <= mid_high <= 60) and (vis_km is not None and vis_km > 20):
        total += 0.5
        reason.append("中/高云35–60且能见度>20km(+0.5)")

    # C) 云底较高加分
    if cloud_base_m is not None and cloud_base_m > 800:
        total += 0.5
        reason.append("云底>800m(+0.5)")

    return total, reason

# ----------------- 文案 -----------------
def build_forecast_text(total, det, sun_t, extra):
    lines = [
        f"【日出预报 | 明早 {sun_t:%m月%d日}】拍摄指数：{total}/18",
        f"地点：{LOC_NAME}  (lat={LAT}, lon={LON})",
        f"日出：{sun_t:%H:%M}",
        ""
    ]
    for name, val, pts in det:
        if isinstance(val, float):
            lines.append(f"- {name}: {val:.1f} → {pts}分")
        else:
            lines.append(f"- {name}: {val} → {pts}分")
    if extra.get("note"):
        lines.append("\n提示：" + extra["note"])
    return "\n".join(lines)

def gen_scene_desc(score5, kv, sun_t):
    lc   = kv.get("低云%",      0) or 0
    mh   = kv.get("中/高云%",    0) or 0
    cb   = kv.get("云底高度m",   -1)
    vis  = kv.get("能见度km",    0) or 0
    wind = kv.get("风速m/s",     0) or 0
    dp   = kv.get("露点差°C",    0) or 0
    rp   = kv.get("降雨量mm",    0) or 0

    if lc < 20:
        lc_level = "低";      low_text = "地平线基本通透，太阳能“蹦”出来"
    elif lc < 40:
        lc_level = "中";      low_text = "地平线可能有一条灰带，太阳或从缝隙钻出"
    elif lc < 60:
        lc_level = "偏高";    low_text = "低云偏多，首轮日光可能被挡一部分"
    else:
        lc_level = "高";      low_text = "一堵低云墙，首轮日光大概率看不到"

    if 20 <= mh <= 60:
        mh_level = "适中";    fire_text = "有“云接光”舞台，可能染上粉橙色（小概率火烧云）"
    elif mh < 20:
        mh_level = "太少";    fire_text = "天空太干净，只有简单渐变色"
    elif mh <= 80:
        mh_level = "偏多";    fire_text = "云多且厚，色彩可能偏闷"
    else:
        mh_level = "很多";    fire_text = "厚云盖顶，大概率阴沉"

    if cb is None or cb < 0:
        cb_level, cb_text, cb_show = "未知", "云底数据缺失，可参考凌晨“低云墙预警”", "未知"
    elif cb > 1000:
        cb_level, cb_text, cb_show = ">1000m", "云底较高，多当“天花板”，不挡海平线", f"{cb:.0f}m"
    elif cb > 500:
        cb_level, cb_text, cb_show = "500~1000m", "可能在海面上方形成一道云棚，注意日出角度", f"{cb:.0f}m"
    else:
        cb_level, cb_text, cb_show = "<500m", "贴海低云/雾，像拉了窗帘", f"{cb:.0f}m"

    if vis >= 15:
        vis_level, vis_text = ">15km", "空气透明度好，远景清晰，金光反射漂亮"
    elif vis >= 8:
        vis_level, vis_text = "8~15km", "中等透明度，远景略灰"
    else:
        vis_level, vis_text = "<8km", "背景灰蒙蒙，层次差"

    if 2 <= wind <= 5:
        wind_level, wind_text = "2~5m/s", "海面有微波纹，反光好，三脚架稳"
    elif wind < 2:
        wind_level, wind_text = "<2m/s", "几乎无风，注意镜头易结露"
    elif wind <= 8:
        wind_level, wind_text = "5~8m/s", "风稍大，留意三脚架稳定性"
    else:
        wind_level, wind_text = ">8m/s", "大风天，拍摄体验差，器材要护好"

    if dp >= 3:
        dp_level, dp_text = "≥3℃", "不易起雾"
    elif dp >= 1:
        dp_level, dp_text = "1~3℃", "稍潮，镜头可能结露"
    else:
        dp_level, dp_text = "<1℃", "极易起雾，注意海雾/镜头起雾风险"

    if rp < 0.1:
        rp_level, rain_text = "<0.1mm", "几乎不会下雨"
    elif rp < 1:
        rp_level, rain_text = "0.1~1mm", "可能有零星小雨/毛毛雨"
    else:
        rp_level, rain_text = "≥1mm", "有下雨可能，注意防水和收纳镜头"

    if score5 >= 4.0:   grade = "建议出发（把握较大）"
    elif score5 >= 3.0: grade = "可去一搏（不稳）"
    elif score5 >= 2.0: grade = "机会一般（看心情或距离）"
    elif score5 >= 1.0: grade = "概率很小（除非就在附近）"
    else:               grade = "建议休息（基本无戏）"

    return (
        f"【直观判断】评分：{score5:.1f}/5 —— {grade}\n"
        f"日出：{sun_t:%H:%M}\n"
        f"- 低云：{lc:.0f}%（{lc_level}）— {low_text}\n"
        f"- 中/高云：{mh:.0f}%（{mh_level}）— {fire_text}\n"
        f"- 云底高度：{cb_show}（{cb_level}）— {cb_text}\n"
        f"- 能见度：{vis:.1f} km（{vis_level}）— {vis_text}\n"
        f"- 风速：{wind:.1f} m/s（{wind_level}）— {wind_text}\n"
        f"- 降雨：{rp:.1f} mm（{rp_level}）— {rain_text}\n"
        f"- 露点差：{dp:.1f} ℃（{dp_level}）— {dp_text}"
    )

# ----------------- 主流程 -----------------
def run_forecast():
    sun_exact, sun_hour = sunrise_time()

    # 1) open-meteo 主数据（本地点）
    om = open_meteo()
    if om is None:
        msg = "[ERR] open-meteo 数据为空，无法评分。"
        print(msg); save_report("forecast_error", msg)
        log_csv(CONFIG["paths"]["log_scores"], {"time": now(), "mode":"forecast", "score":-1, "error":"open-meteo no data"})
        return

    idx, idx_tag = pick_hour_idx(om, sun_hour)
    vals = dict(
        low    = om["hourly"]["cloudcover_low"][idx],
        mid    = om["hourly"]["cloudcover_mid"][idx],
        high   = om["hourly"]["cloudcover_high"][idx],
        vis    = om["hourly"]["visibility"][idx],
        t      = om["hourly"]["temperature_2m"][idx],
        td     = om["hourly"]["dewpoint_2m"][idx],
        wind   = om["hourly"]["windspeed_10m"][idx],
        winddir= om["hourly"].get("winddirection_10m", [None]*len(om["hourly"]["time"]))[idx],
        precip = om["hourly"]["precipitation"][idx]
    )

    # 云底：优先 METAR，否则用经验估算
    cb_metar = parse_cloud_base(metar("ZGSZ"))
    cb_est = max(0.0, (vals["t"] - vals["td"])) * CB_COEF if (vals["t"] is not None and vals["td"] is not None) else None
    cloud_base_used = cb_metar if cb_metar is not None else cb_est

    # 2) 扇区抽样（按日出方位）
    center_bearing = float(CONFIG.get("cloudwall", {}).get("sunrise_azimuth", 90.0))
    samples = sector_samples(sun_hour, center_bearing)
    sector_low_avg, sector_std, n_ok = sector_metrics(samples)

    # 3) 多点低云墙风险
    risk_multi, risk_multi_text = cloudwall_risk_from_samples(samples)

    # 4) 初始评分
    total, det = calc_score(vals, cloud_base_used, CONFIG["scoring"])

    # 5) “天窗加分”与其它加分
    mid_high = max(vals["mid"], vals["high"])
    total2, reasons = apply_window_bonuses(
        total, det,
        sector_low_avg=sector_low_avg, sector_std=sector_std, global_low=vals["low"],
        vis_km=(vals["vis"] or 0)/1000.0 if vals["vis"] is not None else None,
        mid_high=mid_high, cloud_base_m=cloud_base_used,
        wind_dir_deg=vals["winddir"],
        shore_bonus_flag=shoreline_wind_bonus(vals["winddir"])
    )
    total = total2

    # 6) 5分制 + 场景文字
    score5 = round(total / (3 * len(det)) * 5, 1)
    kv = {k: v for k, v, _ in det}
    scene_txt = (
        gen_scene_desc(score5, kv, sun_exact)
        + f"\n- 低云墙预警（模型多点）：{risk_multi_text}"
        + f"\n- 扇区低云均值/碎片度/样本：{sector_low_avg if sector_low_avg is not None else 'NA'}% / "
          f"{sector_std if sector_std is not None else 'NA'} / {n_ok}"
        + f"\n- 加分原因：{', '.join(reasons) if reasons else '无'}"
        + f"\n- 云底估算参数：coef={CB_COEF:.1f} m/℃；METAR云底="
          f"{f'{cb_metar:.0f}m' if cb_metar is not None else 'NA'}，估算云底="
          f"{f'{cb_est:.0f}m' if cb_est is not None else 'NA'}（使用="
          f"{'METAR' if cb_metar is not None else '估算'}）"
    )

    text = scene_txt + "\n\n" + build_forecast_text(total, det, sun_exact, extra={})
    print(text)
    save_report("forecast", text)
# ---------- 新的输出层（粘贴到工具函数区） ----------
def _scene_tags(lc, mh, cb, vis_km, wind, dp, rain, tianchuang=None, event_name="日出"):
    tags = []
    # 低云 & 天窗
    if lc >= 65:
        tags.append("低云墙可能")
    elif lc >= 40:
        tags.append("低云偏多")
    else:
        tags.append("地平线通透")

    if tianchuang is not None:
        if tianchuang >= 0.6:
            tags.append("天窗概率高")
        elif tianchuang >= 0.35:
            tags.append("天窗概率中")
        else:
            tags.append("天窗概率低")

    # 中高云舞台
    if 35 <= mh <= 60:
        tags.append("云接光舞台")
    elif mh < 20:
        tags.append("天空干净")
    else:
        tags.append("厚云盖顶")

    # 云底
    if cb is None or cb < 0:
        pass
    elif cb < 500:
        tags.append("贴海低云")
    elif cb < 1000:
        tags.append("云棚可能")
    else:
        tags.append("高云底")

    # 观感辅助
    if vis_km >= 15:
        tags.append("通透度好")
    if 2 <= wind <= 5:
        tags.append("海面微波反光")
    if dp < 1:
        tags.append("易起雾")
    if rain >= 1:
        tags.append("降雨风险")
    return list(dict.fromkeys(tags))[:6]  # 去重、限长


def render_story_block(event_name, sun_t, score5, tags, one_liner, tips):
    lines = []
    lines.append(f"【{event_name}直观判断】{score5:.1f}/5｜" + "、".join(tags))
    lines.append(f"{event_name}：{sun_t:%m月%d日 %H:%M}")
    if one_liner:
        lines.append(one_liner)
    if tips:
        lines.append("小贴士：" + tips)
    return "\n".join(lines)


def render_data_block(place, lat, lon, total18, detail_rows, extras=None, event_name="日出", sun_t=None):
    lines = []
    lines.append(f"\n拍摄指数：{total18}/18")
    lines.append(f"地点：{place}  (lat={lat:.5f}, lon={lon:.5f})")
    if sun_t:
        lines.append(f"{event_name}：{sun_t:%H:%M}")
    for name, val, pts in detail_rows:
        if isinstance(val, float):
            lines.append(f"- {name}: {val:.1f} → {pts}分")
        else:
            lines.append(f"- {name}: {val} → {pts}分")
    if extras and extras.get("notes"):
        lines.append("\n备注：" + extras["notes"])
    return "\n".join(lines)


def compose_output(event_name, sun_t, score5, total18, det, vals, extra):
    """
    det: 你的 calc_score(...) 返回的 detail 列表
    vals: 你用于评分的原始数值 dict（low/mid/high/vis/t/td/wind/precip）
    extra: 你现在已有的扩展字典，可放 扇区低云均值/碎片度 等
    """
    # 取关键指标
    lc   = vals.get("low", 0) or 0
    mh   = max(vals.get("mid", 0) or 0, vals.get("high", 0) or 0)
    visk = (vals.get("vis", 0) or 0) / 1000.0
    wind = vals.get("wind", 0) or 0
    dp   = (vals.get("t", 0) or 0) - (vals.get("td", 0) or 0)
    rain = vals.get("precip", 0) or 0

    # 云底从 det 里拿（没有则 None）
    cb = None
    for k, v, _ in det:
        if k == "云底高度m":
            cb = None if (isinstance(v, (int,float)) and v < 0) else v
            break

    # 一个很轻量的“天窗因子”（只靠你现有扩展字段，不引入新的数据依赖）
    # 你如果已算过 fan_low_mean / fan_frag / shoreline_wind_bonus，就用它们；
    # 没有就默认 None，逻辑自动忽略
    frag = extra.get("fan_frag")  # 0~1
    fan_low = extra.get("fan_low_mean")  # 0~100
    wind_bonus = extra.get("shoreline_wind_bonus", 0.0)  # -0.2~+0.3 类似
    tianchuang = None
    if frag is not None and fan_low is not None:
        # 低云越少、碎片度越高 → 越容易出缝
        tianchuang = max(0.0, min(1.0, 0.6*frag + 0.4*(1 - fan_low/100.0) + wind_bonus))

    # 场景标签 + 一句话
    tags = _scene_tags(lc, mh, cb, visk, wind, dp, rain, tianchuang, event_name)
    if lc < 30:
        one = "地平线通透，太阳大概率“蹦”出；色彩取决于中高云是否接光。"
    elif lc < 60:
        one = "低云有量，但存在开缝机会；盯住太阳方位的亮缝，等穿缝瞬间。"
    else:
        one = "低云偏厚，首轮日光可能被挡，留意侧向或后程开窗。"

    if mh >= 35 and mh <= 60 and visk >= 20:
        one += " 中高云条件利于“云接光”，晚霞/晨霞层次感可期。"
    if tianchuang is not None:
        one += f"（天窗指示：{'高' if tianchuang>=0.6 else '中' if tianchuang>=0.35 else '低'}）"

    tips = []
    if 2 <= wind <= 5:
        tips.append("海面有微波反光，试 1/125~1/250 固定水纹")
    if dp < 1:
        tips.append("露点差低，镜头易起雾，常擦拭")
    if cb and cb < 600:
        tips.append("云底贴海，低机位/反射水洼更出片")
    tips = "；".join(tips)

    story = render_story_block(event_name, sun_t, score5, tags, one, tips)
    data  = render_data_block(
        CONFIG["location"]["name"], LAT, LON, total18, det,
        extras={"notes": extra.get("note")}, event_name=event_name, sun_t=sun_t
    )
    return story + "\n" + data
    # 日志
    log_csv(CONFIG["paths"]["log_scores"], {
        "time": now(), "mode": "forecast", "score": total, "score5": score5,
        **{k: v for k, v, _ in det},
        "sector_low_avg": sector_low_avg, "sector_std": sector_std, "sector_n": n_ok,
        "risk_multi": risk_multi, "idx_tag": idx_tag
    })

def run_check(mode: str):
    """夜检/凌晨检：仅记多点模型风险"""
    sun_exact, sun_hour = sunrise_time()
    center_bearing = float(CONFIG.get("cloudwall", {}).get("sunrise_azimuth", 90.0))
    samples = sector_samples(sun_hour, center_bearing)
    risk, txt = cloudwall_risk_from_samples(samples)
    msg = f"{mode}: risk={risk}, {txt}"
    print(msg); save_report(mode, msg)
    log_csv(CONFIG["paths"]["log_cloud"], {
        "time": now(), "mode": mode, "cloudwall_score": risk, "text": txt
    })

# ----------------- 主入口 -----------------
if __name__ == "__main__":
    ensure_dirs()
    mode = sys.argv[1] if len(sys.argv) > 1 else "forecast"
    try:
        if mode == "forecast":
            run_forecast()
        elif mode == "nightcheck":
            run_check("nightcheck")
        elif mode == "lastcheck":
            run_check("lastcheck")
        else:
            print("Usage: python sunrise_bot.py [forecast|nightcheck|lastcheck]")
    except Exception as e:
        err_msg = f"[FATAL] {mode} 运行异常：{repr(e)}"
        print(err_msg); save_report(f"{mode}_error", err_msg)
        log_csv(CONFIG["paths"]["log_scores"], {
            "time": now(), "mode": mode, "score": -1, "error": repr(e)
        })
