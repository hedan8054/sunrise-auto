#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sunrise_bot.py （无卫星版）
模式：
  python sunrise_bot.py forecast    # 下午预测（日出评分+文案）
  python sunrise_bot.py nightcheck  # 22:00 夜检（记录低云墙指数-模型）
  python sunrise_bot.py lastcheck   # 03:30 凌晨检（记录低云墙指数-模型）

改动要点：
- 完全移除 Himawari/卫星帧/OpenCV/cloudwall 依赖
- “低云墙预警”用 meteoblue（有 key）或 open-meteo 的多点采样模型替代
- 其他评分逻辑保持不变
"""

import os
import sys
import json
import yaml
import math
import datetime as dt

import requests
import pandas as pd
import numpy as np
import pytz
import warnings
import urllib3
warnings.filterwarnings("ignore", message="Unverified HTTPS request")
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ----------------- 全局配置 -----------------
TZ = pytz.timezone("Asia/Shanghai")
CONFIG = yaml.safe_load(open("config.yaml", "r", encoding="utf-8"))

LAT = CONFIG["location"]["lat"]
LON = CONFIG["location"]["lon"]

# 可选：meteoblue API key（放 GitHub Secrets 里）
MB_API_KEY = os.getenv("MB_API_KEY", "").strip()


# ----------------- 基础工具 -----------------
def now():
    return dt.datetime.now(TZ)

def ensure_dirs():
    os.makedirs("logs", exist_ok=True)
    os.makedirs("out", exist_ok=True)

def log_csv(path_tpl: str, row: dict):
    """按月份滚动写 CSV"""
    ensure_dirs()
    path = now().strftime(path_tpl)
    header = not os.path.exists(path)
    pd.DataFrame([row]).to_csv(path, mode="a", header=header, index=False, encoding="utf-8-sig")

def save_report(name: str, text: str):
    """保存文本报告到 out/"""
    ensure_dirs()
    fname = now().strftime(f"out/{name}_%Y-%m-%d_%H%M.txt")
    with open(fname, "w", encoding="utf-8") as f:
        f.write(text)


# ----------------- 日出时间 -----------------
def sunrise_time():
    """获取明日日出：返回(精确时间, 整点时间)"""
    try:
        js = requests.get(
            f"https://api.sunrise-sunset.org/json?lat={LAT}&lng={LON}&date=tomorrow&formatted=0",
            timeout=30
        ).json()
        t = dt.datetime.fromisoformat(js["results"]["sunrise"]).astimezone(TZ)
    except Exception as e:
        print("[WARN] sunrise-sunset API 失败，使用默认 06:00：", e)
        t = now().replace(hour=6, minute=0, second=0, microsecond=0) + dt.timedelta(days=1)
    t_exact = t
    t_hour  = t_exact.replace(minute=0, second=0, microsecond=0)
    return t_exact, t_hour


# ----------------- open-meteo 读取 -----------------
def open_meteo(lat=None, lon=None):
    """获取 open-meteo 小时级数据，若失败返回 None"""
    lat = LAT if lat is None else lat
    lon = LON if lon is None else lon
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        "&hourly=cloudcover_low,cloudcover_mid,cloudcover_high,visibility,"
        "temperature_2m,dewpoint_2m,windspeed_10m,precipitation"
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


# ----------------- METAR 云底 -----------------
def metar(code="ZGSZ"):
    """NOAA METAR 文本，没有就返回空字符串"""
    try:
        url = f"https://tgftp.nws.noaa.gov/data/observations/metar/stations/{code}.TXT"
        txt = requests.get(url, timeout=20).text.strip().splitlines()[-1]
        return txt
    except Exception as e:
        print("[WARN] METAR 获取失败：", e)
        return ""

def parse_cloud_base(metar_txt):
    """解析云底高度(m)"""
    import re
    m = re.findall(r'(BKN|OVC)(\d{3})', metar_txt)
    if not m:
        return None
    ft = int(m[0][1]) * 100
    return ft * 0.3048


# ----------------- meteoblue point API（可选） -----------------
def mb_point_lowcloud(lat, lon, when_hour):
    """
    读取 meteoblue Point API 的低云覆盖率 & 云底高度
    返回 dict: {"low_cloud": %, "cloud_base": m} 或 None
    """
    if not MB_API_KEY:
        return None
    try:
        # 这里以 basic-1h_basic-day 为例；请根据你的套餐调整 package 名称
        url = ("https://my.meteoblue.com/packages/basic-1h_basic-day"
               f"?apikey={MB_API_KEY}&lat={lat:.4f}&lon={lon:.4f}"
               "&format=json&tz=Asia/Shanghai")
        js = requests.get(url, timeout=30).json()

        # 找小时数据块
        data = js.get("data_1h") or js.get("data_hourly") or {}
        times = data.get("time") or data.get("time_local") or data.get("time_iso8601") or []
        # 格式兼容：YYYY-MM-DD HH:00
        tgt = when_hour.strftime("%Y-%m-%d %H:00")
        if tgt not in times:
            return None
        idx = times.index(tgt)

        def pick(keys, default=None):
            for k in keys:
                if k in data:
                    return data[k][idx]
            return default

        low  = pick(["low_clouds", "low_cloud_cover", "cloudcover_low"])
        base = pick(["cloud_base", "cloudbase", "cloud_base_height"])

        return {"low_cloud": low, "cloud_base": base}
    except Exception as e:
        print("[WARN] meteoblue point API 失败：", e)
        return None


# ----------------- 距离/方位角 -> 经纬度 -----------------
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


# ----------------- “低云墙预警”模型 -----------------
def fallback_cloudwall_model(sun_hour):
    """
    沿日出方向取多个距离点，读取低云覆盖率(必) + 云底高度(尽量)。
    用规则模型给出风险等级：
      2 = 预警；1 = 关注；0 = 正常
    返回 (score, text)
    """
    cfg = CONFIG.get("cloudwall", {})
    bearing = cfg.get("sunrise_azimuth", 90)
    dists   = cfg.get("sample_km", [20, 50, 80, 120])

    samples = []
    for d in dists:
        plat, plon = offset_latlon(LAT, LON, bearing, d)
        rec = mb_point_lowcloud(plat, plon, sun_hour)
        if rec is None:
            # 用 open-meteo 兜底
            om = open_meteo(plat, plon)
            if om is None:
                samples.append((d, None, None))
                continue
            times = om["hourly"]["time"]
            tgt = sun_hour.strftime("%Y-%m-%dT%H:00")
            if tgt not in times:
                idx = min(range(len(times)),
                          key=lambda i: abs(dt.datetime.fromisoformat(times[i]) - sun_hour))
            else:
                idx = times.index(tgt)
            low_pct = om["hourly"]["cloudcover_low"][idx]
            base_m  = None
        else:
            low_pct = rec.get("low_cloud")
            base_m  = rec.get("cloud_base")

        samples.append((d, low_pct, base_m))

    risk = model_lc_risk_v2(samples)
    txt  = risk_text_from_samples(risk, samples)
    return risk, txt


def model_lc_risk_v2(samples):
    """
    规则：
      - 任意样本 low>=50% 且 (cloud_base<600m 或 cloud_base 缺失) => 2(预警)
      - 或 >=50% 的样本 low>=30% => 2
      - 若存在样本 low>=30% 或 cloud_base<800m => 1(关注)
      - 否则 0(正常)
    """
    if not samples:
        return 1  # 无数据 → 关注
    high = sum(1 for _, l, b in samples if (l is not None and l >= 50) and (b is None or b < 600))
    mid  = sum(1 for _, l, b in samples if (l is not None and l >= 30) or (b is not None and b < 800))
    if high >= 1 or mid >= len(samples) * 0.5:
        return 2
    if mid >= 1:
        return 1
    return 0

def risk_text_from_samples(risk, samples):
    stat = {0:"正常(模型)",1:"关注(模型)",2:"预警(模型)"}.get(risk,"?")
    detail = " | ".join([f"{d}km:{(str(l)+'%') if l is not None else '?'} / {int(b) if b else 'NA'}m"
                         for d,l,b in samples])
    return f"{stat}（samples: {detail}）"


# ----------------- 评分逻辑 -----------------
def score_value(v, bounds):
    """
    bounds：
      3个数 [lo, mid, hi] : lo~mid=2分；其它近段=1分；否则0
      2个数 [lo, hi]      : >=hi=2分；>=lo=1分；否则0
    """
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

    # 低云
    lc = vals["low"]
    pt = score_value(lc, cfg["low_cloud"]); detail.append(("低云%", lc, pt)); total += pt

    # 中/高云
    mh = max(vals["mid"], vals["high"])
    pt = score_value(mh, cfg["mid_high_cloud"]); detail.append(("中/高云%", mh, pt)); total += pt

    # 云底
    if cloud_base_m is None:
        pt, val = 1, -1
    else:
        lo, hi = cfg["cloud_base_m"]
        pt = 2 if cloud_base_m > hi else 1 if cloud_base_m > lo else 0
        val = cloud_base_m
    detail.append(("云底高度m", val, pt)); total += pt

    # 能见度
    vis_km = (vals["vis"] or 0) / 1000.0
    lo, hi = cfg["visibility_km"]
    pt = 2 if vis_km >= hi else 1 if vis_km >= lo else 0
    detail.append(("能见度km", vis_km, pt)); total += pt

    # 风速
    w = vals["wind"]
    lo1, lo2, hi2, hi3 = cfg["wind_ms"]
    if lo2 <= w <= hi2:
        pt = 2
    elif lo1 <= w < lo2 or hi2 < w <= hi3:
        pt = 1
    else:
        pt = 0
    detail.append(("风速m/s", w, pt)); total += pt

    # 露点差
    dp = vals["t"] - vals["td"]
    lo, hi = cfg["dewpoint_diff"]
    pt = 2 if dp >= hi else 1 if dp >= lo else 0
    detail.append(("露点差°C", dp, pt)); total += pt

    # 降雨量
    p = vals.get("precip", 0)
    lo, hi = cfg["precip_mm"]
    pt = 2 if p < lo else 1 if p < hi else 0
    detail.append(("降雨量mm", p, pt)); total += pt

    return total, detail


# ----------------- 文案 -----------------
def build_forecast_text(total, det, sun_t, extra):
    lines = [
        f"【日出预报 | 明早 {sun_t:%m月%d日}】拍摄指数：{total}/18",
        f"地点：{CONFIG['location']['name']}  (lat={LAT}, lon={LON})",
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
    """根据主要指标生成普通人可读描述（5分制）"""
    lc   = kv.get("低云%",      0) or 0
    mh   = kv.get("中/高云%",    0) or 0
    cb   = kv.get("云底高度m",   -1)
    vis  = kv.get("能见度km",    0) or 0
    wind = kv.get("风速m/s",     0) or 0
    dp   = kv.get("露点差°C",    0) or 0
    rp   = kv.get("降雨量mm",    0) or 0

    # 低云
    if lc < 20:
        lc_level = "低";      low_text = "地平线基本通透，太阳能“蹦”出来"
    elif lc < 40:
        lc_level = "中";      low_text = "地平线可能有一条灰带，太阳或从缝隙钻出"
    elif lc < 60:
        lc_level = "偏高";    low_text = "低云偏多，首轮日光可能被挡一部分"
    else:
        lc_level = "高";      low_text = "一堵低云墙，首轮日光大概率看不到"

    # 中/高云
    if 20 <= mh <= 60:
        mh_level = "适中";    fire_text = "有“云接光”舞台，可能染上粉橙色（小概率火烧云）"
    elif mh < 20:
        mh_level = "太少";    fire_text = "天空太干净，只有简单渐变色"
    elif mh <= 80:
        mh_level = "偏多";    fire_text = "云多且厚，色彩可能偏闷"
    else:
        mh_level = "很多";    fire_text = "厚云盖顶，大概率阴沉"

    # 云底高度
    if cb is None or cb < 0:
        cb_level, cb_text, cb_show = "未知", "云底数据缺失，可参考凌晨“低云墙预警”", "未知"
    elif cb > 1000:
        cb_level, cb_text, cb_show = ">1000m", "云底较高，多当“天花板”，不挡海平线", f"{cb:.0f}m"
    elif cb > 500:
        cb_level, cb_text, cb_show = "500~1000m", "可能在海面上方形成一道云棚，注意日出角度", f"{cb:.0f}m"
    else:
        cb_level, cb_text, cb_show = "<500m", "贴海低云/雾，像拉了窗帘", f"{cb:.0f}m"

    # 能见度
    if vis >= 15:
        vis_level, vis_text = ">15km", "空气透明度好，远景清晰，金光反射漂亮"
    elif vis >= 8:
        vis_level, vis_text = "8~15km", "中等透明度，远景略灰"
    else:
        vis_level, vis_text = "<8km", "背景灰蒙蒙，层次差"

    # 风速
    if 2 <= wind <= 5:
        wind_level, wind_text = "2~5m/s", "海面有微波纹，反光好，三脚架稳"
    elif wind < 2:
        wind_level, wind_text = "<2m/s", "几乎无风，注意镜头易结露"
    elif wind <= 8:
        wind_level, wind_text = "5~8m/s", "风稍大，留意三脚架稳定性"
    else:
        wind_level, wind_text = ">8m/s", "大风天，拍摄体验差，器材要护好"

    # 露点差
    if dp >= 3:
        dp_level, dp_text = "≥3℃", "不易起雾"
    elif dp >= 1:
        dp_level, dp_text = "1~3℃", "稍潮，镜头可能结露"
    else:
        dp_level, dp_text = "<1℃", "极易起雾，注意海雾/镜头起雾风险"

    # 降雨
    if rp < 0.1:
        rp_level, rain_text = "<0.1mm", "几乎不会下雨"
    elif rp < 1:
        rp_level, rain_text = "0.1~1mm", "可能有零星小雨/毛毛雨"
    else:
        rp_level, rain_text = "≥1mm", "有下雨可能，注意防水和收纳镜头"

    # 评分等级
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


# --------- 简单旧版风险（保留兼容） ---------
def model_lc_risk_simple(lc, dp, wind):
    if lc is None:
        return 1
    if lc >= 50 and dp < 2:
        return 2
    if lc >= 30:
        return 1
    return 0

RISK_MAP = {0: "正常", 1: "关注", 2: "高风险"}


# ----------------- 各模式 -----------------
def run_forecast():
    sun_exact, sun_hour = sunrise_time()

    # 1. 拿 open-meteo 主体数据（评分）
    om = open_meteo()
    if om is None:
        msg = "[ERR] open-meteo 数据为空，无法评分。"
        print(msg)
        save_report("forecast_error", msg)
        log_csv(CONFIG["paths"]["log_scores"], {
            "time": now(), "mode": "forecast", "score": -1, "error": "open-meteo no data"
        })
        return

    hrs = om["hourly"]["time"]
    target = sun_hour.strftime("%Y-%m-%dT%H:00")
    if target not in hrs:
        print("[WARN] 未找到日出整点，取最近小时。")
        idx = min(range(len(hrs)),
                  key=lambda i: abs(dt.datetime.fromisoformat(hrs[i]) - sun_hour))
    else:
        idx = hrs.index(target)

    vals = dict(
        low    = om["hourly"]["cloudcover_low"][idx],
        mid    = om["hourly"]["cloudcover_mid"][idx],
        high   = om["hourly"]["cloudcover_high"][idx],
        vis    = om["hourly"]["visibility"][idx],
        t      = om["hourly"]["temperature_2m"][idx],
        td     = om["hourly"]["dewpoint_2m"][idx],
        wind   = om["hourly"]["windspeed_10m"][idx],
        precip = om["hourly"]["precipitation"][idx]
    )

    # 2. 云底高度：METAR
    cb = parse_cloud_base(metar("ZGSZ"))

    # 3. 总分
    total, det = calc_score(vals, cb, CONFIG["scoring"])

    # 4. 模型低云墙风险（简单 + 多点模型）
    risk_simple = model_lc_risk_simple(vals["low"], vals["t"] - vals["td"], vals["wind"])
    risk_simple_text = f"{RISK_MAP[risk_simple]}（模型12h）"

    risk_model, risk_model_text = fallback_cloudwall_model(sun_hour)

    # 5. 文案
    score5 = round(total / (3 * len(det)) * 5, 1)
    kv = {k: v for k, v, _ in det}
    scene_txt = (
        gen_scene_desc(score5, kv, sun_exact)
        + f"\n- 低云墙风险（模型12h）：{risk_simple_text}"
        + f"\n- 低云墙预警（模型多点）：{risk_model_text}"
    )

    text = scene_txt + "\n\n" + build_forecast_text(total, det, sun_exact, extra={})

    print(text)
    save_report("forecast", text)
    log_csv(CONFIG["paths"]["log_scores"], {
        "time": now(), "mode": "forecast", "score": total, "score5": score5,
        **{k: v for k, v, _ in det},
        "risk_model_simple": risk_simple,
        "risk_model_multi": risk_model
    })


def run_check(mode: str):
    """
    夜检/凌晨检：
    - 不抓卫星，直接跑多点模型预警，记录风险值
    """
    _, sun_hour = sunrise_time()
    risk, txt = fallback_cloudwall_model(sun_hour)
    msg = f"{mode}: risk={risk}, {txt}"
    print(msg)
    save_report(mode, msg)
    log_csv(CONFIG["paths"]["log_cloud"], {
        "time": now(), "mode": mode,
        "cloudwall_score": risk,
        "text": txt
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
        print(err_msg)
        save_report(f"{mode}_error", err_msg)
        log_csv(CONFIG["paths"]["log_scores"], {
            "time": now(), "mode": mode, "score": -1, "error": repr(e)
        })
