#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sunset_bot.py
用途：
  每天早上 05:30（北京时间）单独跑一次“日落预测”，输出拍摄指数+普通人可读文案。

特点：
- 不抓卫星图，只用数值/模型数据兜底判断低云墙风险
- 评分逻辑、文本模版与 sunrise_bot 基本一致，但 event 改为“日落”
"""

import os, sys, json, yaml, datetime as dt
import requests, pandas as pd, numpy as np
import pytz

# ----------------- 全局配置 -----------------
TZ = pytz.timezone("Asia/Shanghai")
CONFIG = yaml.safe_load(open("config.yaml", "r", encoding="utf-8"))

LAT = CONFIG["location"]["lat"]
LON = CONFIG["location"]["lon"]

# ----------------- 工具函数 -----------------
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

# ----------------- 数据获取 -----------------
def open_meteo():
    """获取 open-meteo 小时级数据，并做健壮性校验"""
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={LAT}&longitude={LON}"
        "&hourly=cloudcover_low,cloudcover_mid,cloudcover_high,visibility,"
        "temperature_2m,dewpoint_2m,windspeed_10m,precipitation"
        "&forecast_days=2&timezone=Asia%2FShanghai"
    )
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        data = r.json()
        if "hourly" not in data or "time" not in data["hourly"]:
            print("[ERR] open-meteo 响应缺失 hourly 字段，原始响应：", data)
            return None
        return data
    except Exception as e:
        print("[ERR] open-meteo 请求失败：", e)
        return None

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

def sunset_time():
    """获取今日（日落未过）或明日日落时间：返回(精确时间, 整点时间)"""
    try:
        js = requests.get(
            f"https://api.sunrise-sunset.org/json?lat={LAT}&lng={LON}&date=today&formatted=0",
            timeout=30
        ).json()
        t = dt.datetime.fromisoformat(js["results"]["sunset"]).astimezone(TZ)
        if t < now():  # 若已过日落，取明天
            js = requests.get(
                f"https://api.sunrise-sunset.org/json?lat={LAT}&lng={LON}&date=tomorrow&formatted=0",
                timeout=30
            ).json()
            t = dt.datetime.fromisoformat(js["results"]["sunset"]).astimezone(TZ)
    except Exception as e:
        print("[WARN] sunset API 失败，使用默认 18:30：", e)
        t = now().replace(hour=18, minute=30, second=0, microsecond=0)
    t_exact = t
    t_hour  = t_exact.replace(minute=0, second=0, microsecond=0)
    return t_exact, t_hour

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
def build_forecast_text(total, det, sun_t, extra, event_name="日落"):
    lines = [
        f"拍摄指数：{total}/18",
        f"地点：{CONFIG['location']['name']}  (lat={LAT}, lon={LON})",
        f"{event_name}：{sun_t:%H:%M}",
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

def gen_scene_desc(score5, kv, sun_t, event_name="日落"):
    """普通人可读描述（5分制、带等级+画面感）"""
    lc   = kv.get("低云%",      0) or 0
    mh   = kv.get("中/高云%",    0) or 0
    cb   = kv.get("云底高度m",   -1)
    vis  = kv.get("能见度km",    0) or 0
    wind = kv.get("风速m/s",     0) or 0
    dp   = kv.get("露点差°C",    0) or 0
    rp   = kv.get("降雨量mm",    0) or 0

    # 低云
    if lc < 20:
        lc_level = "低";   low_text = "地平线基本通透，太阳能“蹦”出来"
    elif lc < 40:
        lc_level = "中";   low_text = "地平线可能有一条灰带，太阳或从缝隙钻出"
    elif lc < 60:
        lc_level = "偏高"; low_text = "低云偏多，首轮日光可能被挡一部分"
    else:
        lc_level = "高";   low_text = "一堵低云墙，首轮日光大概率看不到"

    # 中/高云
    if 20 <= mh <= 60:
        mh_level = "适中"; fire_text = "有“云接光”舞台，可能染上粉橙色（小概率火烧云）"
    elif mh < 20:
        mh_level = "太少"; fire_text = "天空太干净，只有简单渐变色"
    elif mh <= 80:
        mh_level = "偏多"; fire_text = "云多且厚，色彩可能偏闷"
    else:
        mh_level = "很多"; fire_text = "厚云盖顶，大概率阴沉"

    # 云底高度
    if cb is None or cb < 0:
        cb_level = "未知";     cb_text  = "云底数据缺失，可参考凌晨实况";   cb_show = "未知"
    elif cb > 1000:
        cb_level = ">1000m";   cb_text  = "云底较高，多当“天花板”";     cb_show = f"{cb:.0f}m"
    elif cb > 500:
        cb_level = "500~1000m";cb_text  = "可能在海面上方形成一道云棚";   cb_show = f"{cb:.0f}m"
    else:
        cb_level = "<500m";    cb_text  = "贴海低云/雾，像拉了窗帘";       cb_show = f"{cb:.0f}m"

    # 能见度
    if vis >= 15:
        vis_level = ">15km"; vis_text  = "空气透明度好，远景清晰，金光反射漂亮"
    elif vis >= 8:
        vis_level = "8~15km"; vis_text  = "中等透明度，远景略灰"
    else:
        vis_level = "<8km";  vis_text  = "背景灰蒙蒙，层次差"

    # 风速
    if 2 <= wind <= 5:
        wind_level = "2~5m/s";  wind_text  = "海面有微波纹，反光好，三脚架稳"
    elif wind < 2:
        wind_level = "<2m/s";   wind_text  = "几乎无风，注意镜头容易结露"
    elif wind <= 8:
        wind_level = "5~8m/s";  wind_text  = "风稍大，留意三脚架稳定性"
    else:
        wind_level = ">8m/s";   wind_text  = "大风天，拍摄体验差，器材要护好"

    # 露点差
    if dp >= 3:
        dp_level = "≥3℃";   dp_text  = "不易起雾"
    elif dp >= 1:
        dp_level = "1~3℃";  dp_text  = "稍潮，镜头可能结露"
    else:
        dp_level = "<1℃";   dp_text  = "极易起雾，注意海雾/镜头起雾风险"

    # 降雨
    if rp < 0.1:
        rp_level = "<0.1mm"; rain_text = "几乎不会下雨"
    elif rp < 1:
        rp_level = "0.1~1mm"; rain_text = "可能有零星小雨/毛毛雨"
    else:
        rp_level = "≥1mm";   rain_text = "有下雨可能，注意防水和收纳镜头"

    # 评分等级
    if score5 >= 4.0:
        grade = "建议出发（把握较大）"
    elif score5 >= 3.0:
        grade = "可去一搏（不稳）"
    elif score5 >= 2.0:
        grade = "机会一般（看心情或距离）"
    elif score5 >= 1.0:
        grade = "概率很小（除非就在附近）"
    else:
        grade = "建议休息（基本无戏）"

    return (
        f"【直观判断】评分：{score5:.1f}/5 —— {grade}\n"
        f"{event_name}：{sun_t:%H:%M}\n"
        f"- 低云：{lc:.0f}%（{lc_level}）— {low_text}\n"
        f"- 中/高云：{mh:.0f}%（{mh_level}）— {fire_text}\n"
        f"- 云底高度：{cb_show}（{cb_level}）— {cb_text}\n"
        f"- 能见度：{vis:.1f} km（{vis_level}）— {vis_text}\n"
        f"- 风速：{wind:.1f} m/s（{wind_level}）— {wind_text}\n"
        f"- 降雨：{rp:.1f} mm（{rp_level}）— {rain_text}\n"
        f"- 露点差：{dp:.1f} ℃（{dp_level}）— {dp_text}"
    )

# ---- 简易模型低云墙风险（卫星兜底失败时用） ----
def model_lc_risk(lc, dp, wind):
    """
    简易规则：
    - lc≥50 且 dp<2 → 高风险(2)
    - lc≥30 → 关注(1)
    - 其它 → 正常(0)
    """
    if lc is None:
        return 1
    if lc >= 50 and dp < 2:
        return 2
    if lc >= 30:
        return 1
    return 0

RISK_MAP = {0: "正常", 1: "关注", 2: "高风险"}

# ----------------- 主流程 -----------------
def run_sunset():
    sun_exact, sun_hour = sunset_time()

    om = open_meteo()
    if om is None:
        msg = "[ERR] open-meteo 数据为空，无法评分（日落）。"
        print(msg)
        save_report("sunset_error", msg)
        log_csv(CONFIG["paths"]["log_scores"], {
            "time": now(), "mode": "sunset", "score": -1, "error": "open-meteo no data"
        })
        return

    hrs = om["hourly"]["time"]
    target = sun_hour.strftime("%Y-%m-%dT%H:00")
    if target not in hrs:
        print("[WARN] 未找到日落整点，取最近小时。")
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

    mtxt = metar("ZGSZ")
    cb = parse_cloud_base(mtxt)

    total, det = calc_score(vals, cb, CONFIG["scoring"])
    score5 = round(total / (3 * len(det)) * 5, 1)
    kv = {k: v for k, v, _ in det}

    risk_model = model_lc_risk(vals["low"], vals["t"] - vals["td"], vals["wind"])
    risk_text  = f"{RISK_MAP[risk_model]}（模型）"

    scene_txt = (
        gen_scene_desc(score5, kv, sun_exact, event_name="日落")
        + f"\n- 低云墙风险（模型）：{risk_text}"
    )
    text = scene_txt + "\n\n" + build_forecast_text(total, det, sun_exact, extra={}, event_name="日落")

    print(text)
    save_report("sunset", text)
    log_csv(CONFIG["paths"]["log_scores"], {
        "time": now(), "mode": "sunset", "score": total, "score5": score5,
        **{k: v for k, v, _ in det},
        "risk_model": risk_model
    })

# ----------------- 主入口 -----------------
if __name__ == "__main__":
    ensure_dirs()
    try:
        run_sunset()
    except Exception as e:
        err_msg = f"[FATAL] sunset 运行异常：{repr(e)}"
        print(err_msg)
        save_report("sunset_error", err_msg)
        log_csv(CONFIG["paths"]["log_scores"], {
            "time": now(), "mode": "sunset", "score": -1, "error": repr(e)
        })
