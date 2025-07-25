#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sunrise_bot.py
运行模式：
  python sunrise_bot.py forecast   # 下午预测
  python sunrise_bot.py nightcheck # 22:00 夜检（只记录低云墙指数）
  python sunrise_bot.py lastcheck  # 03:30 凌晨检（只记录低云墙指数）

注意：本版本不发送飞书消息，只 print + 保存到 out/、logs/，由 GitHub Actions 打包为 Artifact。
"""

import os, sys, json, yaml, time, datetime as dt
import requests, pandas as pd, numpy as np

import pytz

# ---------- 路径/配置 ----------
TZ = pytz.timezone("Asia/Shanghai")
CONFIG = yaml.safe_load(open("config.yaml", "r", encoding="utf-8"))

LAT = CONFIG["location"]["lat"]
LON = CONFIG["location"]["lon"]

# 可选导入 OpenCV（夜检/凌晨检才用到）
USE_CLOUDWALL = True
try:
    import cv2
    from cloudwall import low_cloud_ratio, trend_alert
except Exception as e:
    USE_CLOUDWALL = False
    print("[WARN] OpenCV/cloudwall 未加载，将跳过低云墙检测。", e)

# ---------- 公共工具 ----------
def now():
    return dt.datetime.now(TZ)

def ensure_dirs():
    os.makedirs("logs", exist_ok=True)
    os.makedirs("out", exist_ok=True)

def log_csv(path_tpl: str, row: dict):
    """追加写入 CSV（按月份滚动文件）"""
    ensure_dirs()
    path = now().strftime(path_tpl)
    header = not os.path.exists(path)
    pd.DataFrame([row]).to_csv(path, mode="a", header=header, index=False, encoding="utf-8-sig")

def save_report(name: str, text: str):
    """保存文本报告"""
    ensure_dirs()
    fname = now().strftime(f"out/{name}_%Y-%m-%d_%H%M.txt")
    with open(fname, "w", encoding="utf-8") as f:
        f.write(text)

# ---------- 数据获取 ----------
def open_meteo():
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={LAT}&longitude={LON}"
        "&hourly=low_clouds,mid_clouds,high_clouds,visibility,"
        "temperature_2m,dewpoint_2m,wind_speed_10m,precipitation"
        "&forecast_days=2&timezone=Asia%2FShanghai"
    )
    return requests.get(url, timeout=20).json()

def aqi():
    # 如果未来想用 AQI，预留函数
    token = os.environ.get("AQICN_TOKEN")
    if not token:
        return None
    url = f"https://api.waqi.info/feed/geo:{LAT};{LON}/?token={token}"
    return requests.get(url, timeout=20).json()

def metar(code="ZGSZ"):
    """NOAA METAR 文本"""
    try:
        url = f"https://tgftp.nws.noaa.gov/data/observations/metar/stations/{code}.TXT"
        txt = requests.get(url, timeout=20).text.strip().splitlines()[-1]
        return txt
    except Exception:
        return ""

def parse_cloud_base(metar_txt):
    """解析云底高度(m)"""
    import re
    m = re.findall(r'(BKN|OVC)(\d{3})', metar_txt)
    if not m:
        return None
    ft = int(m[0][1]) * 100
    return ft * 0.3048

def sunrise_time():
    js = requests.get(
        f"https://api.sunrise-sunset.org/json?lat={LAT}&lng={LON}&date=tomorrow&formatted=0",
        timeout=20
    ).json()
    t = dt.datetime.fromisoformat(js["results"]["sunrise"]).astimezone(TZ)
    return t.replace(minute=0, second=0, microsecond=0)

def fetch_himawari_frames(n=6, step=10):
    """抓取最近 n 帧 Himawari-9 Band13 PNG（每 10 分钟一张）"""
    if not USE_CLOUDWALL:
        return []
    frames = []
    base = now()
    for i in range(n):
        t = base - dt.timedelta(minutes=step * i)
        url = (
            "https://himawari8.nict.go.jp/img/D531106/2d/550/"
            f"{t.strftime('%Y%m%d')}/{t.strftime('%H%M')}00_0_0.png"
        )
        r = requests.get(url, timeout=20)
        if r.status_code == 200:
            arr = np.frombuffer(r.content, np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            frames.append((t, img))
    frames.sort(key=lambda x: x[0])
    return frames

# ---------- 评分 ----------
def score_value(v, bounds):
    """
    bounds: 2 或 3 个数
    - 长度3: [lo, mid, hi]  → lo~mid=2分，mid~hi=1分，其它0
    - 长度2: [lo, hi]       → >=hi=2分，>=lo=1分，其它0
    """
    if v is None:
        return 1  # 未知给1分
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
    detail = []
    total = 0

    # 低云
    lc = vals["low"]
    pt = score_value(lc, cfg["low_cloud"])
    detail.append(("低云%", lc, pt)); total += pt

    # 中/高云
    mh = max(vals["mid"], vals["high"])
    pt = score_value(mh, cfg["mid_high_cloud"])
    detail.append(("中/高云%", mh, pt)); total += pt

    # 云底高度
    if cloud_base_m is None:
        pt = 1; val = -1
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

    return total, detail

# ---------- 文案 ----------
def build_forecast_text(total, det, sun_t, extra):
    lines = [
        f"【日出预报 | {now():%m月%d日} → 明早】可拍指数：{total}/18",
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

# ---------- 模式 ----------
def run_forecast():
    sun_t = sunrise_time()

    om = open_meteo()
    hrs = om["hourly"]["time"]
    target = sun_t.strftime("%Y-%m-%dT%H:00")
    if target not in hrs:
        print("[WARN] open-meteo 未找到对应小时，取最近小时")
        # 找最近的小时
        idx = min(range(len(hrs)), key=lambda i: abs(dt.datetime.fromisoformat(hrs[i]) - sun_t))
    else:
        idx = hrs.index(target)

    vals = dict(
        low  = om["hourly"]["low_clouds"][idx],
        mid  = om["hourly"]["mid_clouds"][idx],
        high = om["hourly"]["high_clouds"][idx],
        vis  = om["hourly"]["visibility"][idx],
        t    = om["hourly"]["temperature_2m"][idx],
        td   = om["hourly"]["dewpoint_2m"][idx],
        wind = om["hourly"]["wind_speed_10m"][idx]
    )

    mtxt = metar("ZGSZ")
    cb = parse_cloud_base(mtxt)

    total, det = calc_score(vals, cb, CONFIG["scoring"])
    text = build_forecast_text(total, det, sun_t, extra={})

    print(text)              # 输出到日志
    save_report("forecast", text)
    log_csv(CONFIG["paths"]["log_scores"], {
        "time": now(), "mode": "forecast", "score": total,
        **{k: v for k, v, _ in det}
    })

def run_check(mode):
    if not USE_CLOUDWALL:
        print(f"[INFO] {mode}: 未启用低云墙检测，跳过。")
        save_report(mode, "cloudwall: skipped")
        return

    cfg = CONFIG["cloudwall"]
    frames = fetch_himawari_frames(cfg["frames"], cfg["step_min"])
    ratios = []
    for _, img in frames:
        r = low_cloud_ratio(img, cfg["roi"], cfg["gray_threshold"])
        ratios.append(r)

    cw_score = trend_alert(ratios, cfg["ratio_warn"])

    msg = f"{mode}: cloudwall_score={cw_score}, ratios={ratios}"
    print(msg)
    save_report(mode, msg)
    log_csv(CONFIG["paths"]["log_cloud"], {
        "time": now(), "mode": mode,
        "cloudwall_score": cw_score,
        "ratios": json.dumps(ratios, ensure_ascii=False)
    })

# ---------- 入口 ----------
if __name__ == "__main__":
    ensure_dirs()
    mode = sys.argv[1] if len(sys.argv) > 1 else "forecast"
    if mode == "forecast":
        run_forecast()
    elif mode == "nightcheck":
        run_check("nightcheck")
    elif mode == "lastcheck":
        run_check("lastcheck")
    else:
        print("Usage: python sunrise_bot.py [forecast|nightcheck|lastcheck]")
