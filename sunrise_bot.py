#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sunrise_bot.py
模式：
  python sunrise_bot.py forecast    # 下午预测（日出评分+文案）
  python sunrise_bot.py nightcheck  # 22:00 夜检（记录低云墙指数）
  python sunrise_bot.py lastcheck   # 03:30 凌晨检（记录低云墙指数）

本版本：
- 不推送飞书/邮件，只 print，并保存到 out/、logs/ 目录
- 对 open-meteo 接口做了健壮性判断，避免 KeyError: 'hourly'
"""

import os, sys, json, yaml, datetime as dt
import requests, pandas as pd, numpy as np
import pytz

# ----------------- 全局配置 -----------------
TZ = pytz.timezone("Asia/Shanghai")
CONFIG = yaml.safe_load(open("config.yaml", "r", encoding="utf-8"))

LAT = CONFIG["location"]["lat"]
LON = CONFIG["location"]["lon"]

# 是否启用低云墙检测（需要 opencv）
USE_CLOUDWALL = True
try:
    import cv2
    from cloudwall import low_cloud_ratio, trend_alert
except Exception as e:
    USE_CLOUDWALL = False
    print("[WARN] OpenCV/cloudwall 加载失败，低云墙检测关闭：", e)


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
        "&hourly=low_clouds,mid_clouds,high_clouds,visibility,"
        "temperature_2m,dewpoint_2m,wind_speed_10m,precipitation"
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

def sunrise_time():
    """获取明日日出时间（整点小时）"""
    try:
        js = requests.get(
            f"https://api.sunrise-sunset.org/json?lat={LAT}&lng={LON}&date=tomorrow&formatted=0",
            timeout=30
        ).json()
        t = dt.datetime.fromisoformat(js["results"]["sunrise"]).astimezone(TZ)
    except Exception as e:
        print("[WARN] sunrise-sunset API 失败，使用默认 06:00：", e)
        t = now().replace(hour=6, minute=0, second=0, microsecond=0) + dt.timedelta(days=1)
    return t.replace(minute=0, second=0, microsecond=0)

def fetch_himawari_frames(n=6, step=10):
    """抓取最近 n 帧 Himawari Band13 PNG"""
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
        try:
            r = requests.get(url, timeout=30)
            if r.status_code == 200:
                arr = np.frombuffer(r.content, np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                frames.append((t, img))
        except Exception as e:
            print("[WARN] 下载卫星帧失败：", e)
    frames.sort(key=lambda x: x[0])
    return frames

# ----------------- 评分逻辑 -----------------
def score_value(v, bounds):
    """
    bounds 长度：
      3个数 [lo, mid, hi] : lo~mid=2分；其余靠近区间=1分，否则0
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

    lc = vals["low"]
    pt = score_value(lc, cfg["low_cloud"]); detail.append(("低云%", lc, pt)); total += pt

    mh = max(vals["mid"], vals["high"])
    pt = score_value(mh, cfg["mid_high_cloud"]); detail.append(("中/高云%", mh, pt)); total += pt

    if cloud_base_m is None:
        pt, val = 1, -1
    else:
        lo, hi = cfg["cloud_base_m"]
        pt = 2 if cloud_base_m > hi else 1 if cloud_base_m > lo else 0
        val = cloud_base_m
    detail.append(("云底高度m", val, pt)); total += pt

    vis_km = (vals["vis"] or 0) / 1000.0
    lo, hi = cfg["visibility_km"]
    pt = 2 if vis_km >= hi else 1 if vis_km >= lo else 0
    detail.append(("能见度km", vis_km, pt)); total += pt

    w = vals["wind"]
    lo1, lo2, hi2, hi3 = cfg["wind_ms"]
    if lo2 <= w <= hi2:
        pt = 2
    elif lo1 <= w < lo2 or hi2 < w <= hi3:
        pt = 1
    else:
        pt = 0
    detail.append(("风速m/s", w, pt)); total += pt

    dp = vals["t"] - vals["td"]
    lo, hi = cfg["dewpoint_diff"]
    pt = 2 if dp >= hi else 1 if dp >= lo else 0
    detail.append(("露点差°C", dp, pt)); total += pt

    return total, detail

# ----------------- 文案 -----------------
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

# ----------------- 三个模式 -----------------
def run_forecast():
    sun_t = sunrise_time()
    om = open_meteo()
    if om is None:
        msg = "[ERR] open-meteo 数据为空，无法评分。请稍后人工查看。"
        print(msg)
        save_report("forecast_error", msg)
        log_csv(CONFIG["paths"]["log_scores"], {
            "time": now(), "mode": "forecast", "score": -1, "error": "open-meteo no data"
        })
        return

    hrs = om["hourly"]["time"]
    # 目标小时字符串
    target = sun_t.strftime("%Y-%m-%dT%H:00")
    if target not in hrs:
        print("[WARN] 未找到日出整点，尝试取最近小时。")
        idx = min(range(len(hrs)),
                  key=lambda i: abs(dt.datetime.fromisoformat(hrs[i]) - sun_t))
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

    print(text)
    save_report("forecast", text)
    log_csv(CONFIG["paths"]["log_scores"], {
        "time": now(), "mode": "forecast", "score": total,
        **{k: v for k, v, _ in det}
    })

def run_check(mode: str):
    """夜检/凌晨检：记录低云墙指数，可选预警"""
    if not USE_CLOUDWALL:
        msg = f"{mode}: 未启用低云墙检测，跳过。"
        print(msg)
        save_report(mode, msg)
        log_csv(CONFIG["paths"]["log_cloud"], {
            "time": now(), "mode": mode, "cloudwall_score": -1, "ratios": "[]"
        })
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
        # 捕获所有异常，避免 Actions 标红；也把错误写出来方便排查
        err_msg = f"[FATAL] {mode} 运行异常：{repr(e)}"
        print(err_msg)
        save_report(f"{mode}_error", err_msg)
        log_csv(CONFIG["paths"]["log_scores"], {
            "time": now(), "mode": mode, "score": -1, "error": repr(e)
        })
        # 不 raise，保证 exit code 0
