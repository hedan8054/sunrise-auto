#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sunset_bot.py （无卫星版 + 云底高度自动校准 + 多点低云墙模型）

用途：
  每天早上 05:30 跑一次当日日落预测，输出拍摄指数 + 文案。

与 sunrise_bot.py 保持一致的改动点：
- 移除卫星依赖
- 低云墙预警：meteoblue/open-meteo 多点采样模型
- 云底高度：METAR 优先；无则用 露点差×系数 估算，并自动校准该系数
- 安全格式化，避免 NoneType.__format__ 异常
"""

import os, sys, json, yaml, math, glob
import datetime as dt

import requests
import pandas as pd
import numpy as np
import pytz
import warnings, urllib3
warnings.filterwarnings("ignore", message="Unverified HTTPS request")
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ----------------- 全局配置 -----------------
TZ = pytz.timezone("Asia/Shanghai")
CONFIG = yaml.safe_load(open("config.yaml", "r", encoding="utf-8"))

LAT = CONFIG["location"]["lat"]
LON = CONFIG["location"]["lon"]

# 可选 meteoblue key（放 Secrets）
MB_API_KEY = os.getenv("MB_API_KEY", "").strip()

# ----------------- 小工具 -----------------
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

# ---- 安全格式化，防止 None / NaN 崩溃 ----
def fmt(v, spec=".0f", na="NA"):
    try:
        if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
            return na
        return format(v, spec)
    except Exception:
        return na

# ----------------- 日落时间 -----------------
def sunset_time():
    """
    返回 (精确时间, 整点时间)，如果今天日落已过则取明天。
    """
    try:
        js = requests.get(
            f"https://api.sunrise-sunset.org/json?lat={LAT}&lng={LON}&date=today&formatted=0",
            timeout=30
        ).json()
        t = dt.datetime.fromisoformat(js["results"]["sunset"]).astimezone(TZ)
        if t < now():
            js = requests.get(
                f"https://api.sunrise-sunset.org/json?lat={LAT}&lng={LON}&date=tomorrow&formatted=0",
                timeout=30
            ).json()
            t = dt.datetime.fromisoformat(js["results"]["sunset"]).astimezone(TZ)
    except Exception as e:
        print("[WARN] sunset API 失败，使用默认 18:30：", e)
        t = now().replace(hour=18, minute=30, second=0, microsecond=0)
    return t, t.replace(minute=0, second=0, microsecond=0)

# ----------------- open-meteo -----------------
def open_meteo(lat=None, lon=None):
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
    try:
        url = f"https://tgftp.nws.noaa.gov/data/observations/metar/stations/{code}.TXT"
        txt = requests.get(url, timeout=20).text.strip().splitlines()[-1]
        return txt
    except Exception as e:
        print("[WARN] METAR 获取失败：", e)
        return ""

def parse_cloud_base(metar_txt):
    import re
    m = re.findall(r'(BKN|OVC)(\d{3})', metar_txt)
    if not m:
        return None
    ft = int(m[0][1]) * 100
    return ft * 0.3048

# ----------------- meteoblue point API（可选） -----------------
def mb_point_lowcloud(lat, lon, when_hour):
    """
    读取 meteoblue Point API 低云覆盖率/云底高度（如果 key 可用）。
    返回 dict 或 None。
    """
    if not MB_API_KEY:
        return None
    try:
        url = ("https://my.meteoblue.com/packages/basic-1h_basic-day"
               f"?apikey={MB_API_KEY}&lat={lat:.4f}&lon={lon:.4f}"
               "&format=json&tz=Asia/Shanghai")
        js = requests.get(url, timeout=30).json()

        data = js.get("data_1h") or js.get("data_hourly") or {}
        times = data.get("time") or data.get("time_local") or data.get("time_iso8601") or []
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

# ----------------- 经纬度偏移 -----------------
def offset_latlon(lat, lon, bearing_deg, dist_km):
    R = 6371.0
    brng = math.radians(bearing_deg)
    lat1 = math.radians(lat); lon1 = math.radians(lon)
    lat2 = math.asin(math.sin(lat1)*math.cos(dist_km/R) +
                     math.cos(lat1)*math.sin(dist_km/R)*math.cos(brng))
    lon2 = lon1 + math.atan2(math.sin(brng)*math.sin(dist_km/R)*math.cos(lat1),
                             math.cos(lat1)*math.cos(dist_km/R)-math.sin(lat1)*math.sin(lat2))
    return math.degrees(lat2), math.degrees(lon2)

# ----------------- 云底估算 & 自动校准 -----------------
def estimate_cloud_base_from_dp(dp_c, coef):
    if dp_c is None or dp_c <= 0 or coef is None:
        return None
    return dp_c * coef

def auto_calibrate_coef():
    """
    使用最近 N 天历史（有 METAR 云底 & 露点差）自动估算最佳系数。
    逻辑与 sunrise_bot 一致。
    """
    cfg = CONFIG.get("cloudbase_estimate", {})
    default_coef = cfg.get("default_coef_m_per_deg", 125)
    min_samples  = cfg.get("min_samples", 10)
    max_days     = cfg.get("max_window_days", 45)
    clip_min     = cfg.get("clip_min", 60)
    clip_max     = cfg.get("clip_max", 200)

    tpl = CONFIG["paths"]["log_scores"]
    pattern = tpl.replace("%Y", "*").replace("%m", "*")
    files = sorted(glob.glob(pattern))
    if not files:
        return {"coef_used": default_coef, "coef_new": None, "n": 0,
                "mae_old": None, "mae_new": None, "reason": "no_history_file"}

    cutoff = now() - dt.timedelta(days=max_days)
    ratios, errs_old = [], []
    used = 0
    for f in files:
        try:
            df = pd.read_csv(f, encoding="utf-8-sig")
        except Exception:
            continue
        if "time" not in df.columns:
            continue
        try:
            df["time"] = pd.to_datetime(df["time"])
            df = df[df["time"] >= cutoff]
        except Exception:
            pass
        if "云底高度m" not in df.columns or "露点差°C" not in df.columns:
            continue
        sub = df[(df["云底高度m"] > 0) & (df["露点差°C"] > 0)]
        if sub.empty:
            continue
        for _, row in sub.iterrows():
            cb, dpv = row["云底高度m"], row["露点差°C"]
            r = cb / dpv
            if np.isfinite(r) and clip_min <= r <= clip_max:
                ratios.append(r)
                used += 1
                est_old = dpv * default_coef
                errs_old.append(abs(cb - est_old))

    if used < min_samples:
        return {"coef_used": default_coef, "coef_new": None, "n": used,
                "mae_old": float(np.mean(errs_old)) if errs_old else None,
                "mae_new": None, "reason": f"not_enough_samples({used}/{min_samples})"}

    coef_new = float(np.clip(np.median(ratios), clip_min, clip_max))

    # 新误差
    errs_new = []
    for f in files:
        try:
            df = pd.read_csv(f, encoding="utf-8-sig")
        except Exception:
            continue
        if "云底高度m" not in df.columns or "露点差°C" not in df.columns:
            continue
        sub = df[(df["云底高度m"] > 0) & (df["露点差°C"] > 0)]
        for _, row in sub.iterrows():
            cb, dpv = row["云底高度m"], row["露点差°C"]
            est_new = dpv * coef_new
            errs_new.append(abs(cb - est_new))

    return {"coef_used": coef_new, "coef_new": coef_new, "n": used,
            "mae_old": float(np.mean(errs_old)) if errs_old else None,
            "mae_new": float(np.mean(errs_new)) if errs_new else None,
            "reason": "auto_calibrated"}

# ----------------- 低云墙预警（多点模型） -----------------
def fallback_cloudwall_model(sun_hour):
    """
    与 sunrise_bot 同款：沿日落方向取多个距离点（默认 270°），
    读取低云覆盖率和云底高度，规则打分。
    """
    cfg = CONFIG.get("cloudwall", {})
    # 日落方向默认 270°，若你在 config 里放了 sunset_azimuth 就用那个
    bearing = cfg.get("sunset_azimuth", 270)
    dists   = cfg.get("sample_km", [20, 50, 80, 120])

    samples = []
    for d in dists:
        plat, plon = offset_latlon(LAT, LON, bearing, d)
        rec = mb_point_lowcloud(plat, plon, sun_hour)
        if rec is None:
            om = open_meteo(plat, plon)
            if om is None:
                samples.append((d, None, None, "none", "no_resp"))
                continue
            times = om["hourly"]["time"]
            tgt = sun_hour.strftime("%Y-%m-%dT%H:00")
            if tgt not in times:
                idx = min(range(len(times)),
                          key=lambda i: abs(dt.datetime.fromisoformat(times[i]) - sun_hour))
                flag = "nearest_hour"
            else:
                idx = times.index(tgt); flag = "exact_hour"
            low_pct = om["hourly"]["cloudcover_low"][idx]
            base_m  = None
            samples.append((d, low_pct, base_m, "om", flag))
        else:
            samples.append((d, rec.get("low_cloud"), rec.get("cloud_base"), "mb", "mb_point"))

    risk = model_lc_risk_v2(samples)
    txt  = risk_text_from_samples(risk, samples)
    return risk, txt

def model_lc_risk_v2(samples):
    if not samples:
        return 1
    high = sum(1 for _, l, b, _, _ in samples if (l is not None and l >= 50) and (b is None or b < 600))
    mid  = sum(1 for _, l, b, _, _ in samples if (l is not None and l >= 30) or (b is not None and b < 800))
    if high >= 1 or mid >= len(samples) * 0.5:
        return 2
    if mid >= 1:
        return 1
    return 0

def risk_text_from_samples(risk, samples):
    stat = {0:"正常(模型)",1:"关注(模型)",2:"预警(模型)"}.get(risk,"?")
    detail = []
    for d,l,b,src,why in samples:
        lp = f"{l:.0f}%" if l is not None else "NA%"
        bm = f"{int(b)}m" if (b is not None and np.isfinite(b)) else "NA m"
        detail.append(f"{d}km:{lp} / {bm}[{src}:{why}]")
    return f"{stat}（samples: " + " | ".join(detail) + "）"

# ---- 简单风险（保底） ----
def model_lc_risk_simple(lc, dp, wind):
    if lc is None:
        return 1
    if lc >= 50 and dp < 2:
        return 2
    if lc >= 30:
        return 1
    return 0

RISK_MAP = {0: "正常", 1: "关注", 2: "高风险"}

# ----------------- 评分逻辑 -----------------
def score_value(v, bounds):
    if v is None:
        return 1
    if len(bounds) == 3:
        lo, mid, hi = bounds
        if lo <= v <= mid: return 2
        if (mid < v <= hi) or (0 <= v < lo): return 1
        return 0
    else:
        lo, hi = bounds
        if v >= hi: return 2
        if v >= lo: return 1
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
    if lo2 <= w <= hi2: pt = 2
    elif lo1 <= w < lo2 or hi2 < w <= hi3: pt = 1
    else: pt = 0
    detail.append(("风速m/s", w, pt)); total += pt

    dp = vals["t"] - vals["td"]
    lo, hi = cfg["dewpoint_diff"]
    pt = 2 if dp >= hi else 1 if dp >= lo else 0
    detail.append(("露点差°C", dp, pt)); total += pt

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
    lc   = kv.get("低云%",      0) or 0
    mh   = kv.get("中/高云%",    0) or 0
    cb   = kv.get("云底高度m",   -1)
    vis  = kv.get("能见度km",    0) or 0
    wind = kv.get("风速m/s",     0) or 0
    dp   = kv.get("露点差°C",    0) or 0
    rp   = kv.get("降雨量mm",    0) or 0

    if lc < 20:   lc_level, low_text = "低","地平线基本通透，太阳能“蹦”出来"
    elif lc < 40: lc_level, low_text = "中","地平线可能有一条灰带，太阳或从缝隙钻出"
    elif lc < 60: lc_level, low_text = "偏高","低云偏多，首轮日光可能被挡一部分"
    else:         lc_level, low_text = "高","一堵低云墙，首轮日光大概率看不到"

    if 20 <= mh <= 60: mh_level, fire_text = "适中","有“云接光”舞台，可能染上粉橙色（小概率火烧云）"
    elif mh < 20:      mh_level, fire_text = "太少","天空太干净，只有简单渐变色"
    elif mh <= 80:     mh_level, fire_text = "偏多","云多且厚，色彩可能偏闷"
    else:              mh_level, fire_text = "很多","厚云盖顶，大概率阴沉"

    if cb is None or cb < 0:
        cb_level, cb_text, cb_show = "未知","云底数据缺失，可参考下午实况","未知"
    elif cb > 1000:
        cb_level, cb_text, cb_show = ">1000m","云底较高，多当“天花板”，不挡地平线",f"{cb:.0f}m"
    elif cb > 500:
        cb_level, cb_text, cb_show = "500~1000m","可能在远处形成一道云棚，注意角度",f"{cb:.0f}m"
    else:
        cb_level, cb_text, cb_show = "<500m","贴地低云/雾，像拉了窗帘",f"{cb:.0f}m"

    if vis >= 15: vis_level, vis_text = ">15km","空气透明度好，远景清晰，金光反射漂亮"
    elif vis >= 8: vis_level, vis_text = "8~15km","中等透明度，远景略灰"
    else:          vis_level, vis_text = "<8km","背景灰蒙蒙，层次差"

    if 2 <= wind <= 5: wind_level, wind_text = "2~5m/s","海面有微波纹，反光好，三脚架稳"
    elif wind < 2:     wind_level, wind_text = "<2m/s","几乎无风，注意镜头易结露"
    elif wind <= 8:    wind_level, wind_text = "5~8m/s","风稍大，留意三脚架稳定性"
    else:              wind_level, wind_text = ">8m/s","大风天，拍摄体验差，器材要护好"

    if dp >= 3:   dp_level, dp_text = "≥3℃","不易起雾"
    elif dp >= 1: dp_level, dp_text = "1~3℃","稍潮，镜头可能结露"
    else:         dp_level, dp_text = "<1℃","极易起雾，注意镜头/云雾风险"

    if rp < 0.1:   rp_level, rain_text = "<0.1mm","几乎不会下雨"
    elif rp < 1:   rp_level, rain_text = "0.1~1mm","可能有零星小雨/毛毛雨"
    else:          rp_level, rain_text = "≥1mm","有下雨可能，注意防水"

    if score5 >= 4.0:   grade = "建议出发（把握较大）"
    elif score5 >= 3.0: grade = "可去一搏（不稳）"
    elif score5 >= 2.0: grade = "机会一般（看心情或距离）"
    elif score5 >= 1.0: grade = "概率很小（除非就在附近）"
    else:               grade = "建议休息（基本无戏）"

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

# ----------------- 主流程 -----------------
def run_sunset():
    sun_exact, sun_hour = sunset_time()

    # 1. 主体数据
    om = open_meteo()
    if om is None:
        msg = "[ERR] open-meteo 数据为空，无法评分（日落）。"
        print(msg); save_report("sunset_error", msg)
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

    # 2. 云底高度：METAR or 估算
    metar_cb = parse_cloud_base(metar("ZGSZ"))
    dp_now   = vals["t"] - vals["td"]

    calib = auto_calibrate_coef()
    coef_used = calib["coef_used"]
    est_cb = estimate_cloud_base_from_dp(dp_now, coef_used)
    cb_final = metar_cb if metar_cb is not None else est_cb

    # 3. 总分 + 文案
    total, det = calc_score(vals, cb_final, CONFIG["scoring"])
    score5 = round(total / (3 * len(det)) * 5, 1)
    kv = {k: v for k, v, _ in det}

    # 4. 风险
    risk_simple = model_lc_risk_simple(vals["low"], dp_now, vals["wind"])
    risk_simple_text = f"{RISK_MAP[risk_simple]}（模型12h）"

    risk_model, risk_model_text = fallback_cloudwall_model(sun_hour)

    mae_old_s = fmt(calib.get("mae_old"), ".0f")
    mae_new_s = fmt(calib.get("mae_new"), ".0f")
    metar_cb_s = fmt(metar_cb, ".0f")
    est_cb_s   = fmt(est_cb, ".0f")

    scene_txt = (
        gen_scene_desc(score5, kv, sun_exact, event_name="日落")
        + f"\n- 低云墙风险（模型12h）：{risk_simple_text}"
        + f"\n- 低云墙预警（模型多点）：{risk_model_text}"
        + f"\n- 云底估算参数：coef={coef_used:.1f} m/℃（样本={calib['n']}, 原MAE={mae_old_s}m, 新MAE={mae_new_s}m, 原因={calib['reason']}）"
        + f"\n  ※ METAR云底={metar_cb_s}m，估算云底={est_cb_s}m"
    )

    text = scene_txt + "\n\n" + build_forecast_text(total, det, sun_exact, extra={}, event_name="日落")
    print(text)
    save_report("sunset", text)

    # 5. 写日志
    log_csv(CONFIG["paths"]["log_scores"], {
        "time": now(), "mode": "sunset", "score": total, "score5": score5,
        **{k: v for k, v, _ in det},
        "risk_model_simple": risk_simple,
        "risk_model_multi": risk_model,
        "metar_cb": metar_cb if metar_cb is not None else np.nan,
        "est_cb": est_cb if est_cb is not None else np.nan,
        "coef_used": coef_used,
        "calib_samples": calib["n"],
        "calib_reason": calib["reason"]
    })
    log_csv(CONFIG["paths"]["log_calib"], {"time": now(), **calib})

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
