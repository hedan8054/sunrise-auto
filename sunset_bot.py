#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
sunset_bot.py
日落版，逻辑与 sunrise 相同，仅事件时间与文案关键字不同。
"""

import os, re, yaml, math, datetime as dt
import requests, pandas as pd, numpy as np
import pytz

TZ = pytz.timezone("Asia/Shanghai")
CONFIG = yaml.safe_load(open("config.yaml", "r", encoding="utf-8"))
LAT = CONFIG["location"]["lat"]
LON = CONFIG["location"]["lon"]
PLACE = CONFIG["location"]["name"]
SC = CONFIG["scoring"]
PATHS = CONFIG.get("paths", {})
CB_CFG = CONFIG.get("cloudbase_estimate", {
    "default_coef_m_per_deg": 125,
    "min_samples": 10,
    "max_window_days": 45,
    "clip_min": 60,
    "clip_max": 200
})

def now(): return dt.datetime.now(TZ)
def ensure_dirs():
    os.makedirs("logs", exist_ok=True); os.makedirs("out", exist_ok=True)
def log_csv(path_tpl, row):
    ensure_dirs(); path = now().strftime(path_tpl)
    header = not os.path.exists(path)
    pd.DataFrame([row]).to_csv(path, mode="a", header=header, index=False, encoding="utf-8-sig")
def save_report(name, text):
    ensure_dirs(); fname = now().strftime(f"out/{name}_%Y-%m-%d_%H%M.txt")
    open(fname, "w", encoding="utf-8").write(text)

def api_open_meteo():
    url = ("https://api.open-meteo.com/v1/forecast"
           f"?latitude={LAT}&longitude={LON}"
           "&hourly=cloudcover_low,cloudcover_mid,cloudcover_high,visibility,"
           "temperature_2m,dewpoint_2m,windspeed_10m,precipitation,winddirection_10m"
           "&forecast_days=2&timezone=Asia%2FShanghai")
    r = requests.get(url, timeout=30); r.raise_for_status()
    js = r.json()
    if "hourly" not in js: raise RuntimeError("open-meteo 缺 hourly")
    return js

def api_metar(code="ZGSZ"):
    try:
        url = f"https://tgftp.nws.noaa.gov/data/observations/metar/stations/{code}.TXT"
        return requests.get(url, timeout=20).text.strip().splitlines()[-1]
    except Exception: return ""

def parse_metar_cloud_base_m(m):
    if not m: return None
    f = re.findall(r'(BKN|OVC)(\d{3})', m)
    if not f: return None
    return int(f[0][1])*100*0.3048

def sunset_time():
    def _call(date_str):
        js = requests.get(
            f"https://api.sunrise-sunset.org/json?lat={LAT}&lng={LON}&date={date_str}&formatted=0",
            timeout=30).json()
        return dt.datetime.fromisoformat(js["results"]["sunset"]).astimezone(TZ)
    try:
        t = _call("today")
        if t < now(): t = _call("tomorrow")
    except Exception:
        t = now().replace(hour=18, minute=30, second=0, microsecond=0)
    return t, t.replace(minute=0, second=0, microsecond=0)

def estimate_cloud_base_m(t, td, coef=125, clip_min=60, clip_max=2000):
    if t is None or td is None: return None
    return float(np.clip((t-td)*coef, clip_min, clip_max))

def score_value(v, bounds):
    if v is None: return 1
    if len(bounds)==3:
        lo, mid, hi = bounds
        if lo<=v<=mid: return 2
        if (mid<v<=hi) or (0<=v<lo): return 1
        return 0
    lo, hi = bounds
    if v>=hi: return 2
    if v>=lo: return 1
    return 0

def sector_lowcloud_weight(low, wdir):
    if low is None: return 1.0
    if low<40 or low>70: return 1.0
    w=0.92
    if wdir is not None and (60<=wdir<=120 or 240<=wdir<=300):
        w=0.88
    return w

def cloud_fragmentation(low, mid, high):
    arr=[x for x in [low,mid,high] if x is not None]
    if len(arr)<2: return 0.0
    return float(np.clip(np.std(arr)/50.0, 0.0, 1.0))

def model_lc_risk(lc, dp):
    if lc is None: return 1
    if lc>=50 and dp<2: return 2
    if lc>=30: return 1
    return 0

RISK_MAP={0:"正常",1:"关注",2:"高风险"}

def scene_labels(low, mid, high, cb_m, vis_km, wind, dp, sector_w, frag):
    labels=[]
    if (low or 0) < 20*sector_w: labels.append(("地平线通透","太阳能直接落到海面，首轮光干净"))
    elif (low or 0) < 40*sector_w: labels.append(("低云镶边","低云边缘勾出金边"))
    elif (low or 0) < 70*sector_w and frag>=0.3: labels.append(("破口日落","低云有缝隙，等太阳穿缝"))
    else: labels.append(("低云墙","首轮落日大概率被挡"))

    mh=max(mid or 0, high or 0)
    if 20<=mh<=60: labels.append(("云接光","中高云有层次，粉橙色概率"))
    elif mh<20: labels.append(("晴幕渐变","天空简洁，层次弱"))
    else: labels.append(("厚盖顶","云厚偏闷"))

    if cb_m is None: labels.append(("云底未知","参考傍晚实况"))
    elif cb_m>1000: labels.append(("高云底天花板","不挡地平线"))
    elif cb_m>500: labels.append(("远处云棚","可能在远处形成棚云"))
    else: labels.append(("贴海低云/雾","像拉了窗帘"))

    if (vis_km or 0)>=15: labels.append(("远景通透","反射与层次好"))
    if 2<= (wind or 0) <=5: labels.append(("微波反光","海面细碎反光好看"))
    if (dp or 0) <1: labels.append(("结露风险","镜头易起雾"))
    if 40<= (low or 0) <=70 and frag>=0.35: labels.append(("天窗机率","碎片化低云易出缝"))

    seen=set(); out=[]
    for k,v in labels:
        if k not in seen:
            out.append((k,v)); seen.add(k)
    return out

def human_text(score5, kv, sun_t, labels, event_name="日落"):
    lc   = kv.get("低云%",0) or 0
    mh   = kv.get("中/高云%",0) or 0
    cb   = kv.get("云底高度m",-1)
    vis  = kv.get("能见度km",0) or 0
    wind = kv.get("风速m/s",0) or 0
    dp   = kv.get("露点差°C",0) or 0
    rp   = kv.get("降雨量mm",0) or 0

    if score5>=4: grade="建议出发（把握较大）"
    elif score5>=3: grade="可去一搏（不稳）"
    elif score5>=2: grade="机会一般（看心情或距离）"
    elif score5>=1: grade="概率很小（除非就在附近）"
    else: grade="建议休息（基本无戏）"

    lines=[
        f"【直观判断】评分：{score5:.1f}/5 —— {grade}",
        f"{event_name}：{sun_t:%H:%M}",
        f"- 低云：{lc:.0f}% —— "+("地平线基本通透，太阳能直接落下" if lc<20 else
                              "低云偏多，首轮落日可能被挡一部分" if lc<60 else
                              "一堵低云墙，首轮落日大概率看不到"),
        f"- 中/高云：{mh:.0f}% —— "+("天空太干净，只有简单渐变色" if mh<20 else
                                   "有“云接光”舞台，可能染上粉橙色" if mh<=60 else
                                   "厚云盖顶，大概率阴沉"),
        f"- 云底高度：{('未知' if cb is None or cb<0 else f'{cb:.0f}m')} —— "+(
            "云底数据缺失，可参考傍晚实况" if cb is None or cb<0 else
            "贴海低云/雾，像拉了窗帘" if cb<500 else
            "可能在远处形成一道云棚" if cb<=1000 else
            "云底较高，多当“天花板”"),
        f"- 能见度：{vis:.1f} km —— "+("空气透明度好，远景清晰，金光反射漂亮" if vis>=15 else
                                    "中等透明度，远景略灰" if vis>=8 else
                                    "背景灰蒙蒙，层次差"),
        f"- 风速：{wind:.1f} m/s —— "+("海面有微波纹，反光好，三脚架稳" if 2<=wind<=5 else
                                   "几乎无风，注意镜头容易结露" if wind<2 else
                                   "风稍大，留意三脚架稳定性" if wind<=8 else
                                   "大风天，拍摄体验差，器材要护好"),
        f"- 降雨：{rp:.1f} mm —— "+("几乎不会下雨" if rp<0.1 else
                                  "可能有零星小雨/毛毛雨" if rp<1 else
                                  "有下雨可能，注意防水和收纳镜头"),
        f"- 露点差：{dp:.1f} ℃ —— "+("不易起雾" if dp>=3 else
                                  "稍潮，镜头可能结露" if dp>=1 else
                                  "极易起雾，注意海雾/镜头起雾风险")
    ]

    if labels:
        lines.append("")
        lines.append("【场景标签】" + "、".join([t for t,_ in labels]))
        anns=[d for _,d in labels[:2]]
        if anns: lines.append("—— " + "；".join(anns))
    return "\n".join(lines)

def run_sunset():
    sun_exact, sun_hour = sunset_time()
    om = api_open_meteo()
    hrs = om["hourly"]["time"]
    target = sun_hour.strftime("%Y-%m-%dT%H:00")
    if target in hrs: idx = hrs.index(target)
    else: idx = min(range(len(hrs)), key=lambda i: abs(dt.datetime.fromisoformat(hrs[i]) - sun_hour))

    low  = om["hourly"]["cloudcover_low"][idx]
    mid  = om["hourly"]["cloudcover_mid"][idx]
    high = om["hourly"]["cloudcover_high"][idx]
    vis  = om["hourly"]["visibility"][idx]
    t2m  = om["hourly"]["temperature_2m"][idx]
    td2m = om["hourly"]["dewpoint_2m"][idx]
    wind = om["hourly"]["windspeed_10m"][idx]
    wdir = om["hourly"].get("winddirection_10m", [None]*len(hrs))[idx]
    precip = om["hourly"]["precipitation"][idx]

    mtxt = api_metar("ZGSZ")
    cb_metar = parse_metar_cloud_base_m(mtxt)
    cb_est   = estimate_cloud_base_m(t2m, td2m, CB_CFG.get("default_coef_m_per_deg",125),
                                     CB_CFG.get("clip_min",60), 2000)
    cb_use   = cb_metar if cb_metar is not None else cb_est

    vals = dict(low=low, mid=mid, high=high, vis=vis, t=t2m, td=td2m, wind=wind, precip=precip)
    detail, total = [], 0

    sector_w = sector_lowcloud_weight(low, wdir)
    frag     = cloud_fragmentation(low, mid, high)
    low_adj  = (low or 0) * sector_w
    pt = score_value(low_adj, SC["low_cloud"]); detail.append(("低云%", low, pt)); total += pt

    mh = max(mid, high)
    pt = score_value(mh, SC["mid_high_cloud"]); detail.append(("中/高云%", mh, pt)); total += pt

    if cb_use is None: pt, v = 1, -1
    else:
        lo, hi = SC["cloud_base_m"]
        pt = 2 if cb_use>hi else 1 if cb_use>lo else 0
        v = cb_use
    detail.append(("云底高度m", v, pt)); total += pt

    vis_km = (vis or 0)/1000.0
    lo, hi = SC["visibility_km"]
    pt = 2 if vis_km>=hi else 1 if vis_km>=lo else 0
    detail.append(("能见度km", vis_km, pt)); total += pt

    w = wind or 0
    lo1, lo2, hi2, hi3 = SC["wind_ms"]
    if lo2<=w<=hi2: pt=2
    elif lo1<=w<lo2 or hi2<w<=hi3: pt=1
    else: pt=0
    detail.append(("风速m/s", w, pt)); total += pt

    dp = t2m - td2m
    lo, hi = SC["dewpoint_diff"]
    pt = 2 if dp>=hi else 1 if dp>=lo else 0
    detail.append(("露点差°C", dp, pt)); total += pt

    p = precip or 0
    lo, hi = SC["precip_mm"]
    pt = 2 if p<lo else 1 if p<hi else 0
    detail.append(("降雨量mm", p, pt)); total += pt

    bonus=0
    if 40 <= (low or 0) <= 70 and frag >= 0.35:
        bonus=1; total+=1

    score5 = round(total / (3*len(detail)) * 5, 1)
    kv = {k: v for k, v, _ in detail}

    risk_model = model_lc_risk(low, dp)
    labels = scene_labels(low, mid, high, cb_use, vis_km, wind, dp, sector_w, frag)

    scene_txt = human_text(score5, kv, sun_exact, labels, event_name="日落")
    extra_lines = [
        "",
        f"- 低云墙风险（模型12h）：{RISK_MAP[risk_model]}（模型12h）",
        f"- 扇区低云权重：x{sector_w:.2f}；碎片度：{frag:.2f}；天窗加分：{bonus}",
        f"- 云底估算参数：coef={CB_CFG.get('default_coef_m_per_deg',125)} m/℃",
        f"  ※ METAR云底={'NA' if cb_metar is None else f'{cb_metar:.0f}m'}，估算云底={'NA' if cb_est is None else f'{cb_est:.0f}m'}",
        "",
        f"【日落预报 | 今晚 {sun_exact:%m月%d日}】拍摄指数：{total}/{3*len(detail)}",
        f"地点：{PLACE}  (lat={LAT}, lon={LON})",
        f"日落：{sun_exact:%H:%M}",
        ""
    ]
    for n,v,pt in detail:
        if isinstance(v, float):
            extra_lines.append(f"- {n}: {v:.1f} → {pt}分")
        else:
            extra_lines.append(f"- {n}: {v} → {pt}分")

    text = scene_txt + "\n" + "\n".join(extra_lines)
    print(text)
    save_report("sunset", text)
    log_csv(PATHS.get("log_scores","logs/scores_%Y-%m.csv"), {
        "time": now(), "mode": "sunset", "score": total, "score5": score5,
        **{k: v for k, v, _ in detail},
        "sector_w": sector_w, "frag": frag, "risk_model": risk_model
    })

if __name__ == "__main__":
    ensure_dirs()
    try:
        run_sunset()
    except Exception as e:
        err=f"[FATAL] sunset 运行异常：{repr(e)}"
        print(err); save_report("sunset_error", err)
        log_csv(PATHS.get("log_scores","logs/scores_%Y-%m.csv"), {
            "time": now(), "mode": "sunset", "score": -1, "error": repr(e)
        })
