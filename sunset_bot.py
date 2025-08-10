#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sunset_bot.py  —  无卫星版 + 扇区云量/碎片度/岸线风向修正
每天 05:30 跑一次“日落预测”
"""

import os, sys, json, yaml, math, datetime as dt
import requests, pandas as pd, numpy as np
import pytz
import warnings, urllib3
warnings.filterwarnings("ignore", message="Unverified HTTPS request")
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

TZ = pytz.timezone("Asia/Shanghai")
CONFIG = yaml.safe_load(open("config.yaml", "r", encoding="utf-8"))
LAT = CONFIG["location"]["lat"]
LON = CONFIG["location"]["lon"]
LOC_NAME = CONFIG["location"]["name"]

COAST_NORMAL_DEG = CONFIG.get("coast_normal_deg", 90.0)
CB_COEF = float(CONFIG.get("cloudbase_coef_m_per_C", 125.0))
SECTOR_OFFSETS_DEG = CONFIG.get("sector_offsets_deg", [-20, -10, 0, 10, 20])
SECTOR_DISTS_KM    = CONFIG.get("sector_dists_km", [20, 40, 60])

def now(): return dt.datetime.now(TZ)
def ensure_dirs():
    os.makedirs("logs", exist_ok=True); os.makedirs("out", exist_ok=True)
def log_csv(path_tpl, row):
    ensure_dirs(); path = now().strftime(path_tpl)
    header = not os.path.exists(path)
    pd.DataFrame([row]).to_csv(path, mode="a", header=header, index=False, encoding="utf-8-sig")
def save_report(name, text):
    ensure_dirs(); fname = now().strftime(f"out/{name}_%Y-%m-%d_%H%M.txt")
    with open(fname, "w", encoding="utf-8") as f: f.write(text)

def sunset_time():
    """获取今日未过或明日日落（精确时间、整点）"""
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
    except Exception:
        t = now().replace(hour=18, minute=30, second=0, microsecond=0)
    return t, t.replace(minute=0, second=0, microsecond=0)

def open_meteo(lat=None, lon=None):
    lat = LAT if lat is None else lat; lon = LON if lon is None else lon
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        "&hourly=cloudcover_low,cloudcover_mid,cloudcover_high,visibility,"
        "temperature_2m,dewpoint_2m,windspeed_10m,winddirection_10m,precipitation"
        "&forecast_days=2&timezone=Asia%2FShanghai"
    )
    try:
        r = requests.get(url, timeout=30); r.raise_for_status()
        data = r.json()
        if "hourly" not in data or "time" not in data["hourly"]:
            print("[ERR] open-meteo hourly 字段缺失：", data); return None
        return data
    except Exception as e:
        print("[ERR] open-meteo 请求失败：", e); return None

def metar(code="ZGSZ"):
    try:
        url = f"https://tgftp.nws.noaa.gov/data/observations/metar/stations/{code}.TXT"
        txt = requests.get(url, timeout=20).text.strip().splitlines()[-1]
        return txt
    except Exception as e:
        print("[WARN] METAR 获取失败：", e); return ""

def parse_cloud_base(metar_txt):
    import re
    m = re.findall(r'(BKN|OVC)(\d{3})', metar_txt or "")
    if not m: return None
    ft = int(m[0][1]) * 100
    return ft * 0.3048

def offset_latlon(lat, lon, bearing_deg, dist_km):
    R=6371.0; br=math.radians(bearing_deg)
    lat1=math.radians(lat); lon1=math.radians(lon)
    lat2=math.asin(math.sin(lat1)*math.cos(dist_km/R)+math.cos(lat1)*math.sin(dist_km/R)*math.cos(br))
    lon2=lon1+math.atan2(math.sin(br)*math.sin(dist_km/R)*math.cos(lat1),
                         math.cos(lat1)*math.cos(dist_km/R)-math.sin(lat1)*math.sin(lat2))
    return math.degrees(lat2), math.degrees(lon2)

def ang_diff(a,b): return abs((a-b+180)%360-180)

def pick_hour_idx(om, target_dt):
    hrs = om["hourly"]["time"]; tgt = target_dt.strftime("%Y-%m-%dT%H:00")
    if tgt in hrs: return hrs.index(tgt), "exact_hour"
    idx = min(range(len(hrs)), key=lambda i: abs(dt.datetime.fromisoformat(hrs[i]) - target_dt))
    return idx, "nearest"

def sector_samples(target_hour, center_bearing_deg):
    samples=[]
    for d in SECTOR_DISTS_KM:
        for off in SECTOR_OFFSETS_DEG:
            plat,plon=offset_latlon(LAT,LON,center_bearing_deg+off,d)
            om=open_meteo(plat,plon)
            if om is None:
                samples.append((d, center_bearing_deg+off, None, None, "om:fail")); continue
            idx, tag = pick_hour_idx(om, target_hour)
            low = om["hourly"]["cloudcover_low"][idx]
            samples.append((d, center_bearing_deg+off, low, None, f"om:{tag}"))
    return samples

def sector_metrics(samples):
    lows=[x[2] for x in samples if isinstance(x[2],(int,float))]
    if not lows: return None, None, 0
    return float(np.mean(lows)), float(np.std(lows)), len(lows)

def shoreline_wind_bonus(wind_dir_deg, coast_normal_deg=COAST_NORMAL_DEG):
    if wind_dir_deg is None: return False
    return ang_diff(wind_dir_deg, coast_normal_deg) <= 30.0

def cloudwall_risk_from_samples(samples):
    if not samples: return 1, "关注(模型)（samples: none）"
    lows=[l for _,_,l,_,_ in samples if isinstance(l,(int,float))]
    if not lows: return 1, "关注(模型)（samples: no_valid_low）"
    hi=sum(1 for l in lows if l>=50); mid=sum(1 for l in lows if l>=30)
    if hi>=len(lows)*0.5: score=2; tag="预警(模型)"
    elif mid>=1:          score=1; tag="关注(模型)"
    else:                 score=0; tag="正常(模型)"
    detail=" | ".join([f"{int(d)}km:{(str(l)+'%') if l is not None else 'NA%'} [{src}]"
                       for d,_,l,_,src in samples])
    return score, f"{tag}（samples: {detail}）"

def score_value(v,bounds):
    if v is None: return 1
    if len(bounds)==3:
        lo,mid,hi=bounds
        if lo<=v<=mid: return 2
        if (mid<v<=hi) or (0<=v<lo): return 1
        return 0
    else:
        lo,hi=bounds
        if v>=hi: return 2
        if v>=lo: return 1
        return 0

def calc_score(vals, cloud_base_m, cfg):
    det=[]; total=0
    lc=vals["low"]; pt_low=score_value(lc, cfg["low_cloud"]); det.append(("低云%", lc, pt_low)); total+=pt_low
    mh=max(vals["mid"], vals["high"]); pt_mh=score_value(mh, cfg["mid_high_cloud"]); det.append(("中/高云%", mh, pt_mh)); total+=pt_mh
    if cloud_base_m is None: pt_cb,val_cb=1,-1
    else:
        lo,hi=cfg["cloud_base_m"]; pt_cb=2 if cloud_base_m>hi else 1 if cloud_base_m>lo else 0; val_cb=cloud_base_m
    det.append(("云底高度m", val_cb, pt_cb)); total+=pt_cb
    vis_km=(vals["vis"] or 0)/1000.0; lo,hi=cfg["visibility_km"]; pt_vis=2 if vis_km>=hi else 1 if vis_km>=lo else 0
    det.append(("能见度km", vis_km, pt_vis)); total+=pt_vis
    w=vals["wind"]; lo1,lo2,hi2,hi3=cfg["wind_ms"]
    pt_w=2 if lo2<=w<=hi2 else 1 if (lo1<=w<lo2 or hi2<w<=hi3) else 0
    det.append(("风速m/s", w, pt_w)); total+=pt_w
    dp=vals["t"]-vals["td"]; lo,hi=cfg["dewpoint_diff"]; pt_dp=2 if dp>=hi else 1 if dp>=lo else 0
    det.append(("露点差°C", dp, pt_dp)); total+=pt_dp
    p=vals.get("precip",0); lo,hi=cfg["precip_mm"]; pt_p=2 if p<lo else 1 if p<hi else 0
    det.append(("降雨量mm", p, pt_p)); total+=pt_p
    return total, det

def apply_window_bonuses(total, det, *,
                         sector_low_avg, sector_std, global_low,
                         vis_km, mid_high, cloud_base_m,
                         wind_dir_deg, shore_bonus_flag):
    idx_low = next(i for i,(n,_,_) in enumerate(det) if n=="低云%")
    low_name, low_val, low_pt = det[idx_low]
    reasons=[]
    if (low_val is not None and 40<=low_val<=70):
        cond_frag = (sector_std is not None and sector_std >= 12.0)
        cond_sector = (sector_low_avg is not None and global_low is not None and sector_low_avg < global_low*0.85)
        cond_shore = bool(shore_bonus_flag)
        if cond_frag or cond_sector or cond_shore:
            new_low_pt = min(2, low_pt+1)
            if new_low_pt != low_pt:
                reasons.append("低云40–70%且(碎片度高/扇区更低/岸线风向有利)")
                total += (new_low_pt-low_pt)
                det[idx_low]=(low_name, low_val, new_low_pt)
    if (mid_high is not None and 35<=mid_high<=60) and (vis_km is not None and vis_km>20):
        total += 0.5; reasons.append("中/高云35–60且能见度>20km(+0.5)")
    if cloud_base_m is not None and cloud_base_m>800:
        total += 0.5; reasons.append("云底>800m(+0.5)")
    return total, reasons

def build_forecast_text(total, det, sun_t, extra):
    lines=[f"拍摄指数：{total}/18", f"地点：{LOC_NAME}  (lat={LAT}, lon={LON})", f"日落：{sun_t:%H:%M}",""]
    for name,val,pts in det:
        if isinstance(val,float): lines.append(f"- {name}: {val:.1f} → {pts}分")
        else: lines.append(f"- {name}: {val} → {pts}分")
    if extra.get("note"): lines.append("\n提示："+extra["note"])
    return "\n".join(lines)

def gen_scene_desc(score5, kv, sun_t):
    lc=kv.get("低云%",0) or 0; mh=kv.get("中/高云%",0) or 0; cb=kv.get("云底高度m",-1)
    vis=kv.get("能见度km",0) or 0; wind=kv.get("风速m/s",0) or 0; dp=kv.get("露点差°C",0) or 0; rp=kv.get("降雨量mm",0) or 0
    if lc<20: lc_level="低"; low_text="地平线基本通透，太阳能“蹦”出来"
    elif lc<40: lc_level="中"; low_text="地平线可能有一条灰带，太阳或从缝隙钻出"
    elif lc<60: lc_level="偏高"; low_text="低云偏多，首轮日光可能被挡一部分"
    else: lc_level="高"; low_text="一堵低云墙，首轮日光大概率看不到"
    if 20<=mh<=60: mh_level="适中"; fire_text="有“云接光”舞台，可能染上粉橙色（小概率火烧云）"
    elif mh<20: mh_level="太少"; fire_text="天空太干净，只有简单渐变色"
    elif mh<=80: mh_level="偏多"; fire_text="云多且厚，色彩可能偏闷"
    else: mh_level="很多"; fire_text="厚云盖顶，大概率阴沉"
    if cb is None or cb<0: cb_level,cb_text,cb_show="未知","云底数据缺失，可参考临近实况","未知"
    elif cb>1000: cb_level,cb_text,cb_show=">1000m","云底较高，多当“天花板”",f"{cb:.0f}m"
    elif cb>500: cb_level,cb_text,cb_show="500~1000m","可能在海面上方形成一道云棚",f"{cb:.0f}m"
    else: cb_level,cb_text,cb_show="<500m","贴海低云/雾，像拉了窗帘",f"{cb:.0f}m"
    if vis>=15: vis_level,vis_text=">15km","空气透明度好，远景清晰，金光反射漂亮"
    elif vis>=8: vis_level,vis_text="8~15km","中等透明度，远景略灰"
    else: vis_level,vis_text="<8km","背景灰蒙蒙，层次差"
    if 2<=wind<=5: wind_level,wind_text="2~5m/s","海面有微波纹，反光好，三脚架稳"
    elif wind<2: wind_level,wind_text="<2m/s","几乎无风，注意镜头容易结露"
    elif wind<=8: wind_level,wind_text="5~8m/s","风稍大，留意三脚架稳定性"
    else: wind_level,wind_text=">8m/s","大风天，拍摄体验差，器材要护好"
    if dp>=3: dp_level,dp_text="≥3℃","不易起雾"
    elif dp>=1: dp_level,dp_text="1~3℃","稍潮，镜头可能结露"
    else: dp_level,dp_text="<1℃","极易起雾，注意海雾/镜头起雾风险"
    if rp<0.1: rp_level,rain_text="<0.1mm","几乎不会下雨"
    elif rp<1: rp_level,rain_text="0.1~1mm","可能有零星小雨/毛毛雨"
    else: rp_level,rain_text="≥1mm","有下雨可能，注意防水和收纳镜头"
    if score5>=4.0: grade="建议出发（把握较大）"
    elif score5>=3.0: grade="可去一搏（不稳）"
    elif score5>=2.0: grade="机会一般（看心情或距离）"
    elif score5>=1.0: grade="概率很小（除非就在附近）"
    else: grade="建议休息（基本无戏）"
    return (
        f"【直观判断】评分：{score5:.1f}/5 —— {grade}\n"
        f"日落：{sun_t:%H:%M}\n"
        f"- 低云：{lc:.0f}%（{lc_level}）— {low_text}\n"
        f"- 中/高云：{mh:.0f}%（{mh_level}）— {fire_text}\n"
        f"- 云底高度：{cb_show}（{cb_level}）— {cb_text}\n"
        f"- 能见度：{vis:.1f} km（{vis_level}）— {vis_text}\n"
        f"- 风速：{wind:.1f} m/s（{wind_level}）— {wind_text}\n"
        f"- 降雨：{rp:.1f} mm（{rp_level}）— {rain_text}\n"
        f"- 露点差：{dp:.1f} ℃（{dp_level}）— {dp_text}"
    )

def run_sunset():
    sun_exact, sun_hour = sunset_time()
    om = open_meteo()
    if om is None:
        msg = "[ERR] open-meteo 数据为空，无法评分（日落）。"
        print(msg); save_report("sunset_error", msg)
        log_csv(CONFIG["paths"]["log_scores"], {"time": now(), "mode":"sunset", "score":-1, "error":"open-meteo no data"})
        return
    idx, idx_tag = pick_hour_idx(om, sun_hour)
    vals=dict(
        low=om["hourly"]["cloudcover_low"][idx],
        mid=om["hourly"]["cloudcover_mid"][idx],
        high=om["hourly"]["cloudcover_high"][idx],
        vis=om["hourly"]["visibility"][idx],
        t=om["hourly"]["temperature_2m"][idx],
        td=om["hourly"]["dewpoint_2m"][idx],
        wind=om["hourly"]["windspeed_10m"][idx],
        winddir=om["hourly"].get("winddirection_10m",[None]*len(om["hourly"]["time"]))[idx],
        precip=om["hourly"]["precipitation"][idx]
    )
    cb_metar = parse_cloud_base(metar("ZGSZ"))
    cb_est = max(0.0,(vals["t"]-vals["td"])) * CB_COEF if (vals["t"] is not None and vals["td"] is not None) else None
    cloud_base_used = cb_metar if cb_metar is not None else cb_est

    center_bearing = float(CONFIG.get("cloudwall", {}).get("sunset_azimuth", 270.0))
    samples = sector_samples(sun_hour, center_bearing)
    sector_low_avg, sector_std, n_ok = sector_metrics(samples)
    risk_multi, risk_multi_text = cloudwall_risk_from_samples(samples)

    total, det = calc_score(vals, cloud_base_used, CONFIG["scoring"])
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

    score5 = round(total / (3*len(det)) * 5, 1)
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
    print(text); save_report("sunset", text)

    log_csv(CONFIG["paths"]["log_scores"], {
        "time": now(), "mode": "sunset", "score": total, "score5": score5,
        **{k: v for k, v, _ in det},
        "sector_low_avg": sector_low_avg, "sector_std": sector_std, "sector_n": n_ok,
        "risk_multi": risk_multi, "idx_tag": idx_tag
    })

if __name__ == "__main__":
    ensure_dirs()
    try:
        run_sunset()
    except Exception as e:
        err_msg=f"[FATAL] sunset 运行异常：{repr(e)}"
        print(err_msg); save_report("sunset_error", err_msg)
        log_csv(CONFIG["paths"]["log_scores"], {"time": now(), "mode":"sunset", "score":-1, "error":repr(e)})
