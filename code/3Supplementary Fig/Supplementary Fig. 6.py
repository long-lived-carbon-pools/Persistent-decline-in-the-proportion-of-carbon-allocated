# -*- coding: utf-8 -*-
import os, glob, numpy as np, pandas as pd, xarray as xr, matplotlib.pyplot as plt
from scipy.stats import pearsonr

# ======== 可调参数区域 ========
PAD = 0.2          # 图表左右额外空白比例
# 子图1（左侧）：n、r、星号的偏移量
L_N_OFF = 0.015      # n= 偏移
L_R_OFF = 0.012      # r 偏移
L_STAR_OFF = 0.08    # 星号偏移
# 子图2（右侧）：星号、r、n的偏移量
R_STAR_OFF = 0.15
R_R_OFF = 0.08
R_N_OFF = 0.08
# ============================

site_csv1 = r"data/3Supplementary Fig/Supplementary Fig. 6/1.csv"
obs_csv1  = r"data/3Supplementary Fig/Supplementary Fig. 6/2.csv"
csv_data2 = r"data/3Supplementary Fig/Supplementary Fig. 6/3.csv"
coord_csv2= r"data/3Supplementary Fig/Supplementary Fig. 6/4.csv"
nc_dir    = r"data/6Sample data/npp2" 
out_dir   = r"results/3Supplementary Fig"
os.makedirs(out_dir, exist_ok=True)
out_png   = os.path.join(out_dir, "Supplementary Fig. 6.png")

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["xtick.labelsize"] = 16  # 横坐标刻度数字
plt.rcParams["axes.labelsize"] = 16    # 坐标轴标题

def stars(p): 
    return "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))

def set_xlim_pad(ax, rvals, pad=PAD, rev=False):
    rmin, rmax = float(np.nanmin(rvals)), float(np.nanmax(rvals))
    if rmin == rmax: rmin -= 0.5; rmax += 0.5
    rng = rmax - rmin
    x0, x1 = rmin - pad * rng, rmax + pad * rng
    ax.set_xlim((x1, x0) if rev else (x0, x1))
    return abs(x1 - x0)

# ======== 计算结果：从低到高排序 ========
def compute_results_pooled():
    sites = pd.read_csv(site_csv1)
    obs = pd.read_csv(obs_csv1).replace([-9999, -9999.0], np.nan)
    years = obs["year"].astype(int)
    rows = []
    for fp in sorted(glob.glob(os.path.join(nc_dir, "*.nc"))):
        fn = os.path.basename(fp)
        if fn.startswith("4_"): continue
        try: ds = xr.open_dataset(fp)
        except: continue
        if "npp" not in ds.variables or "year" not in ds.coords:
            ds.close(); continue
        npp = ds["npp"].sel(year=slice(int(years.min()), int(years.max())))
        xs, ys = [], []
        for _, r in sites.iterrows():
            sid = r.get("Site ID")
            if sid not in obs.columns: continue
            try: ts = npp.sel(latitude=r["Lat"], longitude=r["Long"], method="nearest").to_pandas()
            except: continue
            df = pd.DataFrame({"year": years, "obs": obs[sid]}).merge(
                ts.rename("mod").reset_index(), on="year", how="left"
            ).replace([-9999, -9999.0], np.nan).dropna(subset=["obs", "mod"])
            if df.empty: continue
            xs.append(df["obs"].to_numpy(float)); ys.append(df["mod"].to_numpy(float))
        ds.close()
        if sum(map(len, xs)) < 10: continue
        x, y = np.concatenate(xs), np.concatenate(ys)
        r, p = pearsonr(x, y)
        rows.append({"Model": os.path.splitext(fn)[0], "r": r, "p": p, "n": len(x)})
    df = pd.DataFrame(rows).sort_values("r", ascending=True).reset_index(drop=True)  # 从低到高
    df["sig"] = df["p"].apply(stars)
    return df

def compute_results_window():
    lon_min, lon_max = -167.27, -55.00; lat_min, lat_max = 9.58, 74.99
    d = pd.read_csv(csv_data2, usecols=["site_ID", "begin_year", "end_year", "NPP_tot1"])
    xy = pd.read_csv(coord_csv2, usecols=["site_ID", "latitude", "longitude"])
    d = pd.merge(d, xy, on="site_ID").dropna(subset=["latitude", "longitude", "NPP_tot1"])
    d = d[(d["longitude"] >= lon_min) & (d["longitude"] <= lon_max) &
          (d["latitude"] >= lat_min) & (d["latitude"] <= lat_max)]
    d["begin_year"] = pd.to_numeric(d["begin_year"], errors="coerce")
    d["end_year"] = pd.to_numeric(d["end_year"], errors="coerce")
    d = d.dropna(subset=["begin_year", "end_year"])
    rows = []
    for fp in sorted(glob.glob(os.path.join(nc_dir, "*.nc"))):
        fn = os.path.basename(fp)
        if fn.startswith("4_"): continue
        try: ds = xr.open_dataset(fp)
        except: continue
        if not {"npp", "year"}.issubset(ds.variables):
            ds.close(); continue
        npp = ds["npp"]; yrs = ds["year"].values
        xs, ys = [], []
        for _, r in d.iterrows():
            y0, y1 = int(r["begin_year"]), int(r["end_year"])
            m = (yrs >= y0) & (yrs <= y1)
            if not np.any(m): continue
            try: val = float(npp.sel(latitude=float(r["latitude"]), longitude=float(r["longitude"]),
                                     method="nearest").sel(year=yrs[m]).mean().item())
            except: continue
            if np.isfinite(val) and np.isfinite(r["NPP_tot1"]):
                xs.append(float(r["NPP_tot1"])); ys.append(val)
        ds.close()
        if len(xs) >= 3:
            r, p = pearsonr(xs, ys)
            rows.append({"Model": os.path.splitext(fn)[0], "r": r, "p": p, "n": len(xs)})
    df = pd.DataFrame(rows).sort_values("r", ascending=True).reset_index(drop=True)  # 从低到高
    df["sig"] = df["p"].apply(stars)
    return df

# ===== 绘制左图 =====
def draw_left(ax, df, xlabel):
    y = np.arange(len(df)); bars = ax.barh(y, df["r"], edgecolor="black")
    ax.invert_yaxis(); ax.set_yticks(y); ax.set_yticklabels(df["Model"]); ax.set_xlabel(xlabel)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False); ax.xaxis.grid(True, linestyle="--", alpha=0.5)
    span = set_xlim_pad(ax, df["r"].values, pad=PAD, rev=False)
    for yi, b, (n, r, sig) in zip(y, bars, df[["n", "r", "sig"]].to_numpy()):
        w = b.get_width()
        ax.text(w - L_N_OFF * span, yi, f"n={n}", va="center", ha="right", fontsize=9)
        ax.text(w + L_R_OFF * span, yi, f"{r:.2f}", va="center", ha="left")
        if sig: ax.text(w + (L_R_OFF + L_STAR_OFF) * span, yi, sig, va="center", ha="left", color="red")

# ===== 绘制右图 =====
def draw_right(ax, df, xlabel):
    y = np.arange(len(df)); bars = ax.barh(y, df["r"], edgecolor="black")
    ax.invert_yaxis(); ax.set_yticks(y); ax.set_yticklabels(df["Model"]); ax.yaxis.tick_right(); ax.tick_params(labelleft=False, labelright=True)
    ax.spines["top"].set_visible(False); ax.spines["left"].set_visible(False); ax.set_xlabel(xlabel); ax.xaxis.grid(True, linestyle="--", alpha=0.5)
    span = set_xlim_pad(ax, df["r"].values, pad=PAD, rev=True)
    for yi, b, (n, r, sig) in zip(y, bars, df[["n", "r", "sig"]].to_numpy()):
        w = b.get_width()
        if sig: ax.text(w + R_STAR_OFF * span, yi, sig, va="center", ha="left", color="red")
        ax.text(w + R_R_OFF * span, yi, f"{r:.2f}", va="center", ha="left")
        ax.text(w - R_N_OFF * span, yi, f"n={n}", va="center", ha="right", fontsize=9)

# ===== 主程序 =====
df1 = compute_results_pooled()
df2 = compute_results_window()
h = max(6, len(df1)*0.45, len(df2)*0.45)
fig, axes = plt.subplots(1, 2, figsize=(12, h), sharey=True, gridspec_kw={"wspace": 0.08})
draw_left(axes[0], df1, "R(Flux tower & NPP products)")
draw_right(axes[1], df2, "R(NPP site & NPP products)")
fig.tight_layout()
fig.savefig(out_png, dpi=600, bbox_inches="tight")
plt.close(fig)
print(f"已保存: {out_png}")
