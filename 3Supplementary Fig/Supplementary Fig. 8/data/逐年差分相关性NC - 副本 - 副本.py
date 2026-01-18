# -*- coding: utf-8 -*-
import os, re, numpy as np, xarray as xr

a_path = r"D:\1700\3\tree_ring4 - AgeDepSpline(EPS0.85).nc"
b_dir  = r"H:\Climatic data\6"
out_nc = r"D:\1700\5.1\climate_NC\10climate_corr_maps_1981_2010.nc"
yr0, yr1 = 1981, 2010
os.makedirs(os.path.dirname(out_nc), exist_ok=True)

def ensure_latlon(ds):
    if "lat" in ds.coords: ds = ds.rename({"lat":"latitude"})
    if "lon" in ds.coords: ds = ds.rename({"lon":"longitude"})
    return ds

def ensure_year(ds):
    if "year" in ds.coords: return ds
    if "time" in ds.coords:
        ds = ds.rename({"time":"year"})
        if np.issubdtype(ds["year"].dtype, np.datetime64):
            ds["year"] = xr.DataArray(ds["year"].dt.year.values, dims="year")
        elif np.issubdtype(ds["year"].dtype, np.timedelta64):
            days = (ds["year"] / np.timedelta64(1,"D")).astype("float32")
            ds = ds.assign_coords(year=days)
    return ds

def sel_year(ds):
    if "year" not in ds.coords: raise ValueError("缺少year坐标")
    ds = ds.sel(year=slice(yr0, yr1))
    ds["year"] = ds["year"].astype(np.int32)
    return ds

def make_numeric(da):
    if np.issubdtype(da.dtype, np.timedelta64):
        return (da / np.timedelta64(1,"D")).astype("float32")
    if np.issubdtype(da.dtype, np.datetime64):
        t0 = np.datetime64(f"{yr0}-01-01")
        return ((da - t0) / np.timedelta64(1,"D")).astype("float32")
    if da.dtype.kind not in "iuUf": 
        try:
            return da.astype("float32")
        except:
            return None
    return da.astype("float32")

def corr_time(x, y, min_n=3):
    valid = (~np.isnan(x)) & (~np.isnan(y))
    n = np.sum(valid, axis=0)
    x = np.where(valid, x, np.nan)
    y = np.where(valid, y, np.nan)
    xm = x - np.nanmean(x, axis=0, keepdims=True)
    ym = y - np.nanmean(y, axis=0, keepdims=True)
    num = np.nansum(xm * ym, axis=0)
    den = np.sqrt(np.nansum(xm * xm, axis=0) * np.nansum(ym * ym, axis=0))
    r = np.divide(num, den, out=np.full_like(num, np.nan, dtype=float), where=den>0)
    return np.where(n >= 10, r, np.nan)

def sanitize(name):
    s = re.sub(r'[^0-9A-Za-z_]+','_', str(name).strip())
    if re.match(r'^\d', s): s = "_" + s
    return s or "var"

with xr.open_dataset(a_path) as da:
    da = da.load()
da = ensure_latlon(da)
da = ensure_year(da)
da = sel_year(da)
if "AgeDepSpline" not in da: raise ValueError("A中缺少AgeDepSpline")
A = da["AgeDepSpline"].transpose("year","latitude","longitude").sortby("latitude").sortby("longitude")
lat_tgt, lon_tgt = A.latitude, A.longitude

vars_out = {}
for fname in os.listdir(b_dir):
    if not fname.lower().endswith(".nc"): continue
    fpath = os.path.join(b_dir, fname)
    try:
        with xr.open_dataset(fpath) as db:
            db = db.load()
        db = ensure_latlon(db)
        db = ensure_year(db)
        db = sel_year(db)
        for v in list(db.data_vars):
            da_b = db[v]
            if not {"year","latitude","longitude"}.issubset(set(da_b.dims)): continue
            B = da_b.transpose("year","latitude","longitude").sortby("latitude").sortby("longitude")
            B = make_numeric(B)
            if B is None: 
                print(f"[跳过变量-非数值] {fname} :: {v}")
                continue
            try:
                B = B.interp(latitude=lat_tgt, longitude=lon_tgt)
            except Exception as e:
                print(f"[跳过变量-插值失败] {fname} :: {v}: {e}")
                continue
            A1, B1 = xr.align(A, B, join="inner")
            if A1.sizes["year"] < 4: continue
            A1, B1 = A1.diff("year"), B1.diff("year")
            A1, B1 = xr.align(A1, B1, join="inner")
            if A1.sizes["year"] < 3: continue
            r = corr_time(A1.values, B1.values, min_n=3)
            vname = sanitize(os.path.splitext(fname)[0]) + "__" + sanitize(v)
            while vname in vars_out: vname += "_x"
            vars_out[vname] = (["latitude","longitude"], r)
        print(f"[完成] {fname}")
    except Exception as e:
        print(f"[跳过文件] {fname}: {e}")

if not vars_out: raise RuntimeError("没有可用变量生成结果")

ds_out = xr.Dataset(
    data_vars=vars_out,
    coords=dict(latitude=("latitude", lat_tgt.values.astype(np.float32)),
                longitude=("longitude", lon_tgt.values.astype(np.float32))),
    attrs=dict(description="逐像元逐年差分相关系数 r(ΔA,ΔB)，A=AgeDepSpline；B变量已插值并转为数值型", period=f"{yr0}-{yr1}")
)
encoding = {k:{"zlib":True,"complevel":4,"dtype":"float32","_FillValue":np.float32(np.nan)} for k in vars_out}
ds_out.to_netcdf(out_nc, encoding=encoding)
print(f"[输出] {out_nc} 变量数={len(vars_out)} 网格={lat_tgt.size}x{lon_tgt.size}")
