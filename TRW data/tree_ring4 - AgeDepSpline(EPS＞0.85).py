import os
import numpy as np
import pandas as pd
import xarray as xr
try:
    from pykrige.ok import OrdinaryKriging
except ImportError as e:
    raise ImportError("需要安装 PyKrige：pip install pykrige") from e
points_csv=r"D:\all\data\data\基本属性.csv"
series_csv=r"D:\all\data\data\AgeDepSpline.csv"
eps_dir=r"D:\all\data\data\Detrending4"
mask_nc_path=r"D:\all\data\recon_mask_40yr.nc"
out_nc_path=r"D:\all\data\tree_ring4 - AgeDepSpline(EPS0.85).nc"
variogram_model="spherical"
year_min,year_max=1700,2020
pts=pd.read_csv(points_csv,usecols=["name","latitude","longitude"]).dropna().reset_index(drop=True)
if pts.empty:
    raise ValueError("表1为空或缺列（需要 name, latitude, longitude）")
site_names=pts["name"].astype(str).to_numpy()
site_lats=pts["latitude"].to_numpy(dtype=float)
site_lons=pts["longitude"].to_numpy(dtype=float)
ts=pd.read_csv(series_csv)
ts=ts.rename(columns={ts.columns[0]:"year"}).set_index("year")
used_cols=[c for c in ts.columns if c in set(site_names)]
if not used_cols:
    raise ValueError("表2中没有与表1 name 匹配的列")
mat=ts[used_cols].to_numpy(dtype=np.float64)
mat[mat==0]=np.nan
years_full=ts.index.to_numpy(dtype=np.int32)
eps_mask=np.zeros_like(mat,dtype=bool)
def find_eps_path(name):
    p=os.path.join(eps_dir,f"{name}.csv")
    if os.path.isfile(p):
        return p
    names_lower={f.lower():f for f in os.listdir(eps_dir) if f.lower().endswith(".csv")}
    key=f"{name}.csv".lower()
    if key in names_lower:
        return os.path.join(eps_dir,names_lower[key])
    cand=[f for f in os.listdir(eps_dir) if f.lower().endswith(".csv") and os.path.splitext(f)[0].lower()==name.lower()]
    if cand:
        return os.path.join(eps_dir,cand[0])
    cand=[f for f in os.listdir(eps_dir) if f.lower().endswith(".csv") and name.lower() in os.path.splitext(f)[0].lower()]
    if cand:
        return os.path.join(eps_dir,cand[0])
    return None
for j,name in enumerate(used_cols):
    p=find_eps_path(name)
    if p is None:
        continue
    df=pd.read_csv(p,usecols=["year","EPS"])
    df=df.dropna(subset=["year","EPS"])
    df["year"]=df["year"].astype(int)
    ok_years=df.loc[df["EPS"]>=0.85,"year"].to_numpy(dtype=np.int32)
    if ok_years.size==0:
        continue
    pos=np.searchsorted(years_full,ok_years)
    hit=(pos<len(years_full))&(years_full[pos]==ok_years)
    idx=pos[hit]
    eps_mask[idx,j]=True
mat[~eps_mask]=np.nan
col_index={nm:j for j,nm in enumerate(used_cols)}
site_col=np.array([col_index.get(nm,-1) for nm in site_names],dtype=int)
ds_mask=xr.open_dataset(mask_nc_path)
if "recon_mask" not in ds_mask:
    raise ValueError("掩膜文件中找不到变量 recon_mask")
var_mask=ds_mask["recon_mask"]
mask_years=ds_mask.coords["year"].values.astype(np.int32)
lat_grid=ds_mask.coords["latitude"].values.astype(np.float32)
lon_grid=ds_mask.coords["longitude"].values.astype(np.float32)
common_years,idx_data,idx_mask=np.intersect1d(years_full,mask_years,return_indices=True)
if common_years.size==0:
    raise ValueError("掩膜年份与数据年份无交集")
keep=(common_years>=year_min)&(common_years<=year_max)
common_years=common_years[keep]
idx_data=idx_data[keep]
idx_mask=idx_mask[keep]
if common_years.size==0:
    raise ValueError("筛选后没有年份")
mask_arr=var_mask.isel({"year":idx_mask}).values.astype(np.float32)
ny,nx=lat_grid.size,lon_grid.size
out_stack=np.full((len(common_years),ny,nx),np.nan,dtype=np.float32)
regions=[(-11.0,57.0,30.0,75.0),(-167.27,-55.00,9.58,74.99)]
region_idx=[]
for (lon_min,lon_max,lat_min,lat_max) in regions:
    lon_sel=(lon_grid>=lon_min)&(lon_grid<=lon_max)
    lat_sel=(lat_grid>=lat_min)&(lat_grid<=lat_max)
    region_idx.append((lat_sel,lon_sel) if (lon_sel.any() and lat_sel.any()) else None)
for ii,y_idx in enumerate(idx_data):
    v=np.full(site_names.size,np.nan,dtype=np.float32)
    for s in range(site_names.size):
        c=site_col[s]
        if c>=0 and np.isfinite(mat[y_idx,c]):
            v[s]=mat[y_idx,c]
    ok=np.isfinite(v)
    if np.count_nonzero(ok)<3:
        continue
    try:
        OK=OrdinaryKriging(x=site_lons[ok],y=site_lats[ok],z=v[ok],variogram_model=variogram_model,enable_plotting=False,verbose=False)
    except Exception as e:
        print(f"{common_years[ii]}年 建模失败: {e}")
        continue
    for ridx in region_idx:
        if ridx is None:
            continue
        lat_sel,lon_sel=ridx
        gridx=lon_grid[lon_sel]
        gridy=lat_grid[lat_sel]
        try:
            zgrid,ss=OK.execute("grid",gridx,gridy)
            zgrid=np.asarray(zgrid,dtype=np.float32)
            mbool=mask_arr[ii][np.ix_(lat_sel,lon_sel)]>0.5
            zfill=np.where(mbool,zgrid,np.nan).astype(np.float32)
            rr=np.ix_(lat_sel,lon_sel)
            cur=out_stack[ii][rr]
            out_stack[ii][rr]=np.where(np.isnan(cur),zfill,cur)
        except Exception as e:
            print(f"{common_years[ii]}年 区域插值失败: {e}")
            continue
out=xr.Dataset(data_vars={"AgeDepSpline":(["year","latitude","longitude"],out_stack)},coords={"year":common_years.astype(np.int32),"latitude":lat_grid,"longitude":lon_grid},attrs={"description":"Kriged annual values (AgeDepSpline) within two target regions, masked by recon_mask_40yr, with per-site/year EPS≥0.85筛选","source_data":"AgeDepSpline.csv 历年值插值 + Detrening EPS","method":f"Ordinary Kriging (variogram_model={variogram_model})","units":"same as input"})
out["AgeDepSpline"].encoding.update({"zlib":True,"complevel":4,"_FillValue":np.nan})
os.makedirs(os.path.dirname(out_nc_path),exist_ok=True)
out.to_netcdf(out_nc_path)
print(f"已保存: {out_nc_path} | 维度: {dict(out.dims)}")
