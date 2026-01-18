# -*- coding: utf-8 -*-
import os,numpy as np,xarray as xr,matplotlib.pyplot as plt,geopandas as gpd
from matplotlib.colors import TwoSlopeNorm
import matplotlib.patheffects as pe
a_path=r"D:\all\data\tree_ring4 - AgeDepSpline(EPS0.85).nc"
b_dir=r"D:\all\data\npp"
world_shp=r"D:\all\data\世界底图\World.shp"
out_dir_grid=r"D:\all\2Extended Data Fig\Extended Data Fig. 5"
os.makedirs(out_dir_grid,exist_ok=True)
yr0,yr1=1981,2010
lon_min,lon_max=-11.0,57
lat_min,lat_max=28,75.0
plt.rcParams["font.family"]="Times New Roman"
plt.rcParams["axes.unicode_minus"]=False
def sel_year(ds):
    if "year" not in ds.coords: raise ValueError("缺少year坐标")
    ds=ds.sel(year=slice(yr0,yr1))
    ds["year"]=ds["year"].astype(np.int32)
    return ds
def corr_min_n(x,y,min_n=10):
    v=np.isfinite(x)&np.isfinite(y)
    n=v.sum(axis=0)
    vx,vy=np.where(v,x,0.0),np.where(v,y,0.0)
    sx,sy=vx.sum(axis=0),vy.sum(axis=0)
    sxx,syy=(vx*vx).sum(axis=0),(vy*vy).sum(axis=0)
    sxy=(vx*vy).sum(axis=0)
    with np.errstate(invalid="ignore",divide="ignore"):
        mx,my=sx/n,sy/n
        num=sxy-n*mx*my
        den=np.sqrt((sxx-n*mx*mx)*(syy-n*my*my))
        r=np.divide(num,den,out=np.full_like(num,np.nan,dtype=float),where=(den>0)&(n>=min_n))
    return r
world=gpd.read_file(world_shp)
with xr.open_dataset(a_path) as da:
    da=da.load()
da=sel_year(da)
if "AgeDepSpline" not in da: raise ValueError("A中缺少AgeDepSpline")
A=da["AgeDepSpline"].transpose("year","latitude","longitude")
results=[]
for fname in os.listdir(b_dir):
    if not fname.lower().endswith(".nc"): continue
    if fname.startswith("2"): print(f"[跳过排名] {fname}: 以'2'开头"); continue
    fpath=os.path.join(b_dir,fname)
    try:
        with xr.open_dataset(fpath) as db:
            db=db.load()
        db=sel_year(db)
        if "npp" not in db: print(f"[跳过] {fname}: 缺少npp"); continue
        B=db["npp"].transpose("year","latitude","longitude")
        A1,B1=xr.align(A,B,join="inner")
        if A1.sizes["year"]<4: print(f"[跳过] {fname}: 共同年份不足"); continue
        A1=A1.diff("year");B1=B1.diff("year")
        A1,B1=xr.align(A1,B1,join="inner")
        if A1.sizes["year"]<3: print(f"[跳过] {fname}: 差分后年份不足"); continue
        A1=A1.sel(latitude=slice(lat_min,lat_max),longitude=slice(lon_min,lon_max))
        B1=B1.sel(latitude=slice(lat_min,lat_max),longitude=slice(lon_min,lon_max))
        r=corr_min_n(A1.values.astype(float),B1.values.astype(float),min_n=10)
        r_median=float(np.nanmedian(r))
        lat_c=B1["latitude"].values;lon_c=B1["longitude"].values
        dlon=np.abs(np.median(np.diff(lon_c))) if lon_c.size>1 else 0.5
        dlat=np.abs(np.median(np.diff(lat_c))) if lat_c.size>1 else 0.5
        lon_edges=np.concatenate(([lon_c[0]-dlon/2],(lon_c[:-1]+lon_c[1:])/2,[lon_c[-1]+dlon/2]))
        lat_edges=np.concatenate(([lat_c[0]-dlat/2],(lat_c[:-1]+lat_c[1:])/2,[lat_c[-1]+dlat/2]))
        LonE,LatE=np.meshgrid(lon_edges,lat_edges)
        results.append({"name":os.path.splitext(fname)[0],"r":r,"LonE":LonE,"LatE":LatE,"median":r_median})
        print(f"[完成] {fname}  中位数r={r_median:.3f}")
    except Exception as e:
        print(f"[失败] {fname}: {e}")
if len(results)==0: raise RuntimeError("没有可用结果用于拼图")
results_top=sorted(results,key=lambda d:d["median"],reverse=True)[:9]
ncols,nrows=3,3
fig_w,fig_h=ncols*4.2,nrows*2.0
fig,axes=plt.subplots(nrows,ncols,figsize=(fig_w,fig_h),dpi=300,sharex=True,sharey=True)
axes=axes.ravel()
all_abs=np.concatenate([np.abs(x["r"].ravel()) for x in results_top])
vlim=float(np.nanpercentile(all_abs,98))
vlim=min(1.0,max(0.5,vlim))
norm=TwoSlopeNorm(vmin=-vlim,vcenter=0,vmax=vlim)
cmap="coolwarm"
for ax in axes:
    for s in ax.spines.values(): s.set_linewidth(0.6);s.set_edgecolor("#666666")
for ax,item in zip(axes,results_top):
    world.plot(ax=ax,facecolor="#f4f4f4",edgecolor="#bbbbbb",linewidth=0.3)
    ax.pcolormesh(item["LonE"],item["LatE"],item["r"],cmap=cmap,norm=norm,shading="flat")
    ax.set_xlim(lon_min,lon_max);ax.set_ylim(lat_min,lat_max)
    ax.set_xticks([]);ax.set_yticks([])
    ax.text(0.02,0.98,f'{item["name"]}\nMedian r={item["median"]:.2f}',transform=ax.transAxes,ha="left",va="top",fontsize=11,path_effects=[pe.withStroke(linewidth=1.2,foreground="white")])
for k in range(len(results_top),nrows*ncols): axes[k].axis("off")
plt.subplots_adjust(left=0.03,right=0.88,top=0.97,bottom=0.05,wspace=0.035,hspace=0.01)
cb=fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cmap),ax=axes.tolist(),location="right",shrink=0.6,pad=0.05,extend="both")
cb.set_label("r(ΔNPP, ΔTRW)")
out_grid=os.path.join(out_dir_grid,"Extended Data Fig. 5.png")
plt.savefig(out_grid,dpi=300,bbox_inches="tight",pad_inches=0.1)
plt.close()
print(f"[完成] 拼图已保存: {out_grid}")
