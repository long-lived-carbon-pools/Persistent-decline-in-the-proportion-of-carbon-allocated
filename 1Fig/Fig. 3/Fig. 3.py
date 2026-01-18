# -*- coding: utf-8 -*-
import os,numpy as np,pandas as pd,xarray as xr,matplotlib.pyplot as plt,matplotlib as mpl,geopandas as gpd,matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm
from pandas.api.types import is_numeric_dtype
from matplotlib.ticker import FuncFormatter,MaxNLocator,MultipleLocator,FormatStrFormatter
mpl.rcParams["font.family"]="Times New Roman"
mpl.rcParams["axes.unicode_minus"]=False
mpl.rcParams["figure.dpi"]=300
mpl.rcParams["savefig.dpi"]=300
mpl.rcParams["font.size"]=22
recon_path=r"D:\all\data\tree_ring4 - AgeDepSpline(EPS0.85).nc"
model_path=r"D:\all\data\npp\1_ISBA-CTRIP.nc"
world_shp=r"D:\all\data\世界底图\World.shp"
na_clim_shp=r"D:\all\data\世界底图\Koppen_1991_2020_NA_big7_simple.shp"
eu_clim_shp=r"D:\all\data\世界底图\欧洲7类-class.shp"
NA_r_diff_dir=r"D:\all\1Fig\Fig. 3\data\1"
na_raw_dir=r"D:\all\1Fig\Fig. 3\data\2"
EU_r_diff_dir=r"D:\all\1Fig\Fig. 3\data\3"
eu_raw_dir=r"D:\all\1Fig\Fig. 3\data\4"
out_dir=r"D:\all\1Fig\Fig. 3"
os.makedirs(out_dir,exist_ok=True)
out_png=os.path.join(out_dir,"Fig. 3.png")
lonlat_list=[(-167.27,-55.00,9.58,74.99),(-11.0,57.0,30.0,75.0)]
base_classes=[1,2,3,4,5,6,7]
label_map={1:"Tropical",2:"Arid",3:"Temperate",4:"Dfa",5:"Dfb",6:"D other",7:"E"}
agg_defs={102:("Warm",[1,2,3,4,5]),103:("Cold",[6,7]),104:("All",[1,2,3,4,5,6,7])}
MIN_N=10
na_label_map={1:"A",2:"B",3:"C",4:"Dfa",5:"Dfb"}
eu_label_map={2:"B",3:"C",4:"Dfa",5:"Dfb",6:"D other"}
agg_defs2={102:"Warm",103:"Cold",104:"All"}
na_order=[102,103]
label_display_map={"A":"Tropical","B":"Arid","C":"Temperate"}
def get_recon_da(ds):
    cand=[v for v in ds.data_vars if "year" in ds[v].dims]
    prefer=[v for v in cand if ("recon" in ds[v].name.lower()) or ("agedep" in ds[v].name.lower())]
    return ds[prefer[0] if prefer else cand[0]]
def get_model_var(ds):
    cand=[v for v in ds.data_vars if "year" in ds[v].dims]
    prefer=[v for v in cand if v.lower() in ("npp","gpp")]
    return prefer[0] if prefer else cand[0]
def diff_corr_region(lon_min,lon_max,lat_min,lat_max):
    dsA=xr.open_dataset(recon_path)
    daA=get_recon_da(dsA).sel(latitude=slice(lat_min,lat_max),longitude=slice(lon_min,lon_max))
    yearsA=daA["year"].values.astype(int)
    dsB=xr.open_dataset(model_path)
    varB=get_model_var(dsB)
    daB=dsB[varB].sel(latitude=slice(lat_min,lat_max),longitude=slice(lon_min,lon_max))
    yearsB=daB["year"].values.astype(int)
    common=np.intersect1d(yearsA,yearsB)
    common=common[(common>=1980)&(common<=2010)]
    if common.size<MIN_N+1:
        latA=daA["latitude"].values
        lonA=daA["longitude"].values
        r=np.full((latA.size,lonA.size),np.nan,np.float32)
        dsA.close();dsB.close()
        return r,latA,lonA
    A=daA.sel(year=common).values.astype(np.float32)
    B=daB.sel(year=common).values.astype(np.float32)
    dsA.close();dsB.close()
    Ad=np.diff(A,axis=0)
    Bd=np.diff(B,axis=0)
    T,H,W=Ad.shape
    Ad2=Ad.reshape(T,-1)
    Bd2=Bd.reshape(T,-1)
    m=np.isfinite(Ad2)&np.isfinite(Bd2)
    n=m.sum(axis=0)
    Adz=np.where(m,Ad2,0.0)
    Bdz=np.where(m,Bd2,0.0)
    sx=Adz.sum(axis=0)
    sy=Bdz.sum(axis=0)
    sxx=(Adz*Adz).sum(axis=0)
    syy=(Bdz*Bdz).sum(axis=0)
    sxy=(Adz*Bdz).sum(axis=0)
    num=n*sxy-sx*sy
    den=np.sqrt((n*sxx-sx*sx)*(n*syy-sy*sy))
    r=np.full_like(den,np.nan,np.float32)
    v=(n>=MIN_N)&(den>0)
    r[v]=num[v]/den[v]
    r=r.reshape(H,W)
    r=np.clip(r,-1.0,1.0)
    latA=daA["latitude"].values
    lonA=daA["longitude"].values
    return r,latA,lonA
def detect_class_col(gdf):
    num_cols=[c for c in gdf.columns if is_numeric_dtype(gdf[c])]
    for c in num_cols:
        vals=np.unique(gdf[c].dropna().astype(int))
        if vals.size>0 and np.all(np.isin(vals,base_classes)):
            return c
    return num_cols[0]
def zone_box_data(r,lat,lon,shp_path,include_tropical=True):
    r_flat=r.reshape(-1)
    lat2,lon2=np.meshgrid(lat,lon,indexing="ij")
    lat_flat=lat2.reshape(-1)
    lon_flat=lon2.reshape(-1)
    mask=np.isfinite(r_flat)
    r_flat=r_flat[mask]
    lat_flat=lat_flat[mask]
    lon_flat=lon_flat[mask]
    if r_flat.size==0:
        if include_tropical:
            names=["Tropical","Arid","Temperate","Dfa","Dfb","Warm","Cold","All"]
        else:
            names=["Arid","Temperate","Dfa","Dfb","Warm","Cold","All"]
        data=[np.array([np.nan]) for _ in names]
        return names,data
    gdf=gpd.read_file(shp_path)
    col=detect_class_col(gdf)
    df=pd.DataFrame({"lon":lon_flat,"lat":lat_flat,"r":r_flat})
    pts=gpd.GeoDataFrame(df,geometry=gpd.points_from_xy(df["lon"],df["lat"]),crs=gdf.crs if gdf.crs else "EPSG:4326")
    join=gpd.sjoin(pts,gdf[[col,"geometry"]],how="left",predicate="intersects")
    cls=join[col].values
    rvals=join["r"].values
    def get_vals(cls_list):
        m=np.isin(cls,cls_list)&np.isfinite(rvals)
        vals=rvals[m]
        if vals.size==0:vals=np.array([np.nan])
        return vals
    names=[]
    data=[]
    base_order=[1,2,3,4,5]
    for cid in base_order:
        if (not include_tropical) and cid==1:continue
        names.append(label_map[cid])
        data.append(get_vals([cid]))
    for code in [102,103,104]:
        label,clist=agg_defs[code]
        names.append(label)
        data.append(get_vals(clist))
    return names,data
def tag_name_na(code):return na_label_map[code] if code in na_label_map else agg_defs2[code]
def tag_name_eu(code):return eu_label_map[code] if code in eu_label_map else agg_defs2[code]
def read_curve(dir_path,tag):
    fp=os.path.join(dir_path,f"{tag}.csv")
    if not os.path.exists(fp):return None
    df=pd.read_csv(fp)
    if "year" not in df.columns or "mean_r" not in df.columns:return None
    return df["year"].to_numpy(),df["mean_r"].to_numpy()
def peak_of(series,year_min=None):
    if series is None:return None
    x,y=series
    m=np.isfinite(y)
    if year_min is not None:m=m&(x>=year_min)
    if not m.any():return None
    yi=y[m];xi=x[m]
    k=int(np.nanargmax(yi))
    return int(xi[k]),float(yi[k])
def merge_two_series(s1,s2):
    if s1 is None and s2 is None:return None
    if s1 is None:return (s2[0].astype(float),s2[1].astype(float))
    if s2 is None:return (s1[0].astype(float),s1[1].astype(float))
    x1,y1=s1[0].astype(float),s1[1].astype(float)
    x2,y2=s2[0].astype(float),s2[1].astype(float)
    years=np.union1d(x1,x2)
    m={}
    for x,y in ((x1,y1),(x2,y2)):
        for xi,yi in zip(x,y):
            if not np.isfinite(xi):continue
            k=int(xi)
            if k not in m:m[k]=[]
            if np.isfinite(yi):m[k].append(float(yi))
    yy=np.full(years.shape,np.nan,dtype=float)
    for i,yr in enumerate(years.astype(int)):
        if yr in m and len(m[yr])>0:yy[i]=float(np.nanmean(m[yr]))
    return years.astype(int),yy
r_list=[];lat_list=[];lon_list=[]
for (l1,l2,a1,a2) in lonlat_list:
    r,la,lo=diff_corr_region(l1,l2,a1,a2)
    r_list.append(r);lat_list.append(la);lon_list.append(lo)
vals=[]
for r in r_list:
    if np.isfinite(r).any():vals.append(np.abs(r[np.isfinite(r)]))
if len(vals)>0:
    allv=np.concatenate(vals)
    vabs=np.nanpercentile(allv,95)
    if (not np.isfinite(vabs))or(vabs<=0):vabs=0.1
else:vabs=0.1
norm=TwoSlopeNorm(vmin=-vabs,vcenter=0.0,vmax=vabs)
world=gpd.read_file(world_shp)
world=world.set_crs(epsg=4326) if world.crs is None else world.to_crs(epsg=4326)
curves={};all_vals=[]
for code in na_order:
    tag_na=tag_name_na(code)
    na_d=read_curve(NA_r_diff_dir,tag_na)
    na_r=read_curve(na_raw_dir,tag_na)
    tag_eu=tag_name_eu(code)
    eu_d=read_curve(EU_r_diff_dir,tag_eu)
    eu_r=read_curve(eu_raw_dir,tag_eu)
    all_d=merge_two_series(na_d,eu_d)
    all_r=merge_two_series(na_r,eu_r)
    curves[code]={"ALL_r_diff":all_d,"ALL_raw":all_r}
    for k in ("ALL_r_diff","ALL_raw"):
        v=curves[code][k]
        if v is not None:
            yy=v[1]
            all_vals.append(yy[np.isfinite(yy)])
if all_vals:
    all_vals=np.concatenate(all_vals)
    global_min=float(np.nanmin(all_vals))
    global_max=float(np.nanmax(all_vals))
else:
    global_min=-0.1;global_max=0.1
pad=(global_max-global_min)*0.1 if np.isfinite(global_max-global_min) and (global_max-global_min)>0 else 0.05
global_ymin=global_min-pad
global_ymax=global_max+pad
yticks=np.linspace(global_ymin,global_ymax,5)
fig=plt.figure(figsize=(12,13))
gs=gridspec.GridSpec(3,2,width_ratios=[1.15,1.0],height_ratios=[1.15,1.0,0.85])
ax_map_na=fig.add_subplot(gs[0,0])
ax_map_eu=fig.add_subplot(gs[0,1])
ax_box_na=fig.add_subplot(gs[1,0])
ax_box_eu=fig.add_subplot(gs[1,1])
ax_line_warm=fig.add_subplot(gs[2,0])
ax_line_cold=fig.add_subplot(gs[2,1])
fig.subplots_adjust(hspace=0.18,wspace=0.10)
for i,ax in enumerate([ax_map_na,ax_map_eu]):
    r=r_list[i]
    latA=lat_list[i]
    lonA=lon_list[i]
    LON,LAT=np.meshgrid(lonA,latA)
    world.plot(ax=ax,color="lightgrey",edgecolor="grey",linewidth=0.4,zorder=0)
    im=ax.pcolormesh(LON,LAT,r,cmap="BrBG",norm=norm,shading="auto",zorder=1)
    l1,l2,a1,a2=lonlat_list[i]
    ax.set_xlim(l1,l2);ax.set_ylim(a1,a2)
    ax.set_xlabel("");ax.set_ylabel("")
    ax.tick_params(axis="both",which="major",labelsize=16)
    ax.xaxis.set_major_locator(MultipleLocator(20))
    ax.yaxis.set_major_locator(MultipleLocator(10))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%d"))
    if i==1:
        ax.xaxis.set_major_locator(MultipleLocator(10))
        ax.xaxis.set_major_formatter(FormatStrFormatter("%d"))
        ax.yaxis.tick_right()
        ax.tick_params(axis="y",right=True,left=False)
    ax.set_aspect("auto")
med1=np.nanmedian(r_list[0][np.isfinite(r_list[0])]) if np.isfinite(r_list[0]).any() else np.nan
ax_map_na.text(0.03,0.03,f"Median r = {med1:.2f}",transform=ax_map_na.transAxes,ha="left",va="bottom",fontsize=18)
med2=np.nanmedian(r_list[1][np.isfinite(r_list[1])]) if np.isfinite(r_list[1]).any() else np.nan
ax_map_eu.text(0.10,0.97,f"Median r = {med2:.2f}",transform=ax_map_eu.transAxes,ha="left",va="top",fontsize=18)
cax=ax_map_na.inset_axes([0.02,0.25,0.35,0.04])
cbar=fig.colorbar(im,cax=cax,orientation="horizontal",extend="both")
cbar.ax.tick_params(labelsize=16)
names_na,data_na=zone_box_data(r_list[0],lat_list[0],lon_list[0],na_clim_shp,include_tropical=True)
ax_box_na.set_ylabel("r",fontsize=20)
bp=ax_box_na.boxplot(data_na,tick_labels=names_na,showfliers=False)
ax_box_na.tick_params(axis="x",labelrotation=45,labelsize=16)
ax_box_na.tick_params(axis="y",labelsize=16)
ax_box_na.yaxis.set_major_locator(MaxNLocator(nbins=5))
ax_box_na.yaxis.set_major_formatter(FuncFormatter(lambda x,pos:f"{x:.1f}"))
for j,d in enumerate(data_na):
    jitter=(np.random.rand(len(d))-0.5)*0.35
    ax_box_na.scatter(j+1+jitter,d,s=1,color="#1f77b4",alpha=0.1)
names_eu,data_eu=zone_box_data(r_list[1],lat_list[1],lon_list[1],eu_clim_shp,include_tropical=False)
bp2=ax_box_eu.boxplot(data_eu,tick_labels=names_eu,showfliers=False)
ax_box_eu.tick_params(axis="x",labelrotation=45,labelsize=16)
ax_box_eu.tick_params(axis="y",labelleft=False,labelright=False)
ax_box_eu.set_yticklabels([])
pos5=ax_line_warm.get_position();pos6=ax_line_cold.get_position()
dy=0.035
ax_line_warm.set_position([pos5.x0,pos5.y0-dy,pos5.width,pos5.height])
ax_line_cold.set_position([pos6.x0,pos6.y0-dy,pos6.width,pos6.height])
ax_line_warm.text(0.98,0.92,"Warm",transform=ax_line_warm.transAxes,ha="right",va="top",fontsize=20,fontweight="bold")
ax_line_cold.text(0.98,0.92,"Cold",transform=ax_line_cold.transAxes,ha="right",va="top",fontsize=20,fontweight="bold")
ax_box_eu.yaxis.set_major_locator(MaxNLocator(nbins=5))
ax_box_eu.yaxis.set_major_formatter(FuncFormatter(lambda x,pos:f"{x:.1f}"))
for j,d in enumerate(data_eu):
    jitter=(np.random.rand(len(d))-0.5)*0.35
    ax_box_eu.scatter(j+1+jitter,d,s=1,color="#1f77b4",alpha=0.1)
for line in bp["medians"]:line.set_color("#4b5f83")
for line in bp2["medians"]:line.set_color("#4b5f83")
all_na_vals=np.concatenate([d[np.isfinite(d)] for d in data_na if np.isfinite(d).any()]) if any(np.isfinite(d).any() for d in data_na) else np.array([0.0])
all_eu_vals=np.concatenate([d[np.isfinite(d)] for d in data_eu if np.isfinite(d).any()]) if any(np.isfinite(d).any() for d in data_eu) else np.array([0.0])
allb=np.concatenate([all_na_vals,all_eu_vals])
ymin=np.nanmin(allb) if np.isfinite(allb).any() else -1.0
ymax=1.0
ax_box_na.set_ylim(ymin,ymax);ax_box_eu.set_ylim(ymin,ymax)
for j,med_line in enumerate(bp["medians"]):
    y=med_line.get_ydata()[0]
    ax_box_na.text(j+1,y,f"{np.nanmedian(data_na[j]):.2f}",ha="center",va="center",fontsize=17,color="#d62728")
for j,med_line in enumerate(bp2["medians"]):
    y=med_line.get_ydata()[0]
    ax_box_eu.text(j+1,y,f"{np.nanmedian(data_eu[j]):.2f}",ha="center",va="center",fontsize=17,color="#d62728")
r_sig=0.36
for ax in (ax_box_na,ax_box_eu):
    ax.axhline(r_sig,color="k",linestyle="--",linewidth=1.2)
legend_handles={"r_raw":None,"r_diff":None}
axes2=[ax_line_warm,ax_line_cold]
for i,code in enumerate(na_order):
    ax=axes2[i]
    rr=curves[code]["ALL_raw"]
    rd=curves[code]["ALL_r_diff"]
    has_any=False
    if rr is not None and np.isfinite(rr[1]).any():
        p=ax.plot(rr[0],rr[1],linewidth=1.6,alpha=0.45,label="r_raw");has_any=True
        if legend_handles["r_raw"] is None:legend_handles["r_raw"]=p[0]
    if rd is not None and np.isfinite(rd[1]).any():
        p=ax.plot(rd[0],rd[1],linewidth=3.0,label="r_diff");has_any=True
        if legend_handles["r_diff"] is None:legend_handles["r_diff"]=p[0]
    if not has_any:
        ax.text(0.5,0.5,"No data",ha="center",va="center",fontsize=18);ax.set_axis_off();continue
    ax.set_ylim(global_ymin,global_ymax)
    ax.set_yticks(yticks)
    ylabels=[("0" if abs(round(v,1))<1e-6 else f"{v:.1f}") for v in yticks]
    ax.set_yticklabels(ylabels)
    ax.margins(x=0.01)
    ax.set_xlabel("Year",fontsize=20)
    if i==0:ax.set_ylabel("20-yr r",fontsize=20)
    if i!=0:ax.set_yticklabels([])
    ax.tick_params(axis="both",labelsize=16)
    y_top=global_ymax-(global_ymax-global_ymin)*0.4
    y_bot=global_ymin+(global_ymax-global_ymin)*0.10
    p=peak_of(rd)
    if p is not None:
        ax.axvline(p[0],color="k",linestyle="--",linewidth=1.2)
        ax.text(p[0],y_top if code==102 else y_bot,str(p[0]),ha="center",va="top",fontsize=18)
legend_items=[legend_handles[k] for k in ("r_raw","r_diff") if legend_handles[k] is not None]
if legend_items:
    lg=ax_line_warm.legend(handles=legend_items,fontsize=18,frameon=True,loc="lower right",borderpad=0.1,handlelength=1.5,handletextpad=0.5,markerscale=0.8)
    try:lg.get_frame().set_boxstyle("sawtooth",pad=0.05)
    except:pass
panel_fs=24
ax_map_na.text(0.02,0.98,"a",transform=ax_map_na.transAxes,fontsize=panel_fs,fontweight="bold",ha="left",va="top")
ax_map_eu.text(0.02,0.98,"b",transform=ax_map_eu.transAxes,fontsize=panel_fs,fontweight="bold",ha="left",va="top")
ax_box_na.text(0.02,0.98,"c",transform=ax_box_na.transAxes,fontsize=panel_fs,fontweight="bold",ha="left",va="top")
ax_box_eu.text(0.02,0.98,"d",transform=ax_box_eu.transAxes,fontsize=panel_fs,fontweight="bold",ha="left",va="top")
ax_line_warm.text(0.02,0.98,"e",transform=ax_line_warm.transAxes,fontsize=panel_fs,fontweight="bold",ha="left",va="top")
ax_line_cold.text(0.02,0.98,"f",transform=ax_line_cold.transAxes,fontsize=panel_fs,fontweight="bold",ha="left",va="top")
plt.savefig(out_png,bbox_inches="tight",dpi=300)
plt.close(fig)
print("✅ 已保存：",out_png)
