import os,re,numpy as np,xarray as xr,matplotlib.pyplot as plt,geopandas as gpd,pandas as pd,matplotlib as mpl
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator,FuncFormatter
plt.rcParams['font.family']='Times New Roman'
plt.rcParams['figure.dpi']=300
plt.rcParams['font.size']=30
plt.rcParams['axes.labelsize']=26
plt.rcParams['xtick.labelsize']=26
plt.rcParams['ytick.labelsize']=26
nc1=r"data/6Sample data/huadongchafen1.nc"
nc2=r"data/6Sample data/huadongchafen20.nc"
shp_world=r"map/World.shp"
eu_shp_path=r"map/欧洲7类-class.shp"
na_shp_path=r"map/Koppen_1991_2020_NA_big7_simple.shp"
out_png=r"results/1Fig/Fig. 4.png"
TOPK=4
Y0,Y1=1987,2006
def detect_time_dim(da):
    if "window_center_year" in da.dims:return "window_center_year"
    if "year" in da.dims:return "year"
    return da.dims[0]
def region_compute(ds,fac_vars,fac_names,lon_min,lon_max,lat_min,lat_max):
    stack=xr.concat([ds[k] for k in fac_vars],dim="factor").assign_coords(factor=np.array(fac_names,dtype=object))
    sub=stack.sel(window_center_year=slice(Y0,Y1))
    mean_over_time=sub.mean("window_center_year",skipna=True)
    roi=mean_over_time.sel(latitude=slice(lat_min,lat_max),longitude=slice(lon_min,lon_max))
    data=roi.transpose("factor","latitude","longitude").values
    lat=roi["latitude"].values
    lon=roi["longitude"].values
    valid=np.any(np.isfinite(data),axis=0)
    filled=np.where(np.isfinite(data),data,-np.inf)
    order=np.argsort(-filled,axis=0)
    best_val=np.take_along_axis(filled,order[0:1],axis=0)[0]
    second_val=np.take_along_axis(filled,order[1:2],axis=0)[0]
    count_finite=np.sum(np.isfinite(filled),axis=0)
    with np.errstate(invalid="ignore"):
        margin=best_val-second_val
    margins=margin[(count_finite>=2)&np.isfinite(margin)]
    TOL=float(np.percentile(margins,15)) if margins.size>0 else 0.01
    TOL=max(TOL,0.01)
    co_best=(np.nan_to_num(best_val)-np.nan_to_num(filled))<=TOL
    co_best&=np.isfinite(filled)
    co_counts=co_best.sum(axis=(1,2))
    lon2d,lat2d=np.meshgrid(lon,lat)
    return lat,lon,lon2d,lat2d,filled,valid,TOL,co_counts
def assign_region(filled,valid,TOL,sel_idx,K):
    sel_vals=filled[sel_idx,:,:]
    sel_best=np.max(sel_vals,axis=0)
    sel_cobest=(np.nan_to_num(sel_best)[None,:,:]-np.nan_to_num(sel_vals))<=TOL
    sel_cobest&=np.isfinite(sel_vals)
    Ny,Nx=sel_best.shape
    assigned=np.full((Ny,Nx),np.nan)
    counts=np.zeros(K,dtype=int)
    for y in range(Ny):
        for x in range(Nx):
            if not valid[y,x]:continue
            cand=np.where(sel_cobest[:,y,x])[0]
            if cand.size==0:j=int(np.argmax(sel_vals[:,y,x]))
            else:
                j=int(np.argmin(counts[cand]))
                j=cand[j]
            assigned[y,x]=j
            counts[int(j)]+=1
    assigned_ma=np.ma.array(assigned,mask=~valid)
    return assigned_ma,counts
def get_delta_vars(ds):
    return sorted([v for v in ds.data_vars if str(v).startswith("delta_") and "98" not in v and "_12" not in v and "d2m" not in v.lower() and "t2m" not in v.lower() and "frs" not in v.lower()])
def da_to_series(da):
    tdim=detect_time_dim(da)
    other=[d for d in da.dims if d!=tdim]
    if other:da=da.mean(other,skipna=True)
    return da.to_pandas()
def zrdiff(ds,mask=None):
    da=ds["r_raw_diff_win"]
    if mask is not None:da=da.where(mask)
    s=da_to_series(da)
    try:s.index=s.index+10
    except:pass
    return s
def get_zone_col(gdf):
    for c in ["class7","CLASS7","zone","ZONE","Koppen","koppen","CLIMATE","climate","Class","class"]:
        if c in gdf.columns:return c
    for c in gdf.columns:
        if c!="geometry":return c
    return None
def build_masks(ds):
    lat=ds["latitude"].values
    lon=ds["longitude"].values
    lon2,lat2=np.meshgrid(lon,lat)
    flat_lon=lon2.ravel()
    flat_lat=lat2.ravel()
    idx=np.arange(flat_lon.size)
    df=pd.DataFrame({"lon":flat_lon,"lat":flat_lat,"idx":idx})
    gdf_pts=gpd.GeoDataFrame(df,geometry=gpd.points_from_xy(df.lon,df.lat),crs="EPSG:4326")
    eu=gpd.read_file(eu_shp_path)
    na=gpd.read_file(na_shp_path)
    if eu.crs is None:eu=eu.set_crs("EPSG:4326")
    else:eu=eu.to_crs("EPSG:4326")
    if na.crs is None:na=na.set_crs("EPSG:4326")
    else:na=na.to_crs("EPSG:4326")
    col_eu=get_zone_col(eu)
    col_na=get_zone_col(na)
    eu=eu[["geometry",col_eu]].rename(columns={col_eu:"zone"})
    na=na[["geometry",col_na]].rename(columns={col_na:"zone"})
    clim=gpd.GeoDataFrame(pd.concat([eu,na],ignore_index=True),crs="EPSG:4326")
    joined=gpd.sjoin(gdf_pts,clim,how="left",predicate="within")
    joined["zone"]=joined["zone"].astype(str)
    zones=np.array(joined.set_index("idx")["zone"].reindex(idx).values,dtype=object)
    warm_codes=set(["A","B","C","Dfa","Dfb","Tropical","Arid","Temperate"])
    cold_codes=set(["D other","Dother","D","E","Cold","Subpolar","ET","EF"])
    warm=(np.array([(z in warm_codes) for z in zones]).reshape(lat.size,lon.size))
    cold=(np.array([(z in cold_codes) for z in zones]).reshape(lat.size,lon.size))
    return xr.DataArray(warm,coords={"latitude":lat,"longitude":lon},dims=("latitude","longitude")),xr.DataArray(cold,coords={"latitude":lat,"longitude":lon},dims=("latitude","longitude"))
def clean_name(s):
    s=str(s)
    if s.startswith("delta_"):s=s[6:]
    if s.endswith("_68"):s=s[:-3]
    if s.endswith("_summer"):s=s[:-7]
    s=re.sub(r"(?i)scpdsi","scPDSI",s)
    return s
def panel(ds,mask,title,ax,ylabel=None,show_legend=True,fixed_vars=None,color_list=None):
    if fixed_vars is not None:
        vars_delta=[v for v in fixed_vars if v in ds.data_vars]
        if not vars_delta:return
        sample=ds[vars_delta[0]]
        tdim=detect_time_dim(sample)
        da_list=[]
        for v in vars_delta:
            da=ds[v]
            if isinstance(da,xr.DataArray):
                da_list.append(da.where(mask))
        if not da_list:return
        da_all=xr.concat(da_list,dim="var")
        da_all=da_all.assign_coords(var=("var",vars_delta))
        abs_all=xr.apply_ufunc(np.abs,da_all)
        mean_abs=abs_all.mean(tdim,skipna=True)
        mean_abs_filled=mean_abs.fillna(-1.0)
        dom_idx=mean_abs_filled.argmax("var")
        valid_any=np.isfinite(mean_abs).any("var")
        if not bool(valid_any.sum().values):return
        total_valid=int(valid_any.sum().values)
        percent={}
        for i,v in enumerate(vars_delta):
            cnt=int(((dom_idx==i)&valid_any).sum().values)
            percent[v]=cnt/total_valid*100 if total_valid>0 else 0
        series_dict={}
        for i,v in enumerate(vars_delta):
            if percent[v]<=0:continue
            dom_mask=(dom_idx==i)&valid_any
            da_v=ds[v].where(mask&dom_mask)
            s_v=da_to_series(da_v)
            if s_v.isna().all():continue
            try:s_v.index=s_v.index+10
            except:pass
            series_dict[v]=s_v
        if not series_dict:return
        df=pd.DataFrame(series_dict)
        df=df[[c for c in df.columns if not df[c].isna().all()]]
        if df.shape[1]==0:return
        z=zrdiff(ds,mask).reindex(df.index)
        for i,c in enumerate(df.columns):
            col=color_list[i] if color_list is not None and i<len(color_list) else plt.get_cmap("tab10")(i%10)
            lab=f"{clean_name(c)} ({percent[c]:.1f}%)"
            ax.plot(df.index,df[c],lw=1.8,color=col,label=lab)
        ax.plot(df.index,z,ls="-.",lw=2.2,color="k",label="r (NPP-TRW)")
    else:
        vars_delta=get_delta_vars(ds)
        if not vars_delta:return
        sample=ds[vars_delta[0]]
        tdim=detect_time_dim(sample)
        da_list=[]
        name_list=[]
        for v in vars_delta:
            da=ds[v]
            if isinstance(da,xr.DataArray):
                da_list.append(da.where(mask))
                name_list.append(v)
        if not da_list:return
        da_all=xr.concat(da_list,dim="var")
        da_all=da_all.assign_coords(var=("var",name_list))
        abs_all=xr.apply_ufunc(np.abs,da_all)
        mean_abs=abs_all.mean(tdim,skipna=True)
        mean_abs_filled=mean_abs.fillna(-1.0)
        dom_idx_all=mean_abs_filled.argmax("var")
        importance_all={}
        for i,v in enumerate(name_list):
            ma_v=mean_abs.sel(var=v)
            valid=np.isfinite(ma_v)
            cnt=int(((dom_idx_all==i)&valid).sum().values)
            if cnt>0:importance_all[v]=cnt
        if not importance_all:return
        top10=[k for k,_ in sorted(importance_all.items(),key=lambda x:x[1],reverse=True)[:2]]
        mean_abs_top=mean_abs.sel(var=top10)
        valid_any=np.isfinite(mean_abs_top).any("var")
        if not bool(valid_any.sum().values):return
        dom_idx_top=mean_abs_top.fillna(-1.0).argmax("var")
        importance_top={}
        for i,v in enumerate(top10):
            cnt=int(((dom_idx_top==i)&valid_any).sum().values)
            if cnt>0:importance_top[v]=cnt
        if not importance_top:return
        total_valid=int(valid_any.sum().values)
        top5=[k for k,_ in sorted(importance_top.items(),key=lambda x:x[1],reverse=True)[:2]]
        series_dict={}
        percent={}
        for v in top5:
            i=top10.index(v)
            dom_mask=(dom_idx_top==i)&valid_any
            cnt=int(dom_mask.sum().values)
            if cnt==0:continue
            percent[v]=cnt/total_valid*100 if total_valid>0 else 0
            da_v=ds[v].where(mask&dom_mask)
            s_v=da_to_series(da_v)
            if s_v.isna().all():continue
            try:s_v.index=s_v.index+10
            except:pass
            series_dict[v]=s_v
        if not series_dict:return
        df=pd.DataFrame(series_dict)
        df=df[[c for c in df.columns if not df[c].isna().all()]]
        if df.shape[1]==0:return
        order=sorted(percent.keys(),key=lambda x:percent[x],reverse=True)
        df_top=df[order]
        z=zrdiff(ds,mask).reindex(df_top.index)
        cmap=plt.get_cmap("tab10")
        for i,c in enumerate(order):
            lab=f"{clean_name(c)} ({percent[c]:.1f}%)"
            ax.plot(df_top.index,df_top[c],lw=1.8,color=cmap(i%10),label=lab)
        ax.plot(df_top.index,z,ls="-.",lw=2.2,color="k",label="r (NPP-TRW)")
    ax.margins(y=0.12)
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x,_:f"{x:.1f}"))
    if ylabel is not None:ax.set_ylabel(ylabel,fontsize=30)
    ax.set_xlabel("Year",fontsize=30)
    ax.text(0.86,0.96,title,transform=ax.transAxes,ha="left",va="top",fontsize=32,fontweight="bold")
    ymin,ymax=ax.get_ylim()
    z=zrdiff(ds,mask)
    if z.notna().any():
        py=int(z.idxmax())
        ax.axvline(py,color="k",ls="--",lw=1.4)
        ax.text(py,ymax-(ymax-ymin)*0.07,str(py),ha="center",fontsize=26)
    if show_legend:
        handles, labels = ax.get_legend_handles_labels()
        parsed = []
        for h,l in zip(handles,labels):
            m = re.search(r"/(([-0-9/.]+)%/)", l)
            if m:
                pct = float(m.group(1))
            else:
                pct = -999
            parsed.append((pct, h, l))
        parsed_sorted = sorted(parsed, key=lambda x: x[0], reverse=True)
        handles_sorted = [x[1] for x in parsed_sorted]
        labels_sorted = [x[2] for x in parsed_sorted]
        ax.legend(handles_sorted, labels_sorted, loc="upper left",
                bbox_to_anchor=(0,0.9), frameon=True, fontsize=20)

ds1=xr.open_dataset(nc1)
ds2=xr.open_dataset(nc2)
ds_combined=xr.merge([ds1,ds2])
warm_mask,cold_mask=build_masks(ds_combined)
fac_vars1=[k for k in ds1.data_vars if k.startswith("delta_") and ("98" not in k) and ("_12" not in k) and ("d2m" not in k.lower()) and ("t2m" not in k.lower()) and ("frs" not in k.lower())]
fac_vars2=[k for k in ds2.data_vars if k.startswith("delta_") and ("98" not in k) and ("_12" not in k) and ("d2m" not in k.lower()) and ("t2m" not in k.lower()) and ("frs" not in k.lower())]
fac_vars=sorted(set(fac_vars1)|set(fac_vars2))
if len(fac_vars)==0:raise RuntimeError("两个NC在当前过滤条件下没有任何delta_变量可以使用")
fac_names=[re.sub(r"^delta_","",k) for k in fac_vars]
eu_lon_min,eu_lon_max=-11.0,57.0
eu_lat_min,eu_lat_max=30.0,75.0
na_lon_min,na_lon_max=-167.27,-55.00
na_lat_min,na_lat_max=9.58,74.99
eu_lat,eu_lon,eu_lon2d,eu_lat2d,eu_filled,eu_valid,eu_TOL,eu_counts=region_compute(ds_combined,fac_vars,fac_names,eu_lon_min,eu_lon_max,eu_lat_min,eu_lat_max)
na_lat,na_lon,na_lon2d,na_lat2d,na_filled,na_valid,na_TOL,na_counts=region_compute(ds_combined,fac_vars,fac_names,na_lon_min,na_lon_max,na_lat_min,na_lat_max)
global_counts=eu_counts+na_counts
sel_idx=np.argsort(global_counts)[-TOPK:][::-1]
K=len(sel_idx)
sel_names=[fac_names[i] for i in sel_idx]
sel_vars=[fac_vars[i] for i in sel_idx]
eu_assigned,eu_sel_counts=assign_region(eu_filled,eu_valid,eu_TOL,sel_idx,K)
na_assigned,na_sel_counts=assign_region(na_filled,na_valid,na_TOL,sel_idx,K)
total_valid=np.count_nonzero(eu_valid)+np.count_nonzero(na_valid)
area_frac=(eu_sel_counts+na_sel_counts)/total_valid if total_valid>0 else np.zeros(K,dtype=float)
is_spei=np.array([("spei" in n.lower()) or ("scpdsi" in n.lower()) for n in sel_names],dtype=bool)
idx_spei=[i for i in range(K) if is_spei[i]]
idx_other=[i for i in range(K) if not is_spei[i]]
colors=np.zeros((K,4))
dry_colors=["#8c5e37","#d9a97c"]
temp_colors=["#7fbf7f","#2b6f4e"]
if len(idx_spei)>0:
    for r,j in enumerate(idx_spei):
        colors[j]=mpl.colors.to_rgba(dry_colors[r%2])
if len(idx_other)>0:
    for r,j in enumerate(idx_other):
        colors[j]=mpl.colors.to_rgba(temp_colors[r%2])
color_list=[tuple(colors[i]) for i in range(K)]
world=gpd.read_file(shp_world)
world=world.set_crs(epsg=4326) if world.crs is None else world.to_crs(epsg=4326)
fig=plt.figure(figsize=(22,14))
ax1_left,ax1_bottom,ax1_width,ax1_height=0.06,0.5,0.46,0.46
ax2_left,ax2_bottom,ax2_width,ax2_height=0.54,0.5,0.4,0.46
ax3_left,ax3_bottom,ax3_width,ax3_height=0.06,0.08,0.46,0.38
ax4_left,ax4_bottom,ax4_width,ax4_height=0.54,0.08,0.4,0.38
ax_na=fig.add_axes([ax1_left,ax1_bottom,ax1_width,ax1_height])
ax_eu=fig.add_axes([ax2_left,ax2_bottom,ax2_width,ax2_height])
ax_warm=fig.add_axes([ax3_left,ax3_bottom,ax3_width,ax3_height])
ax_cold=fig.add_axes([ax4_left,ax4_bottom,ax4_width,ax4_height])
ax_na.text(0.05,0.98,"a",transform=ax_na.transAxes,ha="right",va="top",fontsize=40,fontweight="bold")
ax_eu.text(0.06,0.98,"b",transform=ax_eu.transAxes,ha="right",va="top",fontsize=40,fontweight="bold")
ax_warm.text(0.05,0.98,"c",transform=ax_warm.transAxes,ha="right",va="top",fontsize=40,fontweight="bold")
ax_cold.text(0.06,0.98,"d",transform=ax_cold.transAxes,ha="right",va="top",fontsize=40,fontweight="bold")
order=idx_spei+idx_other
clean_names=[re.sub(r"_[0-9]+$","",name) for name in sel_names]
clean_names=[re.sub(r"(?i)scpdsi","scPDSI",n) for n in clean_names]
labels=[f"{clean_names[j]} ({area_frac[j]*100:.1f}%)" for j in order]
handles=[Patch(facecolor=colors[j],edgecolor="none",label=labels[i]) for i,j in enumerate(order)]
world.cx[na_lon_min:na_lon_max,na_lat_min:na_lat_max].plot(ax=ax_na,facecolor="0.9",edgecolor="none",zorder=0)
ax_na.pcolormesh(na_lon2d,na_lat2d,na_assigned,cmap=ListedColormap(colors),vmin=-0.5,vmax=K-0.5,shading="nearest",zorder=1)
ax_na.set_aspect("auto")
ax_na.legend(handles=handles,loc="lower left",frameon=True,fontsize=20)
ax_na.set_xlim(na_lon_min,na_lon_max)
ax_na.set_ylim(na_lat_min,na_lat_max)
ax_na.set_xlabel("")
ax_na.set_ylabel("")
ax_na.margins(0,0)
ax_na.set_xlabel("")
ax_na.set_ylabel("")
ax_na.margins(0,0)
ax_na.tick_params(axis="both",which="major",labelsize=26,labelbottom=True,labelleft=True,bottom=True,left=True)
ax_na.xaxis.set_major_locator(MultipleLocator(20))
ax_na.yaxis.set_major_locator(MultipleLocator(10))

world.cx[eu_lon_min:eu_lon_max,eu_lat_min:eu_lat_max].plot(ax=ax_eu,facecolor="0.9",edgecolor="none",zorder=0)
ax_eu.pcolormesh(eu_lon2d,eu_lat2d,eu_assigned,cmap=ListedColormap(colors),vmin=-0.5,vmax=K-0.5,shading="nearest",zorder=1)
ax_eu.set_aspect("auto")
ax_eu.set_xlim(eu_lon_min,eu_lon_max)
ax_eu.set_ylim(eu_lat_min,eu_lat_max)
ax_eu.set_xlabel("")
ax_eu.set_ylabel("")
ax_eu.margins(0,0)
ax_eu.set_yticks(np.arange(np.ceil(eu_lat_min/10)*10,eu_lat_max+1,10))
ax_eu.set_xlabel("")
ax_eu.set_ylabel("")
ax_eu.margins(0,0)
ax_eu.tick_params(axis="both",which="major",labelsize=26,labelbottom=True,labelleft=False,bottom=True,left=False,right=True,labelright=True)
ax_eu.xaxis.set_major_locator(MultipleLocator(10))
ax_eu.yaxis.set_major_locator(MultipleLocator(10))
ax_eu.yaxis.set_major_formatter(FuncFormatter(lambda v,pos:f"{int(v):d}"))



panel(ds_combined,warm_mask,"Warm",ax_warm,ylabel="Partial correlation",show_legend=True,fixed_vars=sel_vars,color_list=color_list)
panel(ds_combined,cold_mask,"Cold",ax_cold,ylabel=None,show_legend=True,fixed_vars=sel_vars,color_list=color_list)
ymin=min(ax_warm.get_ylim()[0],ax_cold.get_ylim()[0])
ymax=max(ax_warm.get_ylim()[1],ax_cold.get_ylim()[1])
ax_warm.set_ylim(ymin,ymax)
ax_cold.set_ylim(ymin,ymax)
ax_cold.set_ylabel("")
ax_cold.get_yaxis().set_visible(False)
for ax in [ax_warm,ax_cold]:
    ax.set_aspect("auto")
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x,_:f"{x:.1f}"))
os.makedirs(os.path.dirname(out_png),exist_ok=True)
plt.savefig(out_png,bbox_inches="tight",pad_inches=0.02,facecolor="white")
plt.close()
print("✅ 已保存：",out_png)
