# -*- coding: utf-8 -*-
import os,numpy as np,pandas as pd,matplotlib.pyplot as plt,geopandas as gpd,xarray as xr
from rasterio.transform import from_bounds
from rasterio.features import rasterize
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap
nc_path1=r"data/6Sample data/huadongchafen1.nc"
nc_path2=r"data/6Sample data/huadongchafen20.nc"
na_shp_path=r"map/Koppen_1991_2020_NA_big7_simple.shp"
eu_shp_path=r"map/欧洲7类-class.shp"
out_dir=r"2results/2Extended Data Fig"
os.makedirs(out_dir,exist_ok=True)
png_bar=os.path.join(out_dir,"Extended Data Fig. 8.png")
plt.rcParams["font.family"]="Times New Roman"
plt.rcParams["axes.unicode_minus"]=False
plt.rcParams["xtick.labelsize"]=20
plt.rcParams["ytick.labelsize"]=22
plt.rcParams["figure.dpi"]=300
ds1=xr.open_dataset(nc_path1)
ds2=xr.open_dataset(nc_path2)
lon=ds1["longitude"].values
lat=ds1["latitude"].values
nx=lon.size
ny=lat.size
left=float(np.nanmin(lon))
right=float(np.nanmax(lon))
bottom=float(np.nanmin(lat))
top=float(np.nanmax(lat))
transform=from_bounds(left,bottom,right,top,nx,ny)
var_names1=[v for v in ds1.data_vars if v.startswith("delta_") and (("68" in v) or ("summer" in v.lower()))]
var_names2=[v for v in ds2.data_vars if v.startswith("delta_") and (("68" in v) or ("summer" in v.lower()))]
all_vars=sorted(set(var_names1)|set(var_names2))
base_factors=[]
for v in all_vars:
    base_factors.append(v.split("delta_")[1])
base_factors=list(dict.fromkeys(base_factors))
Moisture_keys=[s.lower() for s in ["pet_","VPD_","RH_"]]
temp_keys=[s.lower() for s in ["vap_","t2m_","stl4_"]]
rad_keys=[s.lower() for s in ["cld_"]]
drought_keys=[s.lower() for s in ["spei6_","spei12_","scpdsi_"]]
cat_map={}
for f in base_factors:
    s=f.lower()
    if any(k in s for k in Moisture_keys):
        cat_map[f]="Moisture"
    elif any(k in s for k in temp_keys):
        cat_map[f]="Temperature"
    elif any(k in s for k in rad_keys):
        cat_map[f]="Radiation"
    elif any(k in s for k in drought_keys):
        cat_map[f]="Drought"
cat_order=["Moisture","Temperature","Radiation","Drought"]
final_factors=[f for c in cat_order for f in base_factors if cat_map.get(f)==c]
gdf_na=gpd.read_file(na_shp_path)
gdf_eu=gpd.read_file(eu_shp_path)
cand_fields=["zone_name","big7","ZONE","Class","class","class_simple","Koppen","KOPPEN","NAME","name"]
na_field=None
for f in cand_fields:
    if f in gdf_na.columns:
        na_field=f
        break
eu_field=None
for f in cand_fields:
    if f in gdf_eu.columns:
        eu_field=f
        break
order=["A","B","C","Dfa","Dfb","D other","E"]
gdf_na=gdf_na[gdf_na[na_field].isin(order)].copy()
gdf_eu=gdf_eu[gdf_eu[eu_field].isin(order)].copy()
gdf_na["zone_name"]=gdf_na[na_field]
gdf_eu["zone_name"]=gdf_eu[eu_field]
gdf=pd.concat([gdf_na,gdf_eu],ignore_index=True)
name_map={"A":"Tropical","B":"Arid","C":"Temperate","Dfa":"Dfa","Dfb":"Dfb","D other":"D other","E":"Polar"}
zones=[z for z in order if z in set(gdf["zone_name"])]
def zone_mask(z):
    geom=list(gdf[gdf["zone_name"]==z].geometry)
    if len(geom)==0:
        return np.zeros((ny,nx),dtype=bool)
    arr=rasterize([(g,1) for g in geom],out_shape=(ny,nx),transform=transform,fill=0,all_touched=True,dtype="uint8")
    arr=np.flipud(arr)
    return arr.astype(bool)
zone_masks={z:zone_mask(z) for z in zones}
records=[]
for z in zones:
    m=zone_masks[z]
    for f in final_factors:
        var="delta_"+f
        arr_list=[]
        for ds in [ds1,ds2]:
            if var in ds.data_vars:
                da=ds[var]
                da=da.sel(window_center_year=slice(1987,2006))
                arr=da.values
                t_axes=[da.dims.index(d) for d in da.dims if d not in ("latitude","longitude")]
                if len(t_axes)>0:
                    arr=np.nanmedian(arr,axis=tuple(t_axes))
                arr_list.append(arr)
        if len(arr_list)==0:
            med=np.nan
        else:
            arr_all=np.stack(arr_list,axis=0)
            arr_med=np.nanmedian(arr_all,axis=0)
            vals=arr_med[m]
            med=np.nanmedian(vals) if vals.size>0 else np.nan
        records.append((z,f,med))
df=pd.DataFrame(records,columns=["zone_name","factor","median_delta_r_diff"])
def sample_colors(cmap_name,n,lo=0.35,hi=0.75):
    cmap=plt.cm.get_cmap(cmap_name)
    xs=np.linspace(lo,hi,n)
    return [cmap(x) for x in xs]
cat_cmap={"Moisture":"Blues","Temperature":"Greens","Radiation":"Purples","Drought":"YlOrBr"}
cat_range={"Moisture":(0.4,0.75),"Temperature":(0.4,0.7),"Radiation":(0.45,0.75),"Drought":(0.45,0.7)}
items_by_cat={c:[v for v in final_factors if cat_map.get(v)==c] for c in cat_order}
colors_by_cat={c:sample_colors(cat_cmap[c],len(items_by_cat[c]),*cat_range[c]) for c in cat_order}
color_map={v:col for c in cat_order for v,col in zip(items_by_cat[c],colors_by_cat[c])}
cat_seq=[cat_map[f] for f in final_factors]
bounds=[i-0.5 for i in range(1,len(final_factors)) if cat_seq[i]!=cat_seq[i-1]]
data_by_zone={}
for z in zones:
    s=df[df["zone_name"]==z].set_index("factor")["median_delta_r_diff"]
    s=s.reindex(final_factors)
    data_by_zone[z]=s
all_vals=[]
for z in zones:
    all_vals.append(data_by_zone[z].values.astype(float))
all_vals=np.concatenate(all_vals)
all_vals=all_vals[~np.isnan(all_vals)]
vmin=float(all_vals.min()) if all_vals.size>0 else None
vmax=float(all_vals.max()) if all_vals.size>0 else None
if vmin>0:vmin=0
if vmax<0:vmax=0
def disp_name(s):
    return s.replace("_68","").replace("_summer","")
fig,axes=plt.subplots(2,4,figsize=(24,14))
axes=axes.ravel()
for idx,z in enumerate(zones[:7]):
    ax=axes[idx]
    s=data_by_zone[z]
    x=np.arange(len(final_factors))
    ax.bar(x,np.nan_to_num(s.values.astype(float)),color=[color_map[f] for f in final_factors])
    for b in bounds:
        ax.axvline(b,ls="--",lw=1.2,color="k",alpha=0.5)
    ax.axhline(0,lw=1.5,color="k",alpha=0.6)
    ax.set_xlim(-0.5,len(final_factors)-0.5)
    if vmin is not None:
        ax.set_ylim(vmin,vmax)
    ax.set_xticks([])
    if idx in [0,4]:
        ax.set_ylabel(r"$\Delta r = r_{\mathrm{diff}} - r_{\mathrm{partial}}$",fontsize=22)
    else:
        ax.set_yticklabels([])
    ax.text(0.02,0.93,name_map.get(z,z),transform=ax.transAxes,ha="left",va="top",fontsize=24,fontweight="bold")
for j in range(len(zones[:7]),7):
    axes[j].axis("off")
axes[7].axis("off")
yt=axes[0].get_yticks()
axes[4].set_yticks(yt)
handles=[]
labels=[]
for cat in cat_order:
    items=items_by_cat[cat]
    if items:
        handles.append(Patch(facecolor="none",edgecolor="none"))
        labels.append(f"$\\bf{{{cat}}}$")
        for v in items:
            handles.append(Patch(facecolor=color_map[v],edgecolor="none"))
            labels.append(disp_name(v))
axes[7].legend(handles,labels,ncol=2,fontsize=20,loc="center",frameon=False,handlelength=1.2,handletextpad=0.6,borderpad=0.3)
fig.tight_layout()
fig.savefig(png_bar,bbox_inches="tight")
plt.close(fig)
print("图已保存：",os.path.abspath(png_bar))
