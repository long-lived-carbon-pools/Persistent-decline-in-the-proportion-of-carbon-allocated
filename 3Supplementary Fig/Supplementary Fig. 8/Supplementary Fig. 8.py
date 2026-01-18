# -*- coding: utf-8 -*- 
import os,re,numpy as np,pandas as pd,xarray as xr,geopandas as gpd,matplotlib.pyplot as plt,matplotlib as mpl
from rasterio.transform import from_bounds
from rasterio.features import rasterize
from shapely.geometry import box
mpl.rcParams['font.family']='Times New Roman'
mpl.rcParams['axes.unicode_minus']=False
mpl.rcParams['font.size']=18
nc_paths=[r"D:\all\3Supplementary Fig\Supplementary Fig. 8\data\10climate_corr_maps_1981_2010.nc",r"D:\all\3Supplementary Fig\Supplementary Fig. 8\data\10corr_diff_1981_2010.nc"]
na_shp=r"D:\all\data\世界底图\Koppen_1991_2020_NA_big7.shp"
eu_shp=r"D:\all\data\世界底图\欧洲7类.shp"
out_png_best10=r"D:\all\3Supplementary Fig\Supplementary Fig. 8\Supplementary Fig. 8.png"
na_lon_min,na_lon_max=-167.27,-55.00
na_lat_min,na_lat_max=9.58,74.99
eu_lon_min,eu_lon_max=-11.0,57
eu_lat_min,eu_lat_max=30.0,75.0
base_classes=[1,2,3,4,5,6,7]
label_map={1:"Tropical",2:"Arid",3:"Temperate",4:"Dfa",5:"Dfb",6:"D other",7:"E"}
agg_defs={102:("Warm",[1,2,3,4,5]),103:("Cold",[6,7]),104:("All",[1,2,3,4,5,6,7])}
valid_prefixes=["1_","2_","3_","4_","5_","6_"]
def rasterize_classes(shp_path,lon,lat,bounds):
    gdf=gpd.read_file(shp_path)
    if gdf.crs is None:gdf.set_crs(epsg=4326,inplace=True)
    elif getattr(gdf.crs,"to_epsg",lambda:None)()!=4326:gdf=gdf.to_crs(epsg=4326)
    gdf=gpd.clip(gdf,box(*bounds))
    if "code" in gdf.columns:gdf["__code__"]=gdf["code"].astype("Int16")
    else:
        prefer=["class","Class","CLASS","label","Label"]
        cls_col=next((c for c in prefer if c in gdf.columns),None)
        if cls_col is None:cls_col=next(c for c in gdf.columns if str(c).lower() not in ("geometry","geom","id","fid","ogc_fid"))
        name2code={"A":1,"B":2,"C":3,"Dfa":4,"Dfb":5,"D other":6,"E":7}
        gdf["__code__"]=gdf[cls_col].map(name2code).astype("Int16")
    gdf=gdf.dropna(subset=["__code__"])
    transform=from_bounds(float(lon.min()),float(lat.min()),float(lon.max()),float(lat.max()),len(lon),len(lat))
    shapes=[(geom,int(v)) for geom,v in zip(gdf.geometry,gdf["__code__"])]
    grid=rasterize(shapes=shapes,out_shape=(len(lat),len(lon)),transform=transform,fill=0,dtype="int16")
    if lat[0]<lat[-1]:grid=np.flipud(grid)
    return grid
def collect_values(ds,bounds,shp_path):
    vals_by_class={}
    for v in ds.data_vars:
        if not any(str(v).startswith(p) for p in valid_prefixes):continue
        da=ds[v].sel(latitude=slice(bounds[1],bounds[3]),longitude=slice(bounds[0],bounds[2])).transpose("latitude","longitude")
        if da.size==0:continue
        arr=da.values.astype(np.float64)
        lat=da["latitude"].values
        lon=da["longitude"].values
        cls=rasterize_classes(shp_path,lon,lat,(bounds[0],bounds[1],bounds[2],bounds[3]))
        for c in base_classes:
            m=np.isfinite(arr)&(cls==c)
            if np.any(m):vals_by_class.setdefault((v,c),[]).append(arr[m].ravel())
    out={}
    for k,parts in vals_by_class.items():
        out[k]=np.concatenate(parts) if len(parts)>0 else np.array([])
    return out
def main():
    data_by_class={}
    for nc in nc_paths:
        ds=xr.open_dataset(nc)
        na=collect_values(ds,(na_lon_min,na_lat_min,na_lon_max,na_lat_max),na_shp)
        eu=collect_values(ds,(eu_lon_min,eu_lat_min,eu_lon_max,eu_lat_max),eu_shp)
        for k,v in na.items():
            if v.size>0:data_by_class.setdefault(k,[]).append(v)
        for k,v in eu.items():
            if v.size>0:data_by_class.setdefault(k,[]).append(v)
        ds.close()
    merged_data={}
    for k,parts in data_by_class.items():
        merged=np.concatenate(parts)
        merged=merged[np.isfinite(merged)]
        if merged.size>0:merged_data[k]=merged
    zones=[1,2,3,4,5,102,103,104]
    zone_labels=[label_map[1],label_map[2],label_map[3],label_map[4],label_map[5],agg_defs[102][0],agg_defs[103][0],agg_defs[104][0]]
    groups=[]
    best_names=[]
    for z in zones:
        cand={}
        if z in base_classes:
            for v in set([k[0] for k in merged_data.keys() if k[1]==z]):
                cand[v]=merged_data.get((v,z),np.array([]))
        else:
            clist=agg_defs[z][1]
            for v in set([k[0] for k in merged_data.keys() if k[1] in clist]):
                parts=[merged_data.get((v,c),np.array([])) for c in clist if (v,c) in merged_data]
                parts=[p for p in parts if p.size>0]
                if len(parts)>0:cand[v]=np.concatenate(parts)
        if len(cand)==0:
            groups.append(np.array([]));best_names.append("")
            continue
        best_v=max(cand.keys(),key=lambda k:abs(np.nanmedian(cand[k])) if cand[k].size>0 else -np.inf)
        short=re.sub(r'^[1-6]_','',str(best_v))
        groups.append(cand[best_v]);best_names.append(short)
    valid=[(g,lab,bn) for g,lab,bn in zip(groups,zone_labels,best_names) if g.size>0]
    if len(valid)==0:
        fig=plt.figure(figsize=(6,4))
        os.makedirs(os.path.dirname(out_png_best10),exist_ok=True)
        fig.savefig(out_png_best10,dpi=300,bbox_inches='tight');plt.close(fig);return
    series,lab,bnames=zip(*valid)
    fig=plt.figure(figsize=(max(12,len(series)*1.0),6))
    ax=plt.gca()
    bp=ax.boxplot(series,labels=lab,patch_artist=True,showfliers=False,whis=1.5,medianprops=dict(linewidth=2),boxprops=dict(linewidth=1.6),whiskerprops=dict(linewidth=1.6),capprops=dict(linewidth=1.6))
    for b in bp['boxes']:b.set_facecolor((0,0,0,0.06))
    ax.grid(axis='y',linestyle='--',linewidth=0.8,alpha=0.5)
    ax.tick_params(axis='x',rotation=0,length=0)
    ymax=np.nanmax([np.nanmax(s) for s in series])
    ymin=np.nanmin([np.nanmin(s) for s in series])
    dy=(ymax-ymin)*0.06 if np.isfinite(ymax) and np.isfinite(ymin) else 0.1
    y_text=ymax+dy*1.2
    rng=np.random.default_rng(2025)
    for i,vals in enumerate(series,1):
        if vals.size==0:continue
        xj=i+(rng.random(vals.size)-0.5)*0.18
        ax.scatter(xj,vals,s=6,alpha=0.35,linewidths=0)
        med=float(np.nanmedian(vals))
        if np.isfinite(med):ax.text(i,med,str(round(med,2)),ha='center',va='bottom',fontsize=14)
    for i,name in enumerate(bnames,1):
        ax.text(i,y_text,name,ha='center',va='bottom',fontsize=14)

    ax.set_xlabel("Climate Zones",fontsize=22)
    ax.set_ylabel("r",fontsize=22)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png_best10),exist_ok=True)
    plt.savefig(out_png_best10,dpi=300,bbox_inches='tight')
    plt.close(fig)
if __name__=="__main__":
    main()
