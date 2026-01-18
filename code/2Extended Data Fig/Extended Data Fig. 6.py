# -*- coding: utf-8 -*-
import os,re,glob,numpy as np,xarray as xr,geopandas as gpd,matplotlib.pyplot as plt
from rasterio.transform import from_bounds
from rasterio.features import rasterize
from shapely.geometry import box
import matplotlib as mpl
recon_path=r"4TRW/tree_ring4 - AgeDepSpline(EPS0.85).nc"
model_dir=r"6sample data/npp"
shp_path=r"map/Koppen_1991_2020_NA_big7.shp"
out_dir=r"results/2Extended Data Fig"
out_png=os.path.join(out_dir,"Extended Data Fig. 6.png")
YEAR_MIN,YEAR_MAX=1950,2020
lon_min,lon_max=-167.27,-55.00
lat_min,lat_max=9.58,74.99
WIN=20
MIN_N=10
mpl.rcParams['font.family']='Times New Roman'
mpl.rcParams['axes.unicode_minus']=False
mpl.rcParams['font.size']=18
label_map={1:"A",2:"B",3:"C",4:"Dfa",5:"Dfb"}
base_classes=[1,2,3,4,5]
agg_defs={102:("Warm",[1,2,3,4,5]),103:("Cold",[6,7]),104:("All",[1,2,3,4,5,6,7])}
plot_order=[1,2,3,4,5,102,103,104]
legend_pos={1:"lower right",2:"lower right",3:"lower center",4:"lower right",5:"lower right",102:"lower right",103:"lower right",104:"lower right"}
def rasterize_classes(shp_path,lon,lat):
    gdf=gpd.read_file(shp_path)
    if gdf.crs is None:gdf.set_crs(epsg=4326,inplace=True)
    elif gdf.crs.to_epsg()!=4326:gdf=gdf.to_crs(epsg=4326)
    gdf=gpd.clip(gdf,box(lon_min,lat_min,lon_max,lat_max))
    prefer=["code","CODE","Code","class","Class","CLASS","label","Label"]
    col=next((c for c in prefer if c in gdf.columns),None)
    if col is None:col=next(c for c in gdf.columns if c.lower() not in ("geometry","geom","id","fid","ogc_fid"))
    if not np.issubdtype(gdf[col].dtype,np.number):
        name2code={"A 热带":1,"B 干旱":2,"C 温带":3,"Dfa 冷带全湿炎夏":4,"Dfb 冷带全湿暖夏":5,"D 其他冷带":6,"D other":6,"E 寒带":7,"E 极地":7}
        gdf["__code__"]=gdf[col].map(name2code).astype("Int16")
    else:gdf["__code__"]=gdf[col].astype("Int16")
    gdf=gdf.dropna(subset=["__code__"])
    transform=from_bounds(float(lon.min()),float(lat.min()),float(lon.max()),float(lat.max()),len(lon),len(lat))
    shapes=[(geom,int(v)) for geom,v in zip(gdf.geometry,gdf["__code__"])]
    grid=rasterize(shapes=shapes,out_shape=(len(lat),len(lon)),transform=transform,fill=0,dtype="int16")
    if lat[0]<lat[-1]:grid=np.flipud(grid)
    return grid
def zone_mask(cls_grid,code):
    return (cls_grid==code) if code in base_classes else np.isin(cls_grid,agg_defs[code][1])
def get_recon_da(ds):
    cand=[v for v in ds.data_vars if 'year' in ds[v].dims]
    prefer=[v for v in cand if ('recon' in ds[v].name.lower()) or ('agedep' in ds[v].name.lower())]
    name=prefer[0] if prefer else (cand[0] if cand else None)
    if name is None:raise ValueError("no yearly target variable")
    return ds[name]
def get_model_var(ds):
    cand=[v for v in ds.data_vars if 'year' in ds[v].dims]
    prefer=[v for v in cand if v.lower() in ('npp','gpp')]
    return prefer[0] if prefer else (cand[0] if cand else None)
def rolling_corr(a,b,win,min_n):
    a=np.asarray(a);b=np.asarray(b);T=a.shape[0];out_len=T-win+1
    out=np.full((out_len,)+a.shape[1:],np.nan,dtype=np.float32)
    for t in range(out_len):
        X=a[t:t+win];Y=b[t:t+win]
        mask=np.isfinite(X)&np.isfinite(Y);n=mask.sum(axis=0)
        Xz=np.where(mask,X,0.0);Yz=np.where(mask,Y,0.0)
        sx=Xz.sum(axis=0);sy=Yz.sum(axis=0)
        sxx=(Xz*Xz).sum(axis=0);syy=(Yz*Yz).sum(axis=0);sxy=(Xz*Yz).sum(axis=0)
        num=n*sxy-sx*sy;den=np.sqrt((n*sxx-sx*sx)*(n*syy-sy*sy))
        v=(n>=min_n)&(den>0);r=np.full_like(den,np.nan,dtype=np.float32);r[v]=num[v]/den[v];out[t]=r
    return out
def top3(d):
    if not d:return []
    its=[]
    for k,(x,y) in d.items():
        if np.isfinite(y).any():its.append((k,float(np.nanmean(y))))
    its.sort(key=lambda z:z[1],reverse=True)
    return [k for k,_ in its[:3]]
def collect_series(R,years_roll,cls_grid,codes,series_dict,model_name):
    for code in codes:
        msk=zone_mask(cls_grid,code)
        ts=[float(np.nanmean(R[t][msk & np.isfinite(R[t])])) if np.any(msk & np.isfinite(R[t])) else np.nan for t in range(R.shape[0])]
        series_dict[code][model_name]=(years_roll,np.array(ts,dtype=float))
def draw_block(ax,code,d,show_xlabel,show_ylabel,tag_text,legend_loc):
    t3=top3(d)
    if not t3:
        ax.text(0.5,0.5,"No data",ha="center",va="center",fontsize=12);ax.set_axis_off();return
    vmin_list=[];vmax_list=[]
    used=set()
    for m in t3:
        x,y=d[m];lab=m
        if lab in used:
            c=2
            while f"{lab} ({c})" in used:c+=1
            lab=f"{lab} ({c})"
        used.add(lab)
        ax.plot(x,y,label=lab,linewidth=1)
        if np.isfinite(y).any():
            vmin_list.append(np.nanmin(y));vmax_list.append(np.nanmax(y))
    xs_sets=[set(d[m][0].tolist()) for m in t3]
    inter_years=sorted(list(set.intersection(*xs_sets))) if xs_sets else []
    if len(inter_years)>=2:
        Ystack=[]
        for m in t3:
            xm,ym=d[m]
            idx=np.searchsorted(xm,inter_years)
            ok=(idx<len(xm))&(xm[idx]==np.array(inter_years))
            arr=np.full(len(inter_years),np.nan)
            arr[ok]=ym[idx[ok]]
            Ystack.append(arr)
        Ystack=np.vstack(Ystack)
        mean_curve=np.nanmean(Ystack,axis=0)
        ax.plot(inter_years,mean_curve,label="Mean of top3",linewidth=2)
        if np.isfinite(mean_curve).any():
            vmin_list.append(np.nanmin(mean_curve));vmax_list.append(np.nanmax(mean_curve))
    if vmin_list and vmax_list:
        vmin=float(np.nanmin(vmin_list));vmax=float(np.nanmax(vmax_list))
        Rbase=max(vmax-vmin,0.2)
        mid=0.5*(vmin+vmax)
        ax.set_ylim(mid-Rbase/2,mid+Rbase/2)
    ax.margins(x=0.01)
    if show_xlabel:ax.set_xlabel("Year")
    else:ax.set_xlabel("")
    if show_ylabel:ax.set_ylabel(tag_text,fontsize=20,labelpad=8)
    else:ax.set_ylabel("")
    ax.text(0.02,0.92,(label_map[code] if code in base_classes else agg_defs[code][0]),
            transform=ax.transAxes,ha="left",va="top",fontsize=15,fontweight="bold")
    lg=ax.legend(fontsize=10,frameon=True,loc=legend_loc,borderpad=0.05,handlelength=1.2,handletextpad=0.4,markerscale=0.6)
    if lg is not None:
        try:lg.get_frame().set_boxstyle("sawtooth",pad=0.05)
        except Exception:pass
def main():
    os.makedirs(out_dir,exist_ok=True)
    dsA=xr.open_dataset(recon_path)
    daA=get_recon_da(dsA).sel(latitude=slice(lat_min,lat_max),longitude=slice(lon_min,lon_max)).sel(year=slice(YEAR_MIN,YEAR_MAX))
    yearsA=daA['year'].values.astype(int)
    latA,lonA=daA['latitude'].values,daA['longitude'].values
    cls_grid=rasterize_classes(shp_path,lonA,latA)
    files=sorted(glob.glob(os.path.join(model_dir,"*.nc")));files=[f for f in files if not os.path.basename(f).startswith("2")]
    zone_series_diff={code:{} for code in plot_order}
    zone_series_raw={code:{} for code in plot_order}
    for fp in files:
        model_name=os.path.splitext(os.path.basename(fp))[0]
        dsB=xr.open_dataset(fp);varB=get_model_var(dsB)
        if varB is None:dsB.close();continue
        try:daB=dsB[varB].sel(latitude=slice(lat_min,lat_max),longitude=slice(lon_min,lon_max)).sel(year=slice(YEAR_MIN,YEAR_MAX))
        except Exception:dsB.close();continue
        yearsB=daB['year'].values.astype(int)
        common=np.intersect1d(yearsA,yearsB);common=common[(common>=YEAR_MIN)&(common<=YEAR_MAX)]
        if common.size<WIN+1:dsB.close();continue
        A=daA.sel(year=common);B=daB.sel(year=common)
        if (B.sizes.get('latitude')!=A.sizes['latitude']) or (B.sizes.get('longitude')!=A.sizes['longitude']) or (not np.allclose(B['latitude'],A['latitude'])) or (not np.allclose(B['longitude'],A['longitude'])):B=B.interp(latitude=A['latitude'],longitude=A['longitude'])
        A=A.load().values.astype(np.float32);B=B.load().values.astype(np.float32);dsB.close()
        Ad=np.diff(A,axis=0);Bd=np.diff(B,axis=0)
        yrs_diff=common[1:]
        if yrs_diff.size>=WIN:
            R_diff=rolling_corr(Ad,Bd,WIN,MIN_N)
            yrs_roll_diff=yrs_diff[WIN-1:]
            collect_series(R_diff,yrs_roll_diff,cls_grid,plot_order,zone_series_diff,model_name)
        if common.size>=WIN:
            R_raw=rolling_corr(A,B,WIN,MIN_N)
            yrs_roll_raw=common[WIN-1:]
            collect_series(R_raw,yrs_roll_raw,cls_grid,plot_order,zone_series_raw,model_name)
    fig,axes=plt.subplots(4,4,figsize=(16,12),constrained_layout=True)
    for idx,code in enumerate(plot_order):
        r=idx//4;c=idx%4
        ax=axes[r,c]
        d=zone_series_diff[code]
        draw_block(ax,code,d,show_xlabel=False,show_ylabel=(c==0),tag_text="r_diff",legend_loc=legend_pos.get(code,"lower right"))
    for idx,code in enumerate(plot_order):
        r=2+idx//4;c=idx%4
        ax=axes[r,c]
        d=zone_series_raw[code]
        draw_block(ax,code,d,show_xlabel=(r==3),show_ylabel=(c==0),tag_text="r_raw",legend_loc=legend_pos.get(code,"lower right"))
    fig.savefig(out_png,dpi=300,bbox_inches="tight")
    plt.close(fig);dsA.close()
if __name__=="__main__":main()
