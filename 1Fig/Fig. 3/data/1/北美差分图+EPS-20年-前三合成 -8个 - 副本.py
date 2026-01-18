# -*- coding: utf-8 -*-
import os,re,glob,numpy as np,xarray as xr,geopandas as gpd,matplotlib.pyplot as plt
from rasterio.transform import from_bounds
from rasterio.features import rasterize
from shapely.geometry import box
import matplotlib as mpl
recon_path=r"D:\1700\3\tree_ring4 - AgeDepSpline(EPS0.85).nc"
model_dir=r"D:\Model data\NPP-TRENDY\nppS2\9-8\1700-2"
shp_path=r"D:\\map2\\世界底图\\Koppen_1991_2020_NA_big7.shp"
out_dir=r"D:\1700\北美图4\合成线段\1"
out_png=os.path.join(out_dir,"北美差分图+EPS-20年-前八合成.png")
YEAR_MIN,YEAR_MAX=1950,2020
lon_min,lon_max=-167.27,-55.00
lat_min,lat_max=9.58,74.99
WIN=20
MIN_N=10
A_PAD_FRAC=0.04
label_map={1:"A",2:"B",3:"C",4:"Dfa",5:"Dfb"}
base_classes=[1,2,3,4,5]
agg_defs={102:("Warm",[1,2,3,4,5]),103:("Cold",[6,7]),104:("All",[1,2,3,4,5,6,7])}
plot_order=[1,2,3,4,5,102,103,104]
legend_pos={1:"lower right",2:"lower right",3:"lower center",4:"lower center",5:"lower right",102:"lower right",103:"lower right",104:"lower right"}
mpl.rcParams['font.family']='Times New Roman'
mpl.rcParams['axes.unicode_minus']=False
mpl.rcParams['font.size']=16
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
    else:
        gdf["__code__"]=gdf[col].astype("Int16")
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
    prefer=[v for v in cand if ('recon' in v.lower()) or ('agedep' in v.lower())]
    name=prefer[0] if prefer else (cand[0] if cand else None)
    if name is None:raise ValueError("no yearly target variable")
    return ds[name]
def get_model_var(ds):
    cand=[v for v in ds.data_vars if 'year' in ds[v].dims]
    prefer=[v for v in cand if v.lower() in ('npp','gpp')]
    return prefer[0] if prefer else (cand[0] if cand else None)
def rolling_corr_from_diff(a,b,win,min_n):
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
def main():
    os.makedirs(out_dir,exist_ok=True)
    dsA=xr.open_dataset(recon_path)
    daA=get_recon_da(dsA).sel(latitude=slice(lat_min,lat_max),longitude=slice(lon_min,lon_max)).sel(year=slice(YEAR_MIN,YEAR_MAX))
    yearsA=daA['year'].values.astype(int)
    latA,lonA=daA['latitude'].values,daA['longitude'].values
    cls_grid=rasterize_classes(shp_path,lonA,latA)
    files=sorted(glob.glob(os.path.join(model_dir,"*.nc")));files=[f for f in files if not os.path.basename(f).startswith("2")]
    zone_series_all={code:{} for code in plot_order}
    for fp in files:
        name=os.path.splitext(os.path.basename(fp))[0]
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
        if yrs_diff.size<WIN:continue
        R=rolling_corr_from_diff(Ad,Bd,WIN,MIN_N)
        yrs_roll=yrs_diff[WIN-1:]
        for code in plot_order:
            msk=zone_mask(cls_grid,code)
            ts=[float(np.nanmean(R[t][msk & np.isfinite(R[t])])) if np.any(msk & np.isfinite(R[t])) else np.nan for t in range(R.shape[0])]
            zone_series_all[code][name]=(yrs_roll,np.array(ts,dtype=float))
    def top3(d):
        if not d:return []
        its=[]
        for k,(x,y) in d.items():
            if np.isfinite(y).any():its.append((k,float(np.nanmean(y))))
        its.sort(key=lambda z:z[1],reverse=True)
        return [k for k,_ in its[:1]]
    t3z={}
    vminz={}
    vmaxz={}
    for code in plot_order:
        t3=top3(zone_series_all[code]);t3z[code]=t3
        ys=[]
        for m in t3:
            y=zone_series_all[code][m][1]
            if np.isfinite(y).any():ys+=[np.nanmin(y),np.nanmax(y)]
        if ys:
            vminz[code]=float(np.nanmin(ys));vmaxz[code]=float(np.nanmax(ys))
        else:
            vminz[code]=np.nan;vmaxz[code]=np.nan
    rng=[vmaxz[c]-vminz[c] for c in plot_order if np.isfinite(vminz.get(c,np.nan)) and np.isfinite(vmaxz.get(c,np.nan)) and vmaxz[c]>vminz[c]]
    Rbase=max(rng) if rng else 0.2
    fig,axes=plt.subplots(2,4,figsize=(16,6.5),constrained_layout=True);axes=axes.ravel()
    for i,code in enumerate(plot_order):
        ax=axes[i];d=zone_series_all[code];t3=t3z[code]
        if not t3:
            ax.text(0.5,0.5,"No data",ha="center",va="center",fontsize=10);ax.set_axis_off();continue
        used=set();all_x=[];all_y=[]
        for m in t3:
            x,y=d[m];lab=m
            if lab in used:
                c=2
                while f"{lab} ({c})" in used:c+=1
                lab=f"{lab} ({c})"
            used.add(lab)
            ax.plot(x,y,label=lab,linewidth=1)
            k=np.isfinite(y);all_x.extend(x[k]);all_y.extend(y[k])
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
            ax.plot(inter_years,mean_curve,linewidth=2,label="Mean of top3")
            tag=label_map[code] if code in base_classes else agg_defs[code][0]
            csv_path=os.path.join(out_dir,f"{tag}.csv")
            with open(csv_path,"w",encoding="utf-8") as f:
                f.write("year,mean_r\n")
                for xv,yv in zip(inter_years,mean_curve):
                    if np.isfinite(yv):f.write(f"{int(xv)},{yv:.6f}\n")
        vmin=vminz[code];vmax=vmaxz[code]
        if np.isfinite(vmin) and np.isfinite(vmax):mid=0.5*(vmin+vmax);ymin=mid-Rbase/2;ymax=mid+Rbase/2
        else:mid=0.0;ymin=mid-Rbase/2;ymax=mid+Rbase/2
        ax.set_ylim(ymin,ymax);ax.margins(x=0.01);ax.set_xlabel("Year");ax.set_ylabel("Mean r")
        tag=label_map[code] if code in base_classes else agg_defs[code][0]
        ax.text(0.02,0.92,tag,transform=ax.transAxes,ha="left",va="top",fontsize=11,fontweight="bold")
        lg=ax.legend(fontsize=8,frameon=True,loc=legend_pos.get(code,"lower right"),borderpad=0.05,handlelength=1.2,handletextpad=0.4,markerscale=0.6)
        if lg is not None:
            try:lg.get_frame().set_boxstyle("sawtooth",pad=0.05)
            except Exception:pass
    for j in range(len(plot_order),len(axes)):fig.delaxes(axes[j])
    fig.savefig(out_png,dpi=300,bbox_inches="tight")
    plt.close(fig);dsA.close()
if __name__=="__main__":main()
