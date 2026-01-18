import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
mpl.rcParams['font.family']='Times New Roman'
mpl.rcParams['figure.dpi']=300
npp_file=r"D:\all\data\npp\1_ISBA-CTRIP.nc"
attr_file=r"D:\all\2Extended Data Fig\Extended Data Fig. 3\data\基本属性.csv"
trw_file=r"D:\all\2Extended Data Fig\Extended Data Fig. 3\data\AgeDepSpline.csv"
ds_npp=xr.open_dataset(npp_file)
npp=ds_npp['npp']
years_npp=ds_npp['year'].values
lat_npp_full=ds_npp['latitude'].values
lon_npp_full=ds_npp['longitude'].values
attr=pd.read_csv(attr_file)
df_trw=pd.read_csv(trw_file)
years_trw=df_trw['year'].values
years_target=np.intersect1d(years_npp,years_trw)
years_target=years_target[(years_target>=1970)&(years_target<=2010)]
npp=npp.sel(year=years_target)
df_trw=df_trw.set_index('year').loc[years_target]
na_lon_min,na_lon_max=-167.27,-55.00
na_lat_min,na_lat_max=9.58,74.99
eu_lon_min,eu_lon_max=-11.0,57.0
eu_lat_min,eu_lat_max=30.0,75.0
lon_min_global=min(na_lon_min,eu_lon_min)
lon_max_global=max(na_lon_max,eu_lon_max)
lat_min_global=min(na_lat_min,eu_lat_min)
lat_max_global=max(na_lat_max,eu_lat_max)
idx_lat=np.where((lat_npp_full>=lat_min_global)&(lat_npp_full<=lat_max_global))[0]
idx_lon=np.where((lon_npp_full>=lon_min_global)&(lon_npp_full<=lon_max_global))[0]
lat_npp=lat_npp_full[idx_lat]
lon_npp=lon_npp_full[idx_lon]
npp=npp.isel(latitude=idx_lat,longitude=idx_lon)
npp_arr=npp.values
ny=len(lat_npp)
nx=len(lon_npp)
region_mask=np.zeros((ny,nx),dtype=bool)
for i in range(ny):
    for j in range(nx):
        la=lat_npp[i]; lo=lon_npp[j]
        in_na=(la>=na_lat_min) and (la<=na_lat_max) and (lo>=na_lon_min) and (lo<=na_lon_max)
        in_eu=(la>=eu_lat_min) and (la<=eu_lat_max) and (lo>=eu_lon_min) and (lo<=eu_lon_max)
        if in_na or in_eu:
            region_mask[i,j]=True
sites=[]
for idx,row in attr.iterrows():
    name=row['name']
    if name in df_trw.columns:
        la=float(row['latitude']); lo=float(row['longitude'])
        in_na=(la>=na_lat_min) and (la<=na_lat_max) and (lo>=na_lon_min) and (lo<=na_lon_max)
        in_eu=(la>=eu_lat_min) and (la<=eu_lat_max) and (lo>=eu_lon_min) and (lo<=eu_lon_max)
        if in_na or in_eu:
            ts=df_trw[name].values
            if not np.all(np.isnan(ts)):
                sites.append((name,la,lo,ts))
if len(sites)>0:
    rng=np.random.default_rng()
    n_sample=max(1,int(len(sites)*1))
    idx_sample=rng.choice(len(sites),size=n_sample,replace=False)
    sites_sample=[sites[i] for i in idx_sample]
else:
    sites_sample=[]
max_r=10.0
step=0.5
all_dist=[]
all_corr=[]
for name,lat0,lon0,ts_trw in sites_sample:
    mask_t=~np.isnan(ts_trw)
    if mask_t.sum()<2:
        continue
    ts=ts_trw[mask_t]
    ts_mean=ts.mean()
    ts_std=ts.std()
    if ts_std==0:
        continue
    i0=np.argmin(np.abs(lat_npp-lat0))
    j0=np.argmin(np.abs(lon_npp-lon0))
    if region_mask[i0,j0]:
        ts_npp0=npp_arr[:,i0,j0][mask_t]
        if np.sum(~np.isnan(ts_npp0))>=2:
            c0=np.corrcoef(ts,ts_npp0)[0,1]
            if not np.isnan(c0):
                all_dist.append(np.array([0.0]))
                all_corr.append(np.array([c0]))
    idx_i=np.where(np.abs(lat_npp-lat0)<=max_r)[0]
    idx_j=np.where(np.abs(lon_npp-lon0)<=max_r)[0]
    if (len(idx_i)==0) or (len(idx_j)==0):
        continue
    npp_sub=npp_arr[:,idx_i[:,None],idx_j[None,:]]
    npp_sub=npp_sub[mask_t,:,:]
    region_sub=region_mask[np.ix_(idx_i,idx_j)]
    valid_mask=np.sum(~np.isnan(npp_sub),axis=0)>=2
    valid_mask=valid_mask&region_sub
    if not np.any(valid_mask):
        continue
    npp_mean=np.nanmean(npp_sub,axis=0)
    npp_center=npp_sub-npp_mean
    ts_center=ts-ts_mean
    cov=np.nanmean(npp_center*ts_center[:,None,None],axis=0)
    npp_std=np.nanstd(npp_sub,axis=0)
    denom=npp_std*ts_std
    corr=np.where(denom>0,cov/denom,np.nan)
    lat_sub=lat_npp[idx_i]
    lon_sub=lon_npp[idx_j]
    lat2d,lon2d=np.meshgrid(lat_sub,lon_sub,indexing='ij')
    dist=np.maximum(np.abs(lat2d-lat0),np.abs(lon2d-lon0))
    dist_valid=dist[valid_mask]
    corr_valid=corr[valid_mask]
    ok=~np.isnan(corr_valid)
    if np.any(ok):
        all_dist.append(dist_valid[ok])
        all_corr.append(corr_valid[ok])
all_dist=np.concatenate(all_dist)
all_corr=np.concatenate(all_corr)
bin_edges=np.arange(0,10.0+step,step)
x=bin_edges.copy()
median_corr=[]
iqr_low=[]
iqr_high=[]
for k in range(len(bin_edges)):
    if k==0:
        mask=(all_dist==0)
    else:
        r0=bin_edges[k-1]
        r1=bin_edges[k]
        mask=(all_dist>r0)&(all_dist<=r1)
    vals=all_corr[mask]
    vals=vals[~np.isnan(vals)]
    if vals.size==0:
        median_corr.append(np.nan)
        iqr_low.append(np.nan)
        iqr_high.append(np.nan)
    else:
        m=np.median(vals)
        q1=np.percentile(vals,25)
        q3=np.percentile(vals,75)
        median_corr.append(float(m))
        iqr_low.append(float(q1))
        iqr_high.append(float(q3))
median_corr=np.array(median_corr)
iqr_low=np.array(iqr_low)
iqr_high=np.array(iqr_high)
y_min=np.nanmin(iqr_low)
y_max=np.nanmax(iqr_high)
if np.isfinite(y_min) and np.isfinite(y_max):
    margin=0.05*(y_max-y_min+1e-6)
    y_min_plot=y_min-margin
    y_max_plot=y_max+margin
else:
    y_min_plot=None
    y_max_plot=None
fig,ax=plt.subplots(figsize=(7,5))
ax.fill_between(x,iqr_low,iqr_high,color='#1f77b4',alpha=0.18)
ax.plot(x,median_corr,color='#1f77b4',linewidth=2.3)
ax.axhline(0,color='black',linewidth=1,linestyle='--')
if y_min_plot is not None:
    ax.set_ylim(y_min_plot,y_max_plot)
ax.set_xlim(0,10)
ax.set_xlabel("Radius (°)",fontsize=14)
ax.set_ylabel("Correlation (r)",fontsize=14)
ax.tick_params(axis='both',labelsize=12,length=4,width=1)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(1)
ax.grid(axis='y',linestyle=':',linewidth=0.6,alpha=0.4)
custom_line=Line2D([0],[0],color='#1f77b4',linewidth=2.3)
ax.legend([custom_line],['Median r'],loc='upper right',frameon=False,fontsize=12,handlelength=2.5,handletextpad=0.6,borderpad=0.2)
plt.tight_layout()
plt.savefig(r"D:\all\2Extended Data Fig\Extended Data Fig. 3\Extended Data Fig. 3.png",bbox_inches='tight')
