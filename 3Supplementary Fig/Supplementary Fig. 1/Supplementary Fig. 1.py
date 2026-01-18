# -*- coding: utf-8 -*-
import os,numpy as np,xarray as xr,matplotlib.pyplot as plt,geopandas as gpd
from matplotlib.colors import TwoSlopeNorm
import matplotlib.gridspec as gridspec
recon_path=r"D:\all\data\tree_ring4 - AgeDepSpline(EPS0.85).nc"
model_path=r"D:\all\data\npp\1_ISBA-CTRIP.nc"
world_shp=r"D:\all\data\世界底图\World.shp"
out_png=r"D:\all\3Supplementary Fig\Supplementary Fig. 1\Supplementary Fig. 1.png"
lonlat_list=[(-167.27,-55.00,9.58,74.99),(-11.0,57.0,30.0,75.0)]
WIN=20
MIN_N=10
plt.rcParams['font.family']='Times New Roman'
plt.rcParams['axes.unicode_minus']=False
def get_recon_da(ds):
    cand=[v for v in ds.data_vars if 'year' in ds[v].dims]
    prefer=[v for v in cand if ('recon' in ds[v].name.lower()) or ('agedep' in ds[v].name.lower())]
    return ds[prefer[0] if prefer else cand[0]]
def get_model_var(ds):
    cand=[v for v in ds.data_vars if 'year' in ds[v].dims]
    prefer=[v for v in cand if v.lower() in ('npp','gpp')]
    return prefer[0] if prefer else cand[0]
def rolling_corr_from_diff(a,b,win,min_n):
    a=np.asarray(a);b=np.asarray(b)
    T=a.shape[0];L=T-win+1
    out=np.full((L,)+a.shape[1:],np.nan,np.float32)
    for t in range(L):
        X=a[t:t+win];Y=b[t:t+win]
        m=np.isfinite(X)&np.isfinite(Y)
        n=m.sum(axis=0)
        Xz=np.where(m,X,0.0);Yz=np.where(m,Y,0.0)
        sx=Xz.sum(axis=0);sy=Yz.sum(axis=0)
        sxx=(Xz*Xz).sum(axis=0);syy=(Yz*Yz).sum(axis=0);sxy=(Xz*Yz).sum(axis=0)
        num=n*sxy-sx*sy
        den=np.sqrt((n*sxx-sx*sx)*(n*syy-sy*sy))
        v=(n>=min_n)&(den>0)
        r=np.full_like(den,np.nan,np.float32);r[v]=num[v]/den[v]
        out[t]=r
    return out
def poly_slope_years(x,y):
    mask=np.isfinite(y)
    if np.count_nonzero(mask)>=2:
        c=np.polyfit(x[mask].astype(float),y[mask].astype(float),1)
        return float(c[0])
    return np.nan
def calc_region(lon_min,lon_max,lat_min,lat_max):
    dsA=xr.open_dataset(recon_path)
    daA_full=get_recon_da(dsA).sel(latitude=slice(lat_min,lat_max),longitude=slice(lon_min,lon_max))
    yearsA_all=daA_full['year'].values.astype(int)
    latA=daA_full['latitude'].values
    lonA=daA_full['longitude'].values
    dsB=xr.open_dataset(model_path)
    varB=get_model_var(dsB)
    daB_full=dsB[varB].sel(latitude=slice(lat_min,lat_max),longitude=slice(lon_min,lon_max))
    yearsB_all=daB_full['year'].values.astype(int)
    periods=[(1950,1980),(1960,1990),(1970,2000),(1980,2010),(1990,2020)]
    P=len(periods)
    H=latA.size;W=lonA.size
    slopes=np.full((P,H,W),np.nan,np.float32)
    for ip,(Y0,Y1) in enumerate(periods):
        maskA=(yearsA_all>=Y0)&(yearsA_all<=Y1)
        maskB=(yearsB_all>=Y0)&(yearsB_all<=Y1)
        yrsA=yearsA_all[maskA]
        yrsB=yearsB_all[maskB]
        common=np.intersect1d(yrsA,yrsB)
        if common.size<WIN+1:continue
        A=daA_full.sel(year=common).values.astype(np.float32)
        B=daB_full.sel(year=common).values.astype(np.float32)
        Ad=np.diff(A,axis=0)
        Bd=np.diff(B,axis=0)
        if Ad.shape[0]<WIN:continue
        R=rolling_corr_from_diff(Ad,Bd,WIN,MIN_N)
        yrs_roll=common[1:][WIN-1:]
        x=yrs_roll.astype(float)
        s=np.full((H,W),np.nan,np.float32)
        for iy in range(H):
            y=R[:,iy,:]
            mask=np.isfinite(y).sum(axis=0)>=2
            idx=np.where(mask)[0]
            for ix in idx:
                s[iy,ix]=poly_slope_years(x,y[:,ix])
        slopes[ip]=s
    dsA.close();dsB.close()
    return slopes,latA,lonA
slopes_list=[];lat_list=[];lon_list=[]
for (l1,l2,a1,a2) in lonlat_list:
    s,la,lo=calc_region(l1,l2,a1,a2)
    slopes_list.append(s);lat_list.append(la);lon_list.append(lo)
vals=[]
for r in range(2):
    s=slopes_list[r]
    if np.isfinite(s).any():vals.append(np.abs(s[np.isfinite(s)]))
allv=np.concatenate(vals)
vabs=np.nanpercentile(allv,95)
if(not np.isfinite(vabs))or(vabs<=0):vabs=0.1
norm=TwoSlopeNorm(vmin=-vabs,vcenter=0.0,vmax=vabs)
world=gpd.read_file(world_shp)
fig=plt.figure(figsize=(10,18))
gs=gridspec.GridSpec(5,2,figure=fig,width_ratios=[1.25,1.25],hspace=0.03,wspace=0.05)
axes=[[fig.add_subplot(gs[i,0]),fig.add_subplot(gs[i,1])] for i in range(5)]
fig.subplots_adjust(left=0.03,right=0.97,top=0.97,bottom=0.10)
period_labels=["1950-1980","1960-1990","1970-2000","1980-2010","1990-2020"]
mappable=None
for p in range(5):
    ax=axes[p][0]
    latA=lat_list[0];lonA=lon_list[0]
    LON,LAT=np.meshgrid(lonA,latA)
    world.plot(ax=ax,color="lightgrey",edgecolor="grey",linewidth=0.4,zorder=0)
    im=ax.pcolormesh(LON,LAT,slopes_list[0][p],cmap='RdBu_r',norm=norm,shading='auto',zorder=1)
    mappable=im
    l1,l2,a1,a2=lonlat_list[0]
    ax.set_xlim(l1,l2);ax.set_ylim(a1,a2)
    ax.set_xticks([]);ax.set_yticks([])
    ax.set_aspect('auto')
    ax.text(0.03,0.3,period_labels[p],transform=ax.transAxes,ha="left",va="center",fontsize=22)
    ax2=axes[p][1]
    latA2=lat_list[1];lonA2=lon_list[1]
    LON2,LAT2=np.meshgrid(lonA2,latA2)
    world.plot(ax=ax2,color="lightgrey",edgecolor="grey",linewidth=0.4,zorder=0)
    im2=ax2.pcolormesh(LON2,LAT2,slopes_list[1][p],cmap='RdBu_r',norm=norm,shading='auto',zorder=1)
    mappable=im2
    l1b,l2b,a1b,a2b=lonlat_list[1]
    ax2.set_xlim(l1b,l2b);ax2.set_ylim(a1b,a2b)
    ax2.set_xticks([]);ax2.set_yticks([])
    ax2.set_aspect('auto')
cax=axes[0][0].inset_axes([0.05,0.15,0.3,0.05])
cbar=plt.colorbar(mappable,cax=cax,orientation="horizontal",extend='both')
cbar.ax.tick_params(labelsize=18)
os.makedirs(os.path.dirname(out_png),exist_ok=True)
plt.savefig(out_png,dpi=300,bbox_inches='tight')
plt.close(fig)
