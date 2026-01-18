import xarray as xr
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MultipleLocator,FormatStrFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
mpl.rcParams["font.family"]="Times New Roman"
mpl.rcParams["savefig.dpi"]=300
mpl.rcParams["font.size"]=24
npp_path=r"data/6Sample data/1_ISBA-CTRIP.nc"
age_path=r"TRW/tree_ring4 - AgeDepSpline(EPS0.85).nc"
eu_shp_path=r"map/欧洲7类-class.shp"
na_shp_path=r"map/Koppen_1991_2020_NA_big7_simple.shp"
shp_world=r"map/World.shp"
out_dir=r"results/1Fig"
os.makedirs(out_dir,exist_ok=True)
MIN_N=40
eu_lon_min,eu_lon_max=-11.0,57.0
eu_lat_min,eu_lat_max=30.0,75.0
na_lon_min,na_lon_max=-167.27,-55.00
na_lat_min,na_lat_max=9.58,74.99
ds_npp=xr.open_dataset(npp_path)
ds_trw=xr.open_dataset(age_path)
npp=ds_npp["npp"]
trw=ds_trw["AgeDepSpline"]
year_npp=ds_npp["year"].values
year_trw=ds_trw["year"].values
time_dim_npp=[d for d in npp.dims if d not in ("latitude","longitude")][0]
time_dim_trw=[d for d in trw.dims if d not in ("latitude","longitude")][0]
npp=npp.assign_coords({time_dim_npp:year_npp}).rename({time_dim_npp:"year"})
trw=trw.assign_coords({time_dim_trw:year_trw}).rename({time_dim_trw:"year"})
years=np.intersect1d(npp["year"].values,trw["year"].values)
years=years[(years>=1950)&(years<=2020)]
npp=npp.sel(year=years)
trw=trw.sel(year=years)
if float(np.nanmin(npp["longitude"].values))>=0:
    npp=npp.assign_coords(longitude=(((npp["longitude"]+180)%360)-180)).sortby("longitude")
    trw=trw.assign_coords(longitude=(((trw["longitude"]+180)%360)-180)).sortby("longitude")
world=gpd.read_file(shp_world)
world=world.set_crs(epsg=4326) if world.crs is None else world.to_crs(epsg=4326)
ratio=(trw/npp).where(np.isfinite(trw/npp))
mask_ratio=ratio.notnull()
n_valid=mask_ratio.sum("year")
mu_ratio=ratio.mean("year",skipna=True)
std_ratio=ratio.std("year",skipna=True)
ratio_z=((ratio-mu_ratio)/std_ratio).where(std_ratio>0)
x=xr.DataArray(years.astype(float),coords={"year":years},dims=("year",))
x_mean=x.where(mask_ratio).mean("year",skipna=True)
z_mean=ratio_z.where(mask_ratio).mean("year",skipna=True)
xm=x-x_mean
zm=ratio_z-z_mean
num=(xm*zm).where(mask_ratio).sum("year",skipna=True)
den=(xm*xm).where(mask_ratio).sum("year",skipna=True)
slope=(num/den).where((n_valid>=MIN_N)&(den>0)&(std_ratio>0))
na_s=slope.sel(longitude=slice(na_lon_min,na_lon_max),latitude=slice(na_lat_min,na_lat_max))
eu_s=slope.sel(longitude=slice(eu_lon_min,eu_lon_max),latitude=slice(eu_lat_min,eu_lat_max))
allv=np.concatenate([na_s.values.ravel(),eu_s.values.ravel()])
allv=allv[np.isfinite(allv)]
vmax=float(np.nanpercentile(np.abs(allv),95)) if allv.size>0 else 0.04
vmax=max(vmax,1e-6)
vmin=-vmax
ratio_z_stacked=ratio_z.stack(points=("latitude","longitude"))
lon_points=ratio_z_stacked["longitude"].values
lat_points=ratio_z_stacked["latitude"].values
points_index=ratio_z_stacked["points"].values
gdf_points=gpd.GeoDataFrame({"points":points_index,"lon":lon_points,"lat":lat_points},geometry=gpd.points_from_xy(lon_points,lat_points),crs="EPSG:4326")
def extract_region_series_from_points(da_stacked,shp_path):
    shp=gpd.read_file(shp_path)
    shp=shp.set_crs("EPSG:4326") if shp.crs is None else shp.to_crs("EPSG:4326")
    join=gpd.sjoin(gdf_points,shp,how="inner",predicate="within")
    if join.empty:return None
    pt_ids=np.unique(join["points"].values)
    reg=da_stacked.sel(points=pt_ids)
    if np.all(np.isnan(reg.values)):return None
    q_da=reg.quantile(q=[0.25,0.5,0.75],dim="points",skipna=True)
    return q_da.sel(quantile=0.5).values,q_da.sel(quantile=0.25).values,q_da.sel(quantile=0.75).values
eu_med_z,eu_q25_z,eu_q75_z=extract_region_series_from_points(ratio_z_stacked,eu_shp_path)
na_med_z,na_q25_z,na_q75_z=extract_region_series_from_points(ratio_z_stacked,na_shp_path)
if eu_med_z is None or na_med_z is None:raise ValueError("EU 或 NA 区域没有有效的 grid-z TRW/NPP 统计结果")
lon=npp["longitude"]
lat=npp["latitude"]
lat2d,lon2d=xr.broadcast(lat,lon)
eu_mask=(lon2d>=-11.0)&(lon2d<=57.0)&(lat2d>=30.0)&(lat2d<=75.0)
na_mask=(lon2d>=-167.27)&(lon2d<=-55.0)&(lat2d>=9.58)&(lat2d<=74.99)
roi=eu_mask|na_mask
npp_roi=npp.where(roi)
trw_roi=trw.where(roi)
pair=np.isfinite(npp_roi)&np.isfinite(trw_roi)
npp_roi=npp_roi.where(pair)
trw_roi=trw_roi.where(pair)
npp_mu=npp_roi.mean("year",skipna=True)
npp_std=npp_roi.std("year",skipna=True)
trw_mu=trw_roi.mean("year",skipna=True)
trw_std=trw_roi.std("year",skipna=True)
npp_z=((npp_roi-npp_mu)/npp_std).where(npp_std>0)
trw_z=((trw_roi-trw_mu)/trw_std).where(trw_std>0)
npp_mean=npp_z.mean(("latitude","longitude"),skipna=True).values
trw_mean=trw_z.mean(("latitude","longitude"),skipna=True).values
npp_p25=npp_z.quantile(0.25,dim=("latitude","longitude"),skipna=True).values
npp_p75=npp_z.quantile(0.75,dim=("latitude","longitude"),skipna=True).values
trw_p25=trw_z.quantile(0.25,dim=("latitude","longitude"),skipna=True).values
trw_p75=trw_z.quantile(0.75,dim=("latitude","longitude"),skipna=True).values
na_lon_span=na_lon_max-na_lon_min
na_lat_span=na_lat_max-na_lat_min
eu_lon_span=eu_lon_max-eu_lon_min
eu_lat_span=eu_lat_max-eu_lat_min
wr1=max(na_lon_span/na_lat_span,1e-6)
wr2=max(eu_lon_span/eu_lat_span,1e-6)
fig=plt.figure(figsize=(18,14),dpi=300)
gs=gridspec.GridSpec(2,2,figure=fig,left=0.035,right=0.99,bottom=0.055,top=0.99,wspace=0.10,hspace=0.14,width_ratios=[wr1,wr2],height_ratios=[1.35,1.0])
ax_a=fig.add_subplot(gs[0,0])
ax_b=fig.add_subplot(gs[0,1])
ax_c=fig.add_subplot(gs[1,0])
ax_d=fig.add_subplot(gs[1,1])
im=ax_a.pcolormesh(na_s["longitude"].values,na_s["latitude"].values,na_s.values,cmap="RdBu_r",vmin=vmin,vmax=vmax,shading="nearest",zorder=1,alpha=0.95)
world.cx[na_lon_min:na_lon_max,na_lat_min:na_lat_max].boundary.plot(ax=ax_a,color="black",linewidth=0.6,zorder=3)
ax_a.set_xlim(na_lon_min,na_lon_max);ax_a.set_ylim(na_lat_min,na_lat_max)
ax_a.set_aspect(1.3,adjustable="box")
ax_a.set_xlabel("");ax_a.set_ylabel("")
ax_a.tick_params(axis="both",which="both",labelsize=22,bottom=True,left=True)
cax=inset_axes(ax_a,width="36%",height="4.2%",loc="lower left",borderpad=2.4)
cbar=fig.colorbar(im,cax=cax,orientation="horizontal",extend="both")
cbar.set_label("Standardized TRW/NPP",fontsize=20)
cbar.ax.tick_params(labelsize=18)
ax_b.pcolormesh(eu_s["longitude"].values,eu_s["latitude"].values,eu_s.values,cmap="RdBu_r",vmin=vmin,vmax=vmax,shading="nearest",zorder=1,alpha=0.95)
world.cx[eu_lon_min:eu_lon_max,eu_lat_min:eu_lat_max].boundary.plot(ax=ax_b,color="black",linewidth=0.6,zorder=3)
ax_b.set_xlim(eu_lon_min,eu_lon_max);ax_b.set_ylim(eu_lat_min,eu_lat_max)
ax_b.set_aspect(1.3,adjustable="box")
ax_b.yaxis.tick_right()
ax_b.set_xlabel("");ax_b.set_ylabel("")
ax_b.tick_params(axis="both",which="both",labelsize=22,bottom=True,right=True,left=False)
ax_b.yaxis.set_major_locator(MultipleLocator(10))
ax_b.yaxis.set_major_formatter(FormatStrFormatter("%d"))
ax_c.plot(years,eu_med_z,linewidth=2.2,label="Europe")
ax_c.fill_between(years,eu_q25_z,eu_q75_z,alpha=0.25)
ax_c.plot(years,na_med_z,linewidth=2.2,label="North America")
ax_c.fill_between(years,na_q25_z,na_q75_z,alpha=0.25)
ax_c.set_xlim(1950,2020)
ax_c.set_xlabel("Year",fontsize=22)
ax_c.set_ylabel("Standardized TRW/NPP",fontsize=22)
ax_c.tick_params(labelsize=22)
ax_c.xaxis.set_major_locator(MultipleLocator(10))
ax_c.yaxis.set_major_locator(MultipleLocator(1))
ax_c.yaxis.set_major_formatter(FormatStrFormatter("%d"))
ax_c.legend(frameon=True,fancybox=False,edgecolor="black",fontsize=18,loc="upper right")

ax_d.fill_between(years,npp_p25,npp_p75,alpha=0.22)
ax_d.fill_between(years,trw_p25,trw_p75,alpha=0.22)
ax_d.plot(years,npp_mean,linewidth=2.2,label="NPP")
ax_d.plot(years,trw_mean,linewidth=2.2,label="TRW")
ax_d.set_xlim(1950,2020)
ax_d.set_xlabel("Year",fontsize=22)
ax_d.set_ylabel("Standardized value",fontsize=22)
ax_d.tick_params(labelsize=22)
ax_d.xaxis.set_major_locator(MultipleLocator(10))
ax_d.yaxis.set_major_locator(MultipleLocator(1))
ax_d.yaxis.set_major_formatter(FormatStrFormatter("%d"))

ax_d.legend(frameon=True,fancybox=False,edgecolor="black",
            fontsize=18,loc="upper right",
            bbox_to_anchor=(0.8, 0.99))
ax_a.text(0.02,0.98,"a",transform=ax_a.transAxes,ha="left",va="top",fontweight="bold",fontsize=32)
ax_b.text(0.02,0.98,"b",transform=ax_b.transAxes,ha="left",va="top",fontweight="bold",fontsize=32)
ax_c.text(0.02,0.98,"c",transform=ax_c.transAxes,ha="left",va="top",fontweight="bold",fontsize=32)
ax_d.text(0.02,0.98,"d",transform=ax_d.transAxes,ha="left",va="top",fontweight="bold",fontsize=32)
ax_d.text(0.78,0.3,"r = 0.75/np < 0.001",
          transform=ax_d.transAxes,
          ha="left",va="bottom",fontsize=24)

ax_c.text(0.28,0.3,"r = 0.8/np < 0.001",
          transform=ax_c.transAxes,
          ha="left",va="bottom",fontsize=24)



out_png=os.path.join(out_dir,"Fig. 2.png")
plt.savefig(out_png,bbox_inches="tight",pad_inches=0.02,facecolor="white")
plt.close()
print("✅ 已保存：",out_png)
print("MIN_N =",MIN_N,"vmax(p95 |slope|) =",vmax)
def calc_inc_dec_ratio(da):
    v=da.values
    v=v[np.isfinite(v)]
    if v.size==0:
        return np.nan,np.nan
    inc=np.sum(v>0)
    dec=np.sum(v<0)
    total=inc+dec
    return inc/total,dec/total

na_inc,na_dec=calc_inc_dec_ratio(na_s)
eu_inc,eu_dec=calc_inc_dec_ratio(eu_s)

print("North America:")
print("  Increasing ratio =",round(na_inc*100,2),"%")
print("  Decreasing ratio =",round(na_dec*100,2),"%")

print("Europe:")
print("  Increasing ratio =",round(eu_inc*100,2),"%")
print("  Decreasing ratio =",round(eu_dec*100,2),"%")
