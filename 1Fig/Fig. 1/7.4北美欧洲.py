import pandas as pd,geopandas as gpd,matplotlib.pyplot as plt,matplotlib as mpl
from shapely.geometry import Point,shape as shp_shape
import os,numpy as np,xarray as xr
from rasterio.transform import from_bounds
from rasterio.features import shapes
from matplotlib.patches import Patch
mpl.rcParams['font.family']='Times New Roman'
mpl.rcParams['axes.unicode_minus']=False
mpl.rcParams['font.size']=16
na_lon_min,na_lon_max=-167.27,-55.00
na_lat_min,na_lat_max=9.58,74.99
eu_lon_min,eu_lon_max=-11.0,55
eu_lat_min,eu_lat_max=29.0,75.0
csv_na=r"D:\all\1Fig\Fig. 1\data\基本属性.csv"
excel_na=r"D:\all\1Fig\Fig. 1\data\通量站基本信息.xlsx"
world_na=r"D:\all\data\世界底图\World.shp"
koppen_na=r"D:\all\data\世界底图\Koppen_1991_2020_NA_big7.shp"
nc_na=r"D:\all\data\tree_ring4 - AgeDepSpline(EPS0.85).nc"
csv_eu=r"D:\all\1Fig\Fig. 1\data\欧洲基础属性表3.csv"
stations_eu=r"D:\all\1Fig\Fig. 1\data\stations.csv"
world_eu=r"D:\all\data\世界底图\World.shp"
koppen_eu=r"D:\all\data\世界底图\欧洲7类.shp"
nc_eu=r"D:\all\data\tree_ring4 - AgeDepSpline(EPS0.85).nc"
extra_site_csv=r"D:\all\1Fig\Fig. 1\data\坐标.csv"
npp_site_csv=r"D:\all\1Fig\Fig. 1\data\3NPP2.csv"
out_png=r"D:\all\1Fig\Fig. 1\Fig. 1.png"
size_flux=80
size_npp=80
size_tree=100
lw_tree=1.2
aspect_y_over_x=1.3
gap_x=-0.64
color_map={"A":"#ff6e40","B":"#ffd166","C":"#9fbfad","D other":"#2f7ec2","Dfa":"#cfe8ff","Dfb":"#7ab6f2","E":"#0f4c81"}
def cls_simple(s):
    s=str(s)
    if s.startswith("D 其他"):return "D other"
    return s.split()[0] if " " in s else s
def valid_gdf_from_nc(nc_path):
    ds=xr.open_dataset(nc_path)
    da=ds["AgeDepSpline"].sel(year=slice(1981,2000))
    m=np.isfinite(da).any(dim="year").values.astype("uint8")
    lat=ds["latitude"].values if "latitude" in ds.coords else ds["lat"].values
    lon=ds["longitude"].values if "longitude" in ds.coords else ds["lon"].values
    arr=m[::-1,:] if bool(lat[-1]>lat[0]) else m
    T=from_bounds(float(lon.min()),float(lat.min()),float(lon.max()),float(lat.max()),len(lon),len(lat))
    polys=[shp_shape(g) for g,v in shapes(arr,mask=arr==1,transform=T) if v==1]
    return gpd.GeoDataFrame(geometry=[gpd.GeoSeries(polys).union_all().buffer(0)],crs="EPSG:4326")
def prep_region(lon_min,lon_max,lat_min,lat_max,csv_path,flux_df,world_shp,koppen_shp,nc_path):
    world=gpd.read_file(world_shp).to_crs("EPSG:4326")
    world_bbox=world.cx[lon_min:lon_max,lat_min:lat_max]
    valid_gdf=valid_gdf_from_nc(nc_path)
    dfb=pd.read_csv(csv_path)
    gdf_base=gpd.GeoDataFrame(dfb,geometry=[Point(x,y) for x,y in zip(dfb["longitude"],dfb["latitude"])],crs="EPSG:4326")
    gdf_base_clip=gpd.clip(gdf_base,valid_gdf)
    flux_clip=gpd.clip(flux_df,valid_gdf)
    kop=gpd.read_file(koppen_shp).to_crs("EPSG:4326")
    kop=kop.cx[lon_min:lon_max,lat_min:lat_max].clip(valid_gdf)
    kop["class_simple"]=kop["class"].apply(cls_simple)
    kop7=kop.dissolve(by="class_simple",as_index=False,aggfunc="first")
    kop7["facecolor"]=kop7["class_simple"].map(color_map).fillna("#dddddd")
    bb=np.array([lon_min,lat_min,lon_max,lat_max])
    return world_bbox,kop7,gdf_base_clip,flux_clip,bb
na_flux_raw=pd.read_excel(excel_na,usecols=["Site ID","Lat","Long"]).dropna(subset=["Lat","Long"])
na_flux=gpd.GeoDataFrame(na_flux_raw,geometry=[Point(x,y) for x,y in zip(na_flux_raw["Long"],na_flux_raw["Lat"])],crs="EPSG:4326")
eu_flux_raw=pd.read_csv(stations_eu,usecols=["latitude","longitude"]).dropna(subset=["latitude","longitude"])
eu_flux_raw=eu_flux_raw[(eu_flux_raw["longitude"]>=eu_lon_min)&(eu_flux_raw["longitude"]<=eu_lon_max)&(eu_flux_raw["latitude"]>=eu_lat_min)&(eu_flux_raw["latitude"]<=eu_lat_max)]
eu_flux=gpd.GeoDataFrame(eu_flux_raw,geometry=[Point(x,y) for x,y in zip(eu_flux_raw["longitude"],eu_flux_raw["latitude"])],crs="EPSG:4326")
world_na_bbox,kop_na,base_na,flux_na,bbox_na=prep_region(na_lon_min,na_lon_max,na_lat_min,na_lat_max,csv_na,na_flux,world_na,koppen_na,nc_na)
world_eu_bbox,kop_eu,base_eu,flux_eu,bbox_eu=prep_region(eu_lon_min,eu_lon_max,eu_lat_min,eu_lat_max,csv_eu,eu_flux,world_eu,koppen_eu,nc_eu)
df_extra=pd.read_csv(extra_site_csv,usecols=["site_ID","latitude","longitude"])
df_npp=pd.read_csv(npp_site_csv,usecols=["site_ID","NPP_tot1"])
df_extra["site_ID"]=df_extra["site_ID"].astype(str)
df_npp["site_ID"]=df_npp["site_ID"].astype(str)
df_npp["NPP_tot1"]=pd.to_numeric(df_npp["NPP_tot1"],errors="coerce")
dfm=df_extra.merge(df_npp,on="site_ID",how="inner")
dfm=dfm[dfm["NPP_tot1"].notna()]
gdf_extra=gpd.GeoDataFrame(dfm,geometry=[Point(x,y) for x,y in zip(dfm["longitude"],dfm["latitude"])],crs="EPSG:4326")
extra_na=gpd.clip(gdf_extra,valid_gdf_from_nc(nc_na))
extra_eu=gpd.clip(gdf_extra,valid_gdf_from_nc(nc_eu))
na_lat_span=bbox_na[3]-bbox_na[1]
na_lon_span=bbox_na[2]-bbox_na[0]
eu_lat_span=bbox_eu[3]-bbox_eu[1]
eu_lon_span=bbox_eu[2]-bbox_eu[0]
fig=plt.figure(figsize=(18,12),dpi=300)
main_left,main_bottom=0.06,0.08
main_h=0.86
main_w=main_h*(na_lon_span/na_lat_span)
ax_na=fig.add_axes([main_left,main_bottom,main_w,main_h])
world_na_bbox.plot(ax=ax_na,facecolor="#f0f0f0",edgecolor="gray",linewidth=0.5,zorder=1)
kop_na.plot(ax=ax_na,color=kop_na["facecolor"],edgecolor="none",alpha=0.6,zorder=2.3)
ax_na.scatter(flux_na.geometry.x.values,flux_na.geometry.y.values,s=size_flux,c="green",edgecolors="white",linewidths=0.4,alpha=0.9,zorder=3.4)
ax_na.scatter(extra_na.geometry.x.values,extra_na.geometry.y.values,s=size_npp,c="red",edgecolors="white",linewidths=0.4,alpha=0.9,zorder=3.5)
ax_na.scatter([p.x for p in base_na.geometry],[p.y for p in base_na.geometry],s=size_tree,c="black",marker="+",linewidths=lw_tree,alpha=0.95,zorder=3.8)
ax_na.set_xlim(bbox_na[0],bbox_na[2]); ax_na.set_ylim(bbox_na[1],bbox_na[3])
ax_na.set_aspect(aspect_y_over_x,adjustable='box')
ax_na.tick_params(axis='both',which='major',labelsize=30)
eu_h=main_h
eu_w=eu_h*(eu_lon_span/eu_lat_span)
inset_left=main_left+main_w+gap_x
inset_bottom=main_bottom
ax_eu=fig.add_axes([inset_left,inset_bottom,eu_w,eu_h])
world_eu_bbox.plot(ax=ax_eu,facecolor="#f0f0f0",edgecolor="gray",linewidth=0.5,zorder=1)
kop_eu.plot(ax=ax_eu,color=kop_eu["facecolor"],edgecolor="none",alpha=0.6,zorder=2.3)
ax_eu.scatter(flux_eu.geometry.x.values,flux_eu.geometry.y.values,s=size_flux,c="green",edgecolors="white",linewidths=0.4,alpha=0.9,zorder=3.4)
ax_eu.scatter(extra_eu.geometry.x.values,extra_eu.geometry.y.values,s=size_npp,c="red",edgecolors="white",linewidths=0.4,alpha=0.9,zorder=3.5)
ax_eu.scatter([p.x for p in base_eu.geometry],[p.y for p in base_eu.geometry],s=size_tree,c="black",marker="+",linewidths=lw_tree,alpha=0.95,zorder=3.8)
ax_eu.set_xlim(bbox_eu[0],bbox_eu[2])
ax_eu.set_ylim(bbox_eu[1],bbox_eu[3])
ax_eu.set_aspect(aspect_y_over_x,adjustable='box')
ax_eu.tick_params(axis='x',which='major',labelsize=30,bottom=True,top=False)
ax_eu.yaxis.tick_right()
ax_eu.yaxis.set_label_position("right")
ax_eu.tick_params(axis='y',which='major',labelsize=30,right=True,left=False)
pt_flux=ax_na.scatter([],[],s=size_flux,color="green",edgecolor="white",linewidth=0.4,alpha=0.9,label="Flux tower")
pt_npp=ax_na.scatter([],[],s=size_npp,color="red",edgecolor="white",linewidth=0.4,alpha=0.9,label="NPP observation")
pt_tree=ax_na.scatter([],[],s=size_tree,color="black",marker="+",linewidth=lw_tree,alpha=0.95,label="Tree-ring")
legend_name_map={
    "A":"Tropical",
    "B":"Arid",
    "C":"Temperate",
    "Dfa":"Dfa",
    "Dfb":"Dfb",
    "D other":"D other",
    "E":"Polar"
}
classes=["A","B","C","Dfa","Dfb","D other","E"]
present=set(pd.concat([kop_na["class_simple"],kop_eu["class_simple"]]))
faces=[Patch(facecolor=color_map[k],edgecolor="none",alpha=0.6,label=legend_name_map[k]) for k in classes if k in present]



ax_na.legend(handles=[pt_flux,pt_npp,pt_tree]+faces,loc="lower left",frameon=True,prop={"size":28},scatterpoints=1,borderpad=0.8,labelspacing=0.6,handletextpad=0.8,handlelength=2.0,markerscale=1.4,ncol=1)
os.makedirs(os.path.dirname(out_png),exist_ok=True)
plt.savefig(out_png,dpi=300,bbox_inches="tight")
plt.close()
print("✅ 已保存：",out_png)
print("NPP site 数量：", len(extra_na) + len(extra_eu))
print("Flux tower 数量：", len(flux_na) + len(flux_eu))
out_csv = r"D:\5-3\AgeDepSpline\article1\2\7.2.csv"

# 合并 NPP site
npp_sites = pd.concat([
    pd.DataFrame({"type": "NPP site",
                  "longitude": extra_na.geometry.x,
                  "latitude": extra_na.geometry.y}),
    pd.DataFrame({"type": "NPP site",
                  "longitude": extra_eu.geometry.x,
                  "latitude": extra_eu.geometry.y})
], ignore_index=True)

# 合并 Flux tower
flux_sites = pd.concat([
    pd.DataFrame({"type": "Flux tower",
                  "longitude": flux_na.geometry.x,
                  "latitude": flux_na.geometry.y}),
    pd.DataFrame({"type": "Flux tower",
                  "longitude": flux_eu.geometry.x,
                  "latitude": flux_eu.geometry.y})
], ignore_index=True)

# 合并并导出
df_out = pd.concat([npp_sites, flux_sites], ignore_index=True)
df_out.to_csv(out_csv, index=False, encoding="utf-8-sig")

print("✅ 已导出：", out_csv)
