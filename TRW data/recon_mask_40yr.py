# -*- coding: utf-8 -*-
# 逐年“可信范围”掩膜：仅陆地、半径 3°、当年可用点≥3（0 视为缺失）
import os
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
from shapely.geometry import Point, box
from shapely.prepared import prep
from scipy.spatial import cKDTree

# ===== 路径与参数 =====
points_csv     = r"D:\all\data\data\基本属性.csv"     # 需含列：longitude, latitude, name
agespline_csv  = r"D:\all\data\data\AgeDepSpline.csv" # 首列为年份，其余列为各 name
world_shp_path = r"D:\all\data\世界底图\World_countries.shp"
out_nc         = r"D:\all\data\recon_mask_40yr.nc"

grid_res, anchor_lon, anchor_lat, bbox_margin = 0.5, -179.75, -89.75, 5.0
radius_deg, min_points = 5, 3
full_years  = np.arange(1700, 2021, dtype=np.int32)   # 历年范围（1700~2020）

# ===== 读取点与年表（0 视为缺失） =====
pts = pd.read_csv(points_csv)[['longitude','latitude','name']].dropna().reset_index(drop=True)
if pts.empty:
    raise ValueError("点位数据为空。")
points_xy = pts[['longitude','latitude']].to_numpy()
names     = pts['name'].astype(str).to_numpy()

ts = pd.read_csv(agespline_csv)
ts = ts.rename(columns={ts.columns[0]:'year'}).set_index('year').reindex(full_years)
used = [c for c in ts.columns if c in set(names)]
if not used:
    raise ValueError("年表中未找到匹配的 name 列。")

# 0 视为缺失；fin 标记逐年是否有效
mat = ts[used].to_numpy(dtype=np.float64)
mat[mat == 0] = np.nan
fin = np.isfinite(mat)  # (Y × M)，Y=年数，M=匹配到的年表列数

# 将点位顺序映射到年表列序
idx_by_name = {nm: j for j, nm in enumerate(used)}
col_idx = np.array([idx_by_name.get(nm, -1) for nm in names], dtype=int)  # 长度 = 点数
has_col = col_idx >= 0

# ===== 构建 0.5° 对齐网格（以 anchor 对齐） =====
def axis_aligned(vmin, vmax, anchor, step):
    a = int(np.ceil((vmin - anchor) / step))
    b = int(np.floor((vmax - anchor) / step))
    return (anchor + np.arange(a, b + 1) * step).astype(np.float32)

lon = axis_aligned(points_xy[:,0].min() - bbox_margin,
                   points_xy[:,0].max() + bbox_margin,
                   anchor_lon, grid_res)
lat = axis_aligned(points_xy[:,1].min() - bbox_margin,
                   points_xy[:,1].max() + bbox_margin,
                   anchor_lat, grid_res)
LON, LAT = np.meshgrid(lon, lat)
grid = np.c_[LON.ravel(), LAT.ravel()]
G = grid.shape[0]

# ===== 陆地掩膜（bbox 限定以加速 dissolve） =====
world = gpd.read_file(world_shp_path).to_crs(4326)
bbox = box(lon.min(), lat.min(), lon.max(), lat.max())
try:
    ids = list(world.sindex.query(bbox, predicate="intersects"))
    sub = world.iloc[ids]
except Exception:
    sub = world

# 兼容 geopandas 旧/新版本：优先用 union_all()
try:
    land = sub.union_all()
except AttributeError:
    # 旧版本回退：unary_union（会给出弃用提醒）
    land = sub.unary_union

prep_land = prep(land)
land_mask = np.fromiter((prep_land.intersects(Point(xy)) for xy in grid), count=G, dtype=bool)

# ===== KDTree 半径近邻（一次性） =====
tree = cKDTree(points_xy)
k = 128  # 足够大的上限邻居数
dists, idxs = tree.query(grid, k=k, distance_upper_bound=radius_deg)
valid = np.isfinite(dists) & (idxs < len(pts))  # (G × k)
safe_idxs = np.where(valid, idxs, 0)           # 用 0 填充越界索引

# ===== 计算 recon_mask(year, lat, lon)：逐年统计“当年可用点”数量 =====
Y = full_years.size
mask3d = np.full((Y, lat.size, lon.size), np.nan, dtype=np.float32)

# 预构造：对每个点位，逐年是否有效（按年表列顺序）
M = len(used)
valid_by_point_year = np.zeros((Y, len(pts)), dtype=bool)
if np.any(has_col):
    valid_by_point_year[:, has_col] = fin[:, col_idx[has_col]]

# 主循环：逐年在每个格点统计半径内“当年可用”的点位数
for yi in range(Y):
    # 每个格点近邻点位在该年的有效性，形状 (G × k)
    nn_year = valid_by_point_year[yi, :][safe_idxs]
    nn_year[~valid] = False  # 掩掉半径外/无邻居
    ok = (np.sum(nn_year, axis=1) >= min_points) & land_mask
    mask3d[yi] = np.where(ok.reshape(lat.size, lon.size), 1.0, np.nan).astype(np.float32)

# ===== 写 NC（仅 1 个变量） =====
ds = xr.Dataset(
    data_vars={"recon_mask": (["year","latitude","longitude"], mask3d)},
    coords={"year": full_years, "latitude": lat, "longitude": lon},
    attrs={
        "description": "Per-year reconstructable mask (land-only, >=3 usable points within 3 deg). 1=valid, NaN=invalid.",
        "resolution":  "0.5 degree (centers anchored at -179.75 / -89.75)",
        "note":        "逐年判定，无滑动窗口；当年点位值为 NaN 或 0 视为不可用"
    }
)
ds["recon_mask"].encoding.update({"zlib": True, "complevel": 4, "_FillValue": np.nan})
os.makedirs(os.path.dirname(out_nc), exist_ok=True)
ds.to_netcdf(out_nc)

print("✅ 保存：", out_nc, "| 维度：", dict(ds.sizes))
