# -*- coding: utf-8 -*-
import os,glob,pandas as pd,numpy as np,xarray as xr,matplotlib.pyplot as plt
from scipy.stats import pearsonr
plt.rcParams['font.family']='Times New Roman'
plt.rcParams['font.size']=36  # 放大字体（其他所有元素）
plt.rcParams['axes.unicode_minus']=False

gpp_dt_csv=r"data/3Supplementary Fig/Supplementary Fig. 7/1.csv"
gpp_nt_csv=r"data/3Supplementary Fig/Supplementary Fig. 7/2.csv"
site_csv=r"data/3Supplementary Fig/Supplementary Fig. 7/3.csv"
nc_na=r"data/6Sample data/gpp/GPP4.nc"
nc_eu=r"data/6Sample data/gpp/GPP2.nc"
out_png=r"results/3Supplementary Fig/Supplementary Fig. 7.png"

# 读取数据
sites=pd.read_csv(site_csv)
gpp_dt=pd.read_csv(gpp_dt_csv)
gpp_nt=pd.read_csv(gpp_nt_csv)
site_ids=[sid for sid in sites["Site ID"] if sid in gpp_dt.columns and sid in gpp_nt.columns]

ds_na=xr.open_dataset(nc_na)
ds_eu=xr.open_dataset(nc_eu)
if "gpp" not in ds_na.data_vars or "gpp" not in ds_eu.data_vars:
    raise ValueError("两个NC需包含变量gpp")
gpp_na=ds_na["gpp"]
gpp_eu=ds_eu["gpp"]
per_site_best={}

# 计算相关系数
for _,r in sites.iterrows():
    sid=r["Site ID"]
    if sid not in site_ids:
        continue
    s_na=gpp_na.sel(latitude=r["Lat"],longitude=r["Long"],method="nearest").to_pandas()
    s_eu=gpp_eu.sel(latitude=r["Lat"],longitude=r["Long"],method="nearest").to_pandas()
    try:
        s_na.index=s_na.index.astype(int)
    except:
        s_na.index=pd.Index(pd.to_datetime(s_na.index).year.astype(int))
    try:
        s_eu.index=s_eu.index.astype(int)
    except:
        s_eu.index=pd.Index(pd.to_datetime(s_eu.index).year.astype(int))
    
    def eval_one(s,lab):
        df_dt=pd.DataFrame({"year":gpp_dt["year"],"GPP":gpp_dt[sid],"PROD":s.reindex(gpp_dt["year"]).values}).dropna()
        df_nt=pd.DataFrame({"year":gpp_nt["year"],"GPP":gpp_nt[sid],"PROD":s.reindex(gpp_nt["year"]).values}).dropna()
        if len(df_dt)>=7:
            r_dt,p_dt=pearsonr(df_dt["GPP"],df_dt["PROD"])
        else:
            r_dt,p_dt=np.nan,1.0
        if len(df_nt)>=7:
            r_nt,p_nt=pearsonr(df_nt["GPP"],df_nt["PROD"])
        else:
            r_nt,p_nt=np.nan,1.0
        m=np.nanmean([r_dt,r_nt])
        n_dt=len(df_dt);n_nt=len(df_nt)
        return r_dt,p_dt,r_nt,p_nt,m,n_dt,n_nt,lab

    r1=eval_one(s_na,os.path.basename(nc_na))
    r2=eval_one(s_eu,os.path.basename(nc_eu))
    c1=r1[5]+r1[6]
    c2=r2[5]+r2[6]
    pick=r1 if (c1>c2 or (c1==c2 and (np.nan_to_num(r1[4],nan=-9)>np.nan_to_num(r2[4],nan=-9)))) else r2
    if np.isfinite(pick[4]) and (pick[5]>=1 or pick[6]>=1):
        per_site_best[sid]={"r_dt":pick[0],"p_dt":pick[1],"r_nt":pick[2],"p_nt":pick[3],"metric":pick[4],"n_dt":pick[5],"n_nt":pick[6],"nc":pick[7]}

ds_na.close()
ds_eu.close()

# 判断是否有可用数据
if not per_site_best:
    print("无可用站点样本（对齐后年份<7）")
else:
    df=pd.DataFrame.from_dict(per_site_best,orient="index").reset_index().rename(columns={"index":"Site ID"})
    df=df.sort_values("metric",ascending=False)
    print("各站点最优产品及相关系数：")
    print(df[["Site ID","metric","r_dt","p_dt","r_nt","p_nt","n_dt","n_nt","nc"]].to_string(index=False,max_colwidth=120))

    # 创建子图
    x=np.arange(len(df))
    w=0.4
    fig,ax=plt.subplots(figsize=(25,12),dpi=300, nrows=2, ncols=1)

    # 绘制GPP_DT相关性条形图 (子图1)
    ax[0].bar(x-w/2,df["r_dt"].fillna(0),width=w,label="GPP_DT",color='royalblue',zorder=2)
    ax[0].axhline(0,color='k',linewidth=1,zorder=1)
    ax[0].set_ylabel("Correlation (r)", fontsize=30)  # 放大坐标轴标签
    ax[0].set_xticks([])  # 不显示子图1的横坐标标签
    ax[0].set_ylim(-1,1)
    med_dt=np.nanmedian(df["r_dt"].values)
    ax[0].text(0.99,0.98,f"Median r = {med_dt:.3f}",transform=ax[0].transAxes,ha='right',va='top',fontsize=30)
    for i,(y,p) in enumerate(zip(df["r_dt"].fillna(0),df["p_dt"].fillna(1.0))):
        if p<0.05:
            ax[0].text(i-w/2,y+0.01 if y>=0 else y-0.06,"*",ha="center",va="bottom" if y>=0 else "top", fontsize=30)

    # 绘制GPP_NT相关性条形图 (子图2)
    ax[1].bar(x-w/2,df["r_nt"].fillna(0),width=w,label="GPP_NT",color='darkorange',zorder=2)
    ax[1].axhline(0,color='k',linewidth=1,zorder=1)
    ax[1].set_ylabel("Correlation (r)", fontsize=30)  # 放大坐标轴标签
    ax[1].set_xlabel("Site ID", fontsize=30)  # 放大x轴标签
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(df["Site ID"],rotation=90, fontsize=20)  # 放大x轴的站点标签
    ax[1].set_ylim(-1,1)
    med_nt=np.nanmedian(df["r_nt"].values)
    ax[1].text(0.99,0.98,f"Median r = {med_nt:.3f}",transform=ax[1].transAxes,ha='right',va='top',fontsize=30)
    for i,(y,p) in enumerate(zip(df["r_nt"].fillna(0),df["p_nt"].fillna(1.0))):
        if p<0.05:
            ax[1].text(i-w/2,y+0.01 if y>=0 else y-0.06,"*",ha="center",va="bottom" if y>=0 else "top", fontsize=30)

    ax[0].legend(frameon=True,edgecolor='black',fancybox=False, fontsize=25)
    ax[1].legend(frameon=True,edgecolor='black',fancybox=False, fontsize=25)
    plt.tight_layout()
    plt.savefig(out_png,dpi=300)
    plt.close()
    print(f"站点数: {len(df)}")
    print(f"中位数(GPP_DT) r: {med_dt:.3f}")
    print(f"中位数(GPP_NT) r: {med_nt:.3f}")
    print(f"图已保存到: {out_png}")
