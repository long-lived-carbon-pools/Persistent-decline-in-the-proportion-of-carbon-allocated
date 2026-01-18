# -*- coding: utf-8 -*-
import os,pandas as pd,numpy as np,matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from matplotlib.cm import get_cmap
from matplotlib.ticker import FuncFormatter
warm_gpp_csv=r"D:\all\2Extended Data Fig\Extended Data Fig. 1\data\GPP求和_欧北合并1-12_暖区shp.csv"
warm_npp_csv=r"D:\all\2Extended Data Fig\Extended Data Fig. 1\data\NPP求和_欧北合并1-12_暖区shp.csv"
cold_gpp_csv=r"D:\all\2Extended Data Fig\Extended Data Fig. 1\data\GPP求和_欧北合并1-12_冷区shp.csv"
cold_npp_csv=r"D:\all\2Extended Data Fig\Extended Data Fig. 1\data\NPP求和_欧北合并1-12_冷区shp.csv"
out_png=r"D:\all\2Extended Data Fig\Extended Data Fig. 1\Extended Data Fig. 1.png"
os.makedirs(os.path.dirname(out_png),exist_ok=True)
plt.rcParams.update({"font.family":"Times New Roman","mathtext.fontset":"custom","mathtext.rm":"Times New Roman","mathtext.it":"Times New Roman:italic","mathtext.bf":"Times New Roman:bold","font.size":20,"xtick.labelsize":18,"ytick.labelsize":18,"savefig.dpi":300,"figure.dpi":300})
def load_df(path):
    df=pd.read_csv(path)
    df["Year"]=df["Year"].astype(int)
    df=df.set_index("Year").sort_index().apply(pd.to_numeric,errors="coerce")
    return df
def fit_pca_series(g):
    g2=g.dropna(how="any")
    X=StandardScaler().fit_transform(g2.values)
    p=PCA(n_components=1).fit(X)
    pc1=p.transform(X).ravel()
    evr=float(p.explained_variance_ratio_[0])
    ref=g2.mean(axis=1).values
    a,b=np.polyfit(pc1,ref,1)
    return pd.Series(a*pc1+b,index=g2.index),evr
def set_3_sci_int_ticks(ax,exp=6):
    y0,y1=ax.get_ylim()
    if y1<y0:y0,y1=y1,y0
    scale=10.0**exp
    a=y0/scale
    b=y1/scale
    mid=int(np.round((a+b)/2.0))
    step=max(1,int(np.ceil((b-a)/2.0)))
    ticks_int=np.array([mid-step,mid,mid+step],dtype=int)
    ticks=ticks_int*scale
    ax.set_yticks(ticks)
    ax.set_ylim(ticks[0],ticks[-1])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x,pos:f"{int(np.round(x/scale))}"))
    ax.yaxis.get_offset_text().set_visible(False)
    ax.text(0.0,0.96,rf"$\times10^{{{exp}}}$",transform=ax.transAxes,ha="left",va="bottom")
def set_3_float_ticks(ax,nd=3):
    y0,y1=ax.get_ylim()
    ticks=np.linspace(y0,y1,3)
    ax.set_yticks(ticks)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x,pos:f"{x:.{nd}f}"))
def mark_peak_year(ax,series):
    peak_year=int(series.idxmax())
    ax.axvline(peak_year,ls="--",color="black",lw=1.0,alpha=0.9)
    ax.text(peak_year,0.96,f"{peak_year}",transform=ax.get_xaxis_transform(),ha="center",va="bottom",fontsize=16)
    return peak_year
def prep_block(gpp_csv,npp_csv):
    df_gpp=load_df(gpp_csv)
    df_npp=load_df(npp_csv)
    cols_gpp=[c for c in df_gpp.columns if str(c).startswith("1")]
    cols_npp=[c for c in df_npp.columns if str(c).startswith("1")]
    g_gpp=df_gpp[cols_gpp]
    g_npp=df_npp[cols_npp]
    ratio=g_npp/g_gpp
    pc1_gpp,evr_gpp=fit_pca_series(g_gpp)
    pc1_npp,evr_npp=fit_pca_series(g_npp)
    pc1_ratio,evr_ratio=fit_pca_series(ratio)
    xmin=int(min(g_gpp.index.min(),g_npp.index.min()))
    xmax=int(max(g_gpp.index.max(),g_npp.index.max()))
    return {"g_gpp":g_gpp,"g_npp":g_npp,"ratio":ratio,"pc1_gpp":pc1_gpp,"pc1_npp":pc1_npp,"pc1_ratio":pc1_ratio,"evr_gpp":evr_gpp,"evr_npp":evr_npp,"evr_ratio":evr_ratio,"xmin":xmin,"xmax":xmax}
warm=prep_block(warm_gpp_csv,warm_npp_csv)
cold=prep_block(cold_gpp_csv,cold_npp_csv)
xmin=min(warm["xmin"],cold["xmin"])
xmax=max(warm["xmax"],cold["xmax"])
cmap_gpp=get_cmap("Blues")
cmap_npp=get_cmap("Oranges")
cmap_ratio=get_cmap("Greens")
fig,axes=plt.subplots(3,2,figsize=(14,11),sharex=True)
w=3
def draw_col(ax_gpp,ax_npp,ax_ratio,blk):
    cols_gpp=list(blk["g_gpp"].columns)
    cols_npp=list(blk["g_npp"].columns)
    for i,c in enumerate(cols_gpp):
        s=blk["g_gpp"][c]
        y=s.rolling(w,1,center=True).mean()
        m=s.notna()
        ax_gpp.plot(s.index[m],y[m],lw=1.2,alpha=0.6,color=cmap_gpp(0.4+i*0.4/max(1,len(cols_gpp))))
    ax_gpp.plot(blk["pc1_gpp"].index,blk["pc1_gpp"],lw=3.2,color=cmap_gpp(0.95))
    for i,c in enumerate(cols_npp):
        s=blk["g_npp"][c]
        y=s.rolling(w,1,center=True).mean()
        m=s.notna()
        ax_npp.plot(s.index[m],y[m],lw=1.2,alpha=0.6,color=cmap_npp(0.4+i*0.4/max(1,len(cols_npp))))
    ax_npp.plot(blk["pc1_npp"].index,blk["pc1_npp"],lw=3.2,color=cmap_npp(0.95))
    ax_ratio.plot(blk["pc1_ratio"].index,blk["pc1_ratio"],lw=3.2,color=cmap_ratio(0.95))
    z=np.polyfit(blk["pc1_ratio"].index,blk["pc1_ratio"],1)
    ax_ratio.plot(blk["pc1_ratio"].index,np.poly1d(z)(blk["pc1_ratio"].index),"--",color="black")
draw_col(axes[0,0],axes[1,0],axes[2,0],warm)
draw_col(axes[0,1],axes[1,1],axes[2,1],cold)
for r in range(3):
    for c in range(2):
        ax=axes[r,c]
        ax.set_xlim(xmin,xmax)
        ax.grid(True,ls="--",lw=0.6,alpha=0.2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
set_3_sci_int_ticks(axes[0,0],6)
set_3_sci_int_ticks(axes[0,1],6)
set_3_sci_int_ticks(axes[1,0],6)
set_3_sci_int_ticks(axes[1,1],6)
set_3_float_ticks(axes[2,0],3)
set_3_float_ticks(axes[2,1],3)
mark_peak_year(axes[0,0],warm["pc1_gpp"])
mark_peak_year(axes[1,0],warm["pc1_npp"])
mark_peak_year(axes[0,1],cold["pc1_gpp"])
mark_peak_year(axes[1,1],cold["pc1_npp"])
axes[0,0].set_ylabel(r"GPP ($\mathrm{kg\ m^{-2}\ s^{-1}}$)",fontsize=22)
axes[1,0].set_ylabel(r"NPP ($\mathrm{kg\ m^{-2}\ s^{-1}}$)",fontsize=22)
axes[2,0].set_ylabel(r"NPP/GPP",fontsize=22)
for ax in (axes[0,0],axes[1,0],axes[2,0]):
    ax.yaxis.set_label_coords(-0.12,0.5)
for ax in (axes[0,0],axes[0,1],axes[1,0],axes[1,1]):
    p=ax.get_position()
    ax.set_position([p.x0,p.y0-0.02,p.width,p.height])
axes[0,0].text(0.2,0.92,f"PC1: {warm['evr_gpp']*100:.1f}%",transform=axes[0,0].transAxes,ha="right",va="top",fontsize=16)
axes[1,0].text(0.2,0.92,f"PC1: {warm['evr_npp']*100:.1f}%",transform=axes[1,0].transAxes,ha="right",va="top",fontsize=16)
axes[0,1].text(0.20,0.92,f"PC1: {cold['evr_gpp']*100:.1f}%",transform=axes[0,1].transAxes,ha="right",va="top",fontsize=16)
axes[1,1].text(0.20,0.92,f"PC1: {cold['evr_npp']*100:.1f}%",transform=axes[1,1].transAxes,ha="right",va="top",fontsize=16)

axes[2,0].set_xlabel("Year",fontsize=20)
axes[2,1].set_xlabel("Year",fontsize=20)

axes[0,0].set_title("Warm",fontsize=22,pad=18)
axes[0,1].set_title("Cold",fontsize=22,pad=18)



panel=["a","b","c","d","e","f"]
k=0
for r in range(3):
    for c in range(2):
        axes[r,c].text(0.5,0.97,panel[k],transform=axes[r,c].transAxes,fontweight="bold",fontsize=24,ha="left",va="top")
        k+=1
plt.subplots_adjust(left=0.10,right=0.98,bottom=0.08,top=0.94,wspace=0.14,hspace=0.10)
plt.savefig(out_png,bbox_inches="tight",pad_inches=0.03)
plt.close()
print(f"Saved: {out_png}")
