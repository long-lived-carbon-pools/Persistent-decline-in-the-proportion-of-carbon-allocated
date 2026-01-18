# -*- coding: utf-8 -*-
import os,numpy as np,pandas as pd,matplotlib.pyplot as plt,matplotlib as mpl
mpl.rcParams['font.family']='Times New Roman'
mpl.rcParams['axes.unicode_minus']=False
mpl.rcParams['font.size']=22
NA_r_diff_dir=r"D:\all\3Supplementary Fig\Supplementary Fig. 3\data\1"
na_raw_dir=r"D:\all\3Supplementary Fig\Supplementary Fig. 3\data\2"
EU_r_diff_dir=r"D:\all\3Supplementary Fig\Supplementary Fig. 3\data\3"
eu_raw_dir=r"D:\all\3Supplementary Fig\Supplementary Fig. 3\data\4"
out_dir=r"D:\all\3Supplementary Fig\Supplementary Fig. 3"

os.makedirs(out_dir,exist_ok=True)
out_png=os.path.join(out_dir,"Supplementary Fig. 3.png")
na_label_map={1:"A",2:"B",3:"C",4:"Dfa",5:"Dfb"}
eu_label_map={2:"B",3:"C",4:"Dfa",5:"Dfb",6:"D other"}
agg_defs={102:"Warm",103:"Cold",104:"All"}
na_order=[1,2,3,4,5,102,103,104]
label_display_map={"A":"Tropical","B":"Arid","C":"Temperate"}
def tag_name_na(code):return na_label_map[code] if code in na_label_map else agg_defs[code]
def tag_name_eu(code):return eu_label_map[code] if code in eu_label_map else agg_defs[code]
def read_curve(dir_path,tag):
    fp=os.path.join(dir_path,f"{tag}.csv")
    if not os.path.exists(fp):return None
    df=pd.read_csv(fp)
    if "year" not in df.columns or "mean_r" not in df.columns:return None
    return df["year"].to_numpy(),df["mean_r"].to_numpy()
def peak_of(series,year_min=None):
    if series is None:return None
    x,y=series
    m=np.isfinite(y)
    if year_min is not None:m=m&(x>=year_min)
    if not m.any():return None
    yi=y[m];xi=x[m]
    k=int(np.nanargmax(yi))
    return int(xi[k]),float(yi[k])
def composite_peak(nd,ed):
    if nd is None and ed is None:return None
    if nd is None:return peak_of(ed)
    if ed is None:return peak_of(nd)
    xn,yn=nd;xe,ye=ed
    dn=pd.DataFrame({"year":xn,"na":yn})
    de=pd.DataFrame({"year":xe,"eu":ye})
    df=dn.merge(de,on="year",how="outer").sort_values("year")
    zn=df["na"].to_numpy(dtype=float);ze=df["eu"].to_numpy(dtype=float)
    m1=np.isfinite(zn);m2=np.isfinite(ze)
    if m1.any():
        mu=float(np.nanmean(zn[m1]));sd=float(np.nanstd(zn[m1]));sd=sd if sd>0 else 1.0
        zn=(zn-mu)/sd
    else:
        zn=np.zeros_like(zn)
    if m2.any():
        mu=float(np.nanmean(ze[m2]));sd=float(np.nanstd(ze[m2]));sd=sd if sd>0 else 1.0
        ze=(ze-mu)/sd
    else:
        ze=np.zeros_like(ze)
    comp=np.where(np.isfinite(zn)&np.isfinite(ze),(zn+ze)/2.0,np.where(np.isfinite(zn),zn,ze))
    mm=np.isfinite(comp)
    if not mm.any():return None
    k=int(np.nanargmax(comp[mm]))
    year=int(df["year"].to_numpy()[mm][k])
    val=float(comp[mm][k])
    return year,val
curves={};all_vals=[]
for code in na_order:
    tag_na=tag_name_na(code)
    na_d=read_curve(NA_r_diff_dir,tag_na)
    na_r=read_curve(na_raw_dir,tag_na)
    if code in eu_label_map or code in (102,103,104):
        tag_eu=tag_name_eu(code)
        eu_d=read_curve(EU_r_diff_dir,tag_eu)
        eu_r=read_curve(eu_raw_dir,tag_eu)
    else:
        eu_d=None;eu_r=None
    curves[code]={"NA_r_diff":na_d,"NA_raw":na_r,"EU_r_diff":eu_d,"EU_raw":eu_r}
    for k in ("NA_r_diff","NA_raw","EU_r_diff","EU_raw"):
        v=curves[code][k]
        if v is not None:
            y=v[1]
            all_vals.append(y[np.isfinite(y)])
if all_vals:
    all_vals=np.concatenate(all_vals)
    global_min=float(np.nanmin(all_vals))
    global_max=float(np.nanmax(all_vals))
else:
    global_min=-0.1;global_max=0.1
pad=(global_max-global_min)*0.1
global_ymin=global_min-pad
global_ymax=global_max+pad
yticks=np.linspace(global_ymin,global_ymax,5)
cycle=list(mpl.rcParams['axes.prop_cycle'].by_key()['color'])
c_na=cycle[0] if len(cycle)>0 else "#1f77b4"
c_eu=cycle[1] if len(cycle)>1 else "#ff7f0e"
fig,axes=plt.subplots(2,4,figsize=(16,6.8),constrained_layout=True);axes=axes.ravel()
legend_handles={"EU r_raw":None,"EU r_diff":None,"NA r_raw":None,"NA r_diff":None}
for i,code in enumerate(na_order):
    ax=axes[i]
    tag=tag_name_na(code)
    disp_tag=label_display_map.get(tag,tag)
    nd=curves[code]["NA_r_diff"];nr=curves[code]["NA_raw"];ed=curves[code]["EU_r_diff"];er=curves[code]["EU_raw"]
    has_any=False
    if er is not None and np.isfinite(er[1]).any():
        p_er=ax.plot(er[0],er[1],linewidth=1.6,alpha=0.45,label="EU r_raw",color=c_eu);has_any=True
        if legend_handles["EU r_raw"] is None:legend_handles["EU r_raw"]=p_er[0]
    if ed is not None and np.isfinite(ed[1]).any():
        p_ed=ax.plot(ed[0],ed[1],linewidth=3.0,label="EU r_diff",color=c_eu);has_any=True
        if legend_handles["EU r_diff"] is None:legend_handles["EU r_diff"]=p_ed[0]
    if nr is not None and np.isfinite(nr[1]).any():
        p_nr=ax.plot(nr[0],nr[1],linewidth=1.6,alpha=0.45,label="NA r_raw",color=c_na);has_any=True
        if legend_handles["NA r_raw"] is None:legend_handles["NA r_raw"]=p_nr[0]
    if nd is not None and np.isfinite(nd[1]).any():
        p_nd=ax.plot(nd[0],nd[1],linewidth=3.0,label="NA r_diff",color=c_na);has_any=True
        if legend_handles["NA r_diff"] is None:legend_handles["NA r_diff"]=p_nd[0]
    if not has_any:
        ax.text(0.5,0.5,"No data",ha="center",va="center",fontsize=18);ax.set_axis_off();continue
    ax.set_ylim(global_ymin,global_ymax)
    ax.set_yticks(yticks)
    ylabels=[("0" if abs(round(v,1))<1e-6 else f"{v:.1f}") for v in yticks]
    ax.set_yticklabels(ylabels)
    ax.margins(x=0.01)
    if i>=4:ax.set_xlabel("Year",fontsize=22)
    if i%4==0:ax.set_ylabel("r",fontsize=22)
    ax.tick_params(axis="both",labelsize=20)
    if i<4:ax.set_xticklabels([])
    if i not in (0,4):ax.set_yticklabels([])
    y_top=global_ymax-(global_ymax-global_ymin)*0.04
    y_bot_main=global_ymin+(global_ymax-global_ymin)*0.08
    y_bot_alt=global_ymin+(global_ymax-global_ymin)*0.18
    use_top=(i==0)
    if code==103:
        if use_top:
            ax.axvline(1984,color="k",linestyle="--",linewidth=1.2)
            ax.text(1984,y_top,"1984",ha="center",va="top",fontsize=20)
            p_after_nd=peak_of(nd,year_min=2010)
            if p_after_nd is not None:
                ax.axvline(p_after_nd[0],color="k",linestyle="--",linewidth=1.2)
                ax.text(p_after_nd[0,],y_bot_alt,str(p_after_nd[0]),ha="center",va="top",fontsize=20)
        else:
            ax.axvline(1984,color="k",linestyle="--",linewidth=1.2)
            ax.text(1984,y_bot_alt,"1984",ha="center",va="top",fontsize=20)
            p_after_nd=peak_of(nd,year_min=2010)
            if p_after_nd is not None:
                ax.axvline(p_after_nd[0],color="k",linestyle="--",linewidth=1.2)
                ax.text(p_after_nd[0],y_bot_main,str(p_after_nd[0]),ha="center",va="top",fontsize=20)
    else:
        p_na=peak_of(nd)
        p_eu=peak_of(ed)
        years=[]
        if p_na is not None:years.append(p_na[0])
        if p_eu is not None:years.append(p_eu[0])
        if len(years)==1:
            year=years[0]
            ax.axvline(year,color="k",linestyle="--",linewidth=1.2)
            if use_top:
                ax.text(year,y_top,str(year),ha="center",va="top",fontsize=20)
            else:
                ax.text(year,y_bot_main,str(year),ha="center",va="top",fontsize=20)
        elif len(years)==2:
            if years[0]==years[1]:
                year=years[0]
                ax.axvline(year,color="k",linestyle="--",linewidth=1.2)
                if use_top:
                    ax.text(year,y_top,str(year),ha="center",va="top",fontsize=20)
                else:
                    ax.text(year,y_bot_main,str(year),ha="center",va="top",fontsize=20)
            else:
                if use_top:
                    if p_na is not None:
                        year_na=p_na[0]
                        ax.axvline(year_na,color="k",linestyle="--",linewidth=1.2)
                        ax.text(year_na,y_top,str(year_na),ha="center",va="top",fontsize=20)
                    if p_eu is not None:
                        year_eu=p_eu[0]
                        ax.axvline(year_eu,color="k",linestyle="--",linewidth=1.2)
                        ax.text(year_eu,y_bot_alt,str(year_eu),ha="center",va="top",fontsize=20)
                else:
                    if p_na is not None:
                        year_na=p_na[0]
                        ax.axvline(year_na,color="k",linestyle="--",linewidth=1.2)
                        ax.text(year_na,y_bot_alt,str(year_na),ha="center",va="top",fontsize=20)
                    if p_eu is not None:
                        year_eu=p_eu[0]
                        ax.axvline(year_eu,color="k",linestyle="--",linewidth=1.2)
                        ax.text(year_eu,y_bot_main,str(year_eu),ha="center",va="top",fontsize=20)
    ax.text(0.02,0.90,disp_tag,transform=ax.transAxes,ha="left",va="top",fontsize=22,fontweight="bold")
order=["EU r_raw","EU r_diff","NA r_raw","NA r_diff"]
legend_items=[legend_handles[k] for k in order if legend_handles[k] is not None]
if legend_items:
    lg=axes[0].legend(handles=legend_items,fontsize=16,frameon=True,loc="lower right",borderpad=0.1,handlelength=1.5,handletextpad=0.5,markerscale=0.8)
    try:lg.get_frame().set_boxstyle("sawtooth",pad=0.05)
    except:pass
for j in range(len(na_order),len(axes)):fig.delaxes(axes[j])
fig.savefig(out_png,dpi=600,bbox_inches="tight")
plt.close(fig)
