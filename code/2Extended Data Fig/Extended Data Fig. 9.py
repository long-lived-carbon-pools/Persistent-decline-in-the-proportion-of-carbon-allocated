# -*- coding: utf-8 -*-
import os,re,numpy as np,pandas as pd,matplotlib.pyplot as plt

csv_path=r"data/2Extended Data Fig/Extended Data Fig. 9/data/1.csv"
out_dir=r"results/2Extended Data Fig"
os.makedirs(out_dir,exist_ok=True)
out_png=os.path.join(out_dir,"Extended Data Fig. 9.png")

df=pd.read_csv(csv_path)
df=df[df["variable"].astype(str).str.match(r"^\d")].copy()

def get_cat(s):
    m=re.match(r"^(\d)_",str(s))
    return int(m.group(1)) if m else np.nan

def simp_name(s):
    p=str(s).split("_")
    return "_".join(p[1:]) if len(p)>1 else s

df["category"]=df["variable"].apply(get_cat)
df=df[df["category"].isin([1,2,3,4,5,6])].copy()

for c in df.columns:
    if c not in ["variable","category"]:
        df[c]=pd.to_numeric(df[c],errors="coerce")

raw_cols=[c for c in df.columns if c not in ["variable","category"]]
raw_cols=[c for c in raw_cols if c not in ("D other","E")]

rename_map={"A":"Tropical","B":"Arid","C":"Temperate"}
climate_cols=[rename_map.get(c,c) for c in raw_cols]

cats=[1,2,3,4,5,6]
cat_labels={1:"DGVM_S2",2:"DGVM_S3",3:"CMIP6",4:"GPP1-4",5:"NPP1-2",6:"Other"}
palette={1:"#1f77b4",2:"#ff7f0e",3:"#2ca02c",4:"#d62728",5:"#9467bd",6:"#8c564b"}
markers={1:"o",2:"s",3:"^",4:"P",5:"D",6:"X"}
offsets={1:-0.25,2:-0.15,3:-0.05,4:0.05,5:0.15,6:0.25}

plt.rcParams["font.family"]="Times New Roman"
plt.rcParams["axes.unicode_minus"]=False

xpos=np.arange(len(climate_cols))
fig,ax=plt.subplots(figsize=(14,7),constrained_layout=True)

ax.set_facecolor("white")
ax.grid(True,color="#dddddd",linestyle="--",linewidth=0.6,alpha=0.7)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# 坐标轴刻度字体 = 20
ax.tick_params(axis="both",which="both",direction="in",
               length=5,labelsize=20)

ax.axhline(0,color="black",linewidth=1.2,linestyle="--")

for i,orig_col in enumerate(raw_cols):

    col_vals=df[orig_col].to_numpy()
    col_cats=df["category"].to_numpy()

    for cat in cats:
        m=(col_cats==cat)&np.isfinite(col_vals)
        if not np.any(m): continue
        y=col_vals[m]
        ax.scatter(np.full(y.size,xpos[i]+offsets[cat]),
                   y,s=80,color=palette[cat],marker=markers[cat],
                   alpha=0.85,linewidths=0.6,edgecolors="black")

    if np.isfinite(col_vals).any():
        idx_max=np.nanargmax(col_vals)
        y_max=col_vals[idx_max]
        cat_max=df.iloc[idx_max]["category"]
        x_max=xpos[i]+offsets.get(cat_max,0.0)
        name_max=simp_name(df.iloc[idx_max]["variable"])

        ax.scatter(x_max,y_max,s=180,facecolors="none",
                   edgecolors="red",linewidths=2.0,zorder=3)

        ymin=np.nanmin(col_vals);ymax=np.nanmax(col_vals)
        dy=(ymax-ymin)*0.03 if ymax>ymin else 0.02

        ax.text(x_max-0.18,y_max+dy,
                f"{name_max}({y_max:.2f})",
                ha="left",va="bottom",fontsize=15,color="black")   # 点标签 = 16

handles=[
    ax.scatter([],[],s=110,color=palette[cat],
               marker=markers[cat],edgecolors="black",
               linewidths=0.6,label=cat_labels[cat])
    for cat in cats
]

# 图例字体 = 22
leg=ax.legend(handles=handles,loc="lower center",
              bbox_to_anchor=(0.5,1.03),ncol=6,
              frameon=True,fontsize=22,
              handletextpad=0.6,columnspacing=1.2)

try:
    leg.get_frame().set_boxstyle("sawtooth",pad=0.05)
except:
    pass

ax.set_xticks(xpos)
ax.set_xticklabels(climate_cols,rotation=0,fontsize=20)

# 坐标轴标签字体 = 22
ax.set_xlabel("Climate Zones",fontsize=22)
ax.set_ylabel("r",fontsize=22)

plt.savefig(out_png,dpi=300)
plt.close()

print("✅ Saved figure:",out_png)
