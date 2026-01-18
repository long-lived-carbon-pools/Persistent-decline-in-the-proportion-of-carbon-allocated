import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12

labels = ["Point Count","0.5° Grid Coverage","Accumulated years"]
tree_ring = [4897,1923,151283]
flux_tower = [366,226,1907]
npp_site = [119,95,230]

colors = ["gray","mediumseagreen","indianred"]
names = ["Tree-ring","Flux tower","NPP site"]

# 保持 5684×2699 像素，600dpi => figsize = (9.47, 4.50)
fig,axes = plt.subplots(1,3,figsize=(9.47,4.50),dpi=600)

for i,ax in enumerate(axes):
    values = [tree_ring[i],flux_tower[i],npp_site[i]]
    bars = ax.bar(names,values,color=colors)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x()+bar.get_width()/2,h,f"{h}",ha="center",va="bottom",fontsize=12)
    # 在右上角加文字，白底，无边框，字体大小=12
    ax.text(0.95,0.95,labels[i],
            transform=ax.transAxes,
            ha="right",va="top",
            fontsize=12,
            bbox=dict(facecolor="white",edgecolor="none",boxstyle="round,pad=0.3"))

plt.tight_layout()
plt.savefig(r"results/3Supplementary Fig/Supplementary Fig. 4.png",dpi=600,bbox_inches="tight")
plt.close()
