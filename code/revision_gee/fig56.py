import sys as _s, os as _o; _R = _o.path.dirname(_o.path.abspath(__file__)); _s.path[:0] = [_R, _o.path.join(_R, '..', 'revision_offline'), _o.path.join(_R, '..')]  # run from the repository root: python code/<folder>/<script>.py
import sys; sys.path.insert(0,'.')
import numpy as np, pandas as pd, matplotlib as mpl, matplotlib.pyplot as plt
from scipy import stats
from rev import IDX, WIN, CORE, ALL7, metric_table
mpl.rcParams.update({"font.size":9,"axes.spines.top":False,"axes.spines.right":False,"figure.dpi":150,"savefig.dpi":300})
OUT='figures/'
NAMES={'corlu':'Çorlu','edirne':'Edirne','kirklareli':'Kırklareli','luleburgaz':'Lüleburgaz','tekirdag':'Tekirdağ','uzunkopru':'Uzunköprü','ipsala':'İpsala'}
# ---------- Figure 5: compositing artefact ----------
q=pd.read_csv('data/revision_results/H3_ipsala_scene_quality.csv').set_index('variant')
sp=pd.read_csv('data/revision_results/H3_ipsala_subperiods.csv'); mi=pd.read_csv('data/revision_results/H6_max_inflation.csv').set_index('key')
order=['corlu','uzunkopru','edirne','kirklareli','tekirdag','luleburgaz','ipsala']
rmax=q.loc['paper (all scenes, NDVI max / LST mean)']; rmed=q.loc['median composite (all scenes)']; rnos=q.loc['no SLC-off scenes']
fig,axes=plt.subplots(1,3,figsize=(11,3.6),gridspec_kw=dict(width_ratios=[1.35,1,1]))
ax=axes[0]; x=np.arange(len(order)); w=0.27
ax.bar(x-w,[rmax[f'r_{k}'] if k!='ipsala' else rmax['r_ipsala'] for k in order],w,color='#b9c4d3',label='NDVI-max / LST-mean (submitted)')
ax.bar(x,[rnos[f'r_{k}'] if k!='ipsala' else rnos['r_ipsala'] for k in order],w,color='#7d9ac0',label='same, SLC-off scenes excluded')
ax.bar(x+w,[rmed[f'r_{k}'] if k!='ipsala' else rmed['r_ipsala'] for k in order],w,color='#264f92',label='median composite (revised primary)')
ax.set_xticks(x); ax.set_xticklabels([NAMES[k] for k in order],rotation=35,ha='right'); ax.set_ylabel('June–October VHI–SPEI-3 correlation (r)'); ax.set_ylim(0,0.82)
ax.axvspan(5.5,6.5,color='#f3e2de',zorder=0); ax.legend(frameon=False,fontsize=7.2,loc='upper left',ncol=1); ax.set_title('(a) District coupling by compositing scheme',loc='left',fontsize=9)
ax=axes[1]; per=['1985-2002 (pre SLC-off)','2003-2012','2013-2024']; lab=['1985–2002','2003–2012\n(L5 + SLC-off L7)','2013–2024']
for comp,col,mk in [('paper','#b9c4d3','o'),('no SLC-off','#7d9ac0','s'),('median','#264f92','D')]:
    s=sp[sp.composite==comp].set_index('period').loc[per]
    ax.plot(range(3),s.r_ipsala,marker=mk,color=col,lw=1.6,label={'paper':'NDVI-max (submitted)','no SLC-off':'NDVI-max, no SLC-off','median':'median'}[comp])
    ax.plot(range(3),s.plateau,ls=':',color=col,lw=1.2)
ax.set_xticks(range(3)); ax.set_xticklabels(lab,fontsize=8); ax.set_ylim(0,0.7); ax.set_ylabel('r'); ax.legend(frameon=False,fontsize=7.5,loc='lower left')
ax.text(0.02,0.97,'solid: İpsala · dotted: plateau mean',transform=ax.transAxes,va='top',fontsize=7.5); ax.set_title('(b) İpsala by sub-period',loc='left',fontsize=9)
ax=axes[2]; y=np.arange(len(order))
ax.barh(y,[mi.loc[k,'max_minus_median'] for k in order],color='#9cb3d0',label='median month')
ax.barh(y,[mi.loc[k,'max_minus_median_p90'] for k in order],color='none',edgecolor='#264f92',lw=1.3,label='90th percentile month')
ax.set_yticks(y); ax.set_yticklabels([NAMES[k] for k in order]); ax.invert_yaxis(); ax.set_xlabel('NDVI max − median, June–October months'); ax.set_xlim(0,0.34)
ax.legend(frameon=False,fontsize=7.5,loc='upper right'); ax.set_title('(c) Inflation of the maximum operator',loc='left',fontsize=9)
fig.tight_layout(); fig.savefig(OUT+'Fig5_compositing_artefact.png',bbox_inches='tight',facecolor='white'); plt.close(fig)
# ---------- Figure 6: season-dependent scales ----------
m=pd.read_csv('data/trakya_district_vhi_vci_tci_monthly.csv'); m['key']=m.name.str.lower()
idx=pd.read_csv('data/trakya_spi_spei_flexible_1965_2024.csv').drop(columns='name'); mm=m.merge(idx,on=['key','year','month'])
SC=[1,3,6,9,12]
wins={'Apr–Jun (wheat-dominated canopy)':([4,5,6],'#5aa469'),'Jun–Oct (primary window)':([6,7,8,9,10],'#1c7c54'),'Jul–Sep (sunflower-dominated canopy)':([7,8,9],'#c98a1b')}
boot={'Apr–Jun (wheat-dominated canopy)':'SPEI-1: 92%','Jun–Oct (primary window)':'SPEI-3: 68%','Jul–Sep (sunflower-dominated canopy)':'SPEI-9: 78%'}
fig,axes=plt.subplots(1,2,figsize=(10,3.9))
ax=axes[0]
for lab,(mo,col) in wins.items():
    s=mm[mm.key.isin(CORE)&mm.month.isin(mo)].dropna(subset=['VHI']+IDX)
    r=[stats.pearsonr(s.VHI,s[f'SPEI{k}']).statistic for k in SC]; ax.plot(SC,r,marker='o',color=col,lw=2,label=f'{lab} — n = {len(s)}')
    kb=int(np.argmax(r)); ax.scatter([SC[kb]],[r[kb]],s=160,facecolor='none',edgecolor=col,lw=2,zorder=4)
    ax.annotate(boot[lab],(SC[kb],r[kb]),xytext=(6,8),textcoords='offset points',fontsize=7.5,color=col)
ax.set_xticks(SC); ax.set_xlabel('SPEI accumulation scale (months)'); ax.set_ylabel('VHI–SPEI Pearson r'); ax.axhline(0,color='0.6',lw=0.8)
ax.set_ylim(-0.42,0.68); ax.legend(frameon=False,fontsize=7.5,loc='lower left',title='circled: best scale; label: block-bootstrap selection frequency (1,000 replicates)',title_fontsize=7); ax.set_title('(a) Canopy stress: the optimum lengthens through the season',loc='left',fontsize=9)
ax=axes[1]
for crop,mon,col,mk in [('sunflower',8,'#c98a1b','s'),('wheat',6,'#5b8fd6','^')]:
    p=pd.read_csv(f'data/yield_panel_{crop}.csv').dropna(subset=['yield']); levels=sorted(p.key.unique())
    X=np.column_stack([np.ones(len(p)),p.year_c]+[(p.key==l).astype(float) for l in levels[1:]])
    def resid(v): b,*_=np.linalg.lstsq(X,v,rcond=None); return v-X@b
    ry=resid(p['yield'].astype(float).to_numpy()); r=[stats.pearsonr(ry,resid(p[f'SPEI{k}'].to_numpy()))[0] for k in SC]
    ax.plot(SC,r,marker=mk,color=col,lw=2,label=f'{crop} yield ({"August" if crop=="sunflower" else "June"} SPEI, within-district)')
ax.axvspan(2.5,12.5,color='#c98a1b',alpha=0.08,lw=0); ax.text(7.5,0.31,'sunflower: scale not identified\n(bootstrap selection 37–71% across trend specifications)',ha='center',fontsize=7.5,color='#8a5f12')
ax.set_xticks(SC); ax.set_xlabel('SPEI accumulation scale (months)'); ax.set_ylabel('within-district correlation'); ax.axhline(0,color='0.6',lw=0.8); ax.set_ylim(-0.2,0.62)
ax.legend(frameon=False,fontsize=7.5,loc='upper left'); ax.set_title('(b) District yields: direction, not a point optimum',loc='left',fontsize=9)
fig.tight_layout(); fig.savefig(OUT+'Fig6_season_dependent_scales.png',bbox_inches='tight',facecolor='white'); plt.close(fig)
print('done')
