import sys as _s, os as _o; _R = _o.path.dirname(_o.path.abspath(__file__)); _s.path[:0] = [_R, _o.path.join(_R, '..', 'revision_offline'), _o.path.join(_R, '..')]  # run from the repository root: python code/<folder>/<script>.py
import pandas as pd, numpy as np, matplotlib as mpl, matplotlib.pyplot as plt
mpl.rcParams.update({"font.size":9,"axes.spines.top":False,"axes.spines.right":False,"figure.dpi":150,"savefig.dpi":300})
T='data/revision_results/'
s=pd.read_csv(T+'categorical_skill_summary.csv').set_index('metric'); d=pd.read_csv(T+'categorical_skill_by_district.csv')
NAMES={'corlu':'Çorlu','edirne':'Edirne','kirklareli':'Kırklareli','luleburgaz':'Lüleburgaz','tekirdag':'Tekirdağ','uzunkopru':'Uzunköprü'}
fig,(a,b)=plt.subplots(1,2,figsize=(9,3.4),gridspec_kw=dict(width_ratios=[1,1.15]))
mets=['POD','FAR','CSI','HSS']; x=np.arange(len(mets)); est=s.loc[mets,'estimate']; lo=s.loc[mets,'ci_low']; hi=s.loc[mets,'ci_high']
a.bar(x,est,color=['#264f92','#b85c38','#5c8a5c','#6a5c8a'],width=0.6)
a.errorbar(x,est,yerr=[est-lo,hi-est],fmt='none',ecolor='black',elinewidth=1.2,capsize=4)
for xi,(e,l,h) in enumerate(zip(est,lo,hi)): a.text(xi,h+0.02,f'{e:.2f}',ha='center',fontsize=8)
a.set_xticks(x); a.set_xticklabels(mets); a.set_ylim(0,0.85); a.set_ylabel('score'); a.set_title('(a) Pooled categorical agreement, 95% block-bootstrap intervals',loc='left',fontsize=9)
a.text(0.02,0.95,f"bias = {s.loc['bias','estimate']:.2f} [{s.loc['bias','ci_low']:.2f}, {s.loc['bias','ci_high']:.2f}]",transform=a.transAxes,va='top',fontsize=8)
d=d.sort_values('HSS',ascending=False); y=np.arange(len(d))
b.barh(y,d.HSS,color='#6a5c8a'); b.set_yticks(y); b.set_yticklabels([NAMES[k] for k in d.district]); b.invert_yaxis(); b.set_xlim(0,0.35); b.set_xlabel('Heidke skill score')
for yi,(h,n) in enumerate(zip(d.HSS,d.n)): b.text(h+0.005,yi,f'{h:.2f}  (n = {n})',va='center',fontsize=8)
b.axvline(s.loc['HSS','estimate'],color='0.4',ls='--',lw=1); b.text(s.loc['HSS','estimate']+0.005,len(d)-0.6,'pooled',fontsize=7.5,color='0.4')
b.set_title('(b) HSS by primary district (index and threshold learned without the test year)',loc='left',fontsize=9)
fig.tight_layout(); fig.savefig('figures/Fig7_categorical_agreement.png',bbox_inches='tight',facecolor='white'); print('ok')
