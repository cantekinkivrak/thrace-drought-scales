import sys as _s, os as _o; _R = _o.path.dirname(_o.path.abspath(__file__)); _s.path[:0] = [_R, _o.path.join(_R, '..', 'revision_offline'), _o.path.join(_R, '..')]  # run from the repository root: python code/<folder>/<script>.py
"""R2 #3/#4, R4 #2: yield-panel inference with few clusters + trend sensitivity."""
import numpy as np, pandas as pd
from rev import *
rng=np.random.default_rng(7)

def wild_cluster_p(d, idx, levels, trend, B=4999):
    """Wild cluster bootstrap-t (Webb 6-point weights), clustering by district, H0: beta_idx = 0.
    Returns (coef, cluster-se, bootstrap p-value, percentile-t 95% CI)."""
    f=fit_fe(d,idx,levels,trend); j=f['names'].index(idx)
    X=f['X']; y=d['yield'].to_numpy(float); clus=d['key'].to_numpy()
    G=np.unique(clus); g_idx=[np.where(clus==g)[0] for g in G]
    bread=np.linalg.pinv(X.T@X)
    def cr_se(res):
        meat=np.zeros((X.shape[1],)*2)
        for ii in g_idx:
            s=(X[ii]*res[ii,None]).sum(0); meat+=np.outer(s,s)
        c=len(G)/(len(G)-1)*(len(y)-1)/(len(y)-X.shape[1])
        V=bread@(c*meat)@bread; return np.sqrt(V[j,j])
    b_hat=f['beta'][j]; se_hat=cr_se(f['res']); t_hat=b_hat/se_hat
    # restricted fit under H0 (drop idx column)
    keep=[i for i in range(X.shape[1]) if i!=j]
    Xr=X[:,keep]; br,*_=np.linalg.lstsq(Xr,y,rcond=None); res_r=y-Xr@br; yhat_r=Xr@br
    webb=np.array([-np.sqrt(1.5),-1,-np.sqrt(.5),np.sqrt(.5),1,np.sqrt(1.5)])
    ts=[]
    for _ in range(B):
        w=rng.choice(webb,size=len(G))
        ystar=yhat_r.copy()
        for gi,ii in enumerate(g_idx): ystar[ii]+=w[gi]*res_r[ii]
        bs,*_=np.linalg.lstsq(X,ystar,rcond=None); rs=ystar-X@bs
        ts.append(bs[j]/cr_se(rs))
    ts=np.array(ts)
    p=np.mean(np.abs(ts)>=abs(t_hat))
    # percentile-t interval (unrestricted residual bootstrap for CI)
    ts2=[]
    for _ in range(B//2):
        w=rng.choice(webb,size=len(G)); ystar=X@f['beta']
        for gi,ii in enumerate(g_idx): ystar[ii]+=w[gi]*f['res'][ii]
        bs,*_=np.linalg.lstsq(X,ystar,rcond=None); rs=ystar-X@bs
        ts2.append((bs[j]-b_hat)/cr_se(rs))
    lo,hi=np.percentile(ts2,[2.5,97.5])
    return b_hat, se_hat, p, (b_hat-hi*se_hat, b_hat-lo*se_hat)

rows=[]
for crop in ['sunflower','wheat']:
    d=load_panel(crop); levels=sorted(d.key.unique())
    for trend in ['linear','district','quadratic']:
        for s in SCALES:
            idx=f'SPEI{s}'; dd=d.dropna(subset=[idx])
            b,se,p,(lo,hi)=wild_cluster_p(dd,idx,levels,trend)
            f=fit_fe(dd,idx,levels,trend)
            rows.append(dict(crop=crop,trend=trend,index=idx,coef=b,cr_se=se,wild_p=p,wild_lo=lo,wild_hi=hi,
                             AIC=f['AIC'],LOYO_RMSE=loyo_rmse(dd,idx,levels,trend)))
    # detrended-yield variant: remove district-specific linear trend from yield first, then no trend term
    dd=d.copy(); dd['yield']=dd['yield'].astype(float)
    for k,g in dd.groupby('key'):
        b=np.polyfit(g.year_c,g['yield'],1); dd.loc[g.index,'yield']=g['yield']-np.polyval(b,g.year_c)+g['yield'].mean()
    for s in SCALES:
        idx=f'SPEI{s}'; x=dd.dropna(subset=[idx])
        b,se,p,(lo,hi)=wild_cluster_p(x,idx,levels,'none'); f=fit_fe(x,idx,levels,'none')
        rows.append(dict(crop=crop,trend='pre-detrended (district)',index=idx,coef=b,cr_se=se,wild_p=p,wild_lo=lo,wild_hi=hi,
                         AIC=f['AIC'],LOYO_RMSE=loyo_rmse(x,idx,levels,'none')))
t=pd.DataFrame(rows); t.to_csv('data/revision_results/A_yield_wildcluster_trends.csv',index=False)
pd.set_option('display.width',220)
for crop in ['sunflower','wheat']:
    print(f'\n===== {crop.upper()} =====')
    sub=t[t.crop==crop]
    for tr,g in sub.groupby('trend',sort=False):
        best_rmse=g.loc[g.LOYO_RMSE.idxmin(),'index']; best_aic=g.loc[g.AIC.idxmin(),'index']
        print(f'\n-- trend: {tr}  | best by LOYO-RMSE: {best_rmse} | best by AIC: {best_aic}')
        print(g[['index','coef','cr_se','wild_p','wild_lo','wild_hi','AIC','LOYO_RMSE']].to_string(index=False,float_format=lambda v:f'{v:8.2f}'))
