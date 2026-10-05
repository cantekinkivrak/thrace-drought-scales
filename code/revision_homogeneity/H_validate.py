import sys as _s, os as _o; _R = _o.path.dirname(_o.path.abspath(__file__)); _s.path[:0] = [_R, _o.path.join(_R, '..', 'revision_offline'), _o.path.join(_R, '..')]  # run from the repository root: python code/<folder>/<script>.py
import sys, time; sys.path.insert(0,'.')
import numpy as np, pandas as pd
from H_indices import *
t0=time.time()
raw=pd.read_csv('data/trakya_TP_PET_1965_2024.csv')
print(raw.name.unique())
r2=add_pet_d(raw[['year','month','temp','precip','name','lat']])
chk=raw.merge(r2,on=['name','year','month'],suffixes=('','_r'))
print('PET max abs diff', (chk.PET-chk.PET_r).abs().max(), 'D', (chk.D-chk.D_r).abs().max(), flush=True)
arch=pd.read_csv('data/trakya_spi_spei_flexible_1965_2024.csv')
new=indices(raw)
new.to_csv('data/revision_results/H_indices_reproduced.csv',index=False)
c=arch.merge(new,on=['key','year','month'],suffixes=('','_r'))
for k in SCALES:
    for f in ['SPI','SPEI']:
        d=(c[f'{f}{k}']-c[f'{f}{k}_r']).abs()
        print(f,k,'maxdiff',round(d.max(),6),'n',d.notna().sum(), 'archived n',c[f'{f}{k}'].notna().sum())
print('elapsed',time.time()-t0)
