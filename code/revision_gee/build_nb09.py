"""Builds 09_gee_revision_extractions.ipynb (Colab / geemap, same conventions as notebooks 01 and 08)."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s))
code = lambda s: C.append(nbf.v4.new_code_cell(s))

md(r"""# 09 · Revision extractions — Turkish Thrace Landsat VHI *(v2, 30 Sept 2026)*

One GEE pass that produces everything the reviewers asked for on the remote-sensing side.
Every Landsat scene is reduced to district means **under several masks at once** (multi-band image →
one `reduceRegions` call), so the cost is one extraction, not six.

| Reviewer point | What this notebook exports | Offline step (Python, sent back) |
|---|---|---|
| R2 #7a CORINE vintage (single 2018 snapshot over 40 yr) | NDVI/LST under CORINE-211 of **1990, 2000, 2006, 2012, 2018** and the **stable-211 intersection**, plus ESA WorldCover cropland | scale selection & İpsala contrast per mask; time-matched mask |
| R2 #7c compositing (NDVI-max vs LST-mean) | **per-scene** district means (not monthly) | alternative composites (mean/median/same-scene pairing) |
| R4 #1 Landsat-7 SLC-off | `sensor` + `slc_off` flag on every row | ablation without SLC-off scenes |
| R2 #1 crop-specific optimum (wheat vs sunflower) | per-year **phenology masks** inside 211: wheat-like (spring peak) and sunflower-like (summer peak) pixels | crop-specific VHI → crop-specific scale selection |
| İpsala irrigation (contrast district) | per-district fractions: CORINE 212/213, **LGRIP30** irrigated share *inside* 211 pixels, GFSAD1000, WorldCover, summer-green share | Table for §3.5 / §4 |

**Run order:** 0 → 7. §3b and §5b are small tables (seconds). §6 is the slow cell (whole record).
All heavy steps run as Drive export tasks (folder `thrace_revision`).
Send back: `rev_scene_means_1985_2024.csv` (or the per-year files), `rev_district_landcover_fractions.csv`,
`rev_phenology_fractions.csv`.""")

md("## 0 · Setup")
code(r"""!pip install -q geemap earthengine-api
import ee, geemap, numpy as np, pandas as pd, time, os
ee.Authenticate()
ee.Initialize(project='vhi-landsat')   # <-- your Cloud project id
print('EE:', ee.String('OK').getInfo())""")

md("## 1 · Parameters")
code(r"""START_YEAR = 1985
END_YEAR   = 2024
SENSORS    = ['L5','L7','L8','L9']
HARMONIZE  = True          # Roy et al. 2016 OLI-equivalent transformation for TM/ETM+ (as in notebook 01)
CLOUD_BITS = True
SCALE_M    = 30
MODE       = 'drive'       # 'drive' (Export.table.toDrive per year, resumable) or 'sync' (ee_to_df)
DRIVE_FOLDER = 'thrace_revision'
L7_SLC_OFF = '2003-05-31'

# Explicit, validated yield-district -> (GAUL province, GAUL ADM2) — identical to notebook 08.
DISTRICT_MAP = {
    'Edirne':     ('Edirne',     'Merkez'),
    'Uzunkopru':  ('Edirne',     'Uzunkopru'),
    'Ipsala':     ('Edirne',     'Ipsala'),
    'Luleburgaz': ('Kirklareli', 'Luleburgaz'),
    'Kirklareli': ('Kirklareli', 'Merkez'),
    'Tekirdag':   ('Tekirdag',   'Merkez'),
    'Corlu':      ('Tekirdag',   'Corlu'),
}
PLATEAU6 = ['Corlu','Edirne','Kirklareli','Luleburgaz','Tekirdag','Uzunkopru']

# CORINE vintages available in GEE (COPERNICUS/CORINE/V20/100m/<year>)
CORINE_YEARS = [1990, 2000, 2006, 2012, 2018]

# Phenology-mask thresholds (per-pixel, per-year, inside CORINE-211 of 2018).
# wheat-like: green in spring, bare by mid-summer; sunflower-like: bare/low in spring, green in mid-summer.
PH = dict(wheat_spring_min=0.45, wheat_summer_max=0.30, sun_summer_min=0.45, sun_spring_max=0.40)
SPRING = (4, 5)   # months for the spring window (Apr–May)
SUMMER = (7, 8)   # months for the summer window (Jul–Aug)""")

md("## 2 · District polygons (GAUL) + ROI")
code(r"""gaul = ee.FeatureCollection('FAO/GAUL/2015/level2').filter(ee.Filter.eq('ADM0_NAME','Turkey'))
def make_district(key, prov, adm2):
    f = gaul.filter(ee.Filter.And(ee.Filter.eq('ADM1_NAME', prov), ee.Filter.eq('ADM2_NAME', adm2))).first()
    return ee.Feature(ee.Feature(f).geometry(), {'name': key})
districts = ee.FeatureCollection([make_district(k,p,a) for k,(p,a) in DISTRICT_MAP.items()])
print('District polygons built:', districts.size().getInfo())
print('Districts:', sorted(districts.aggregate_array('name').getInfo()))
roi = districts.geometry().bounds()""")

md(r"""## 3 · Masks: CORINE vintages, stable-211, WorldCover, irrigation layers

All masks are evaluated at 30 m at extraction time (`updateMask` on the Landsat image).
`stable211` = pixel is class 211 in **every** vintage (the most conservative rainfed mask).""")
code(r"""corine = {y: ee.Image(f'COPERNICUS/CORINE/V20/100m/{y}').select('landcover') for y in CORINE_YEARS}
m211 = {y: corine[y].eq(211) for y in CORINE_YEARS}
stable211 = ee.ImageCollection([m211[y] for y in CORINE_YEARS]).reduce(ee.Reducer.min()).rename('stable')
wc = ee.ImageCollection('ESA/WorldCover/v200').first().select('Map')
wc40 = wc.eq(40)                                              # WorldCover 2021 cropland, 10 m

MASKS = {f'v{y}': m211[y] for y in CORINE_YEARS}
MASKS['stable'] = stable211
MASKS['wc40']   = wc40

for k, im in MASKS.items():
    cnt = im.selfMask().reduceRegion(ee.Reducer.count(), roi, 100, maxPixels=1e10).getInfo()
    print(f'{k:8s} pixels(100 m) in ROI: {list(cnt.values())[0] if cnt else 0}')

# ---- irrigation layers (for §3b table; not used as extraction masks) ----
irrig_layers = {}
irrig_layers['corine2018_212_213'] = corine[2018].eq(212).Or(corine[2018].eq(213))   # permanently irrigated + rice
irrig_layers['corine2018_213_rice'] = corine[2018].eq(213)
try:   # LGRIP30 (Teluguntla et al. 2023): 30 m, 2015 nominal; 2 = irrigated cropland, 3 = rainfed cropland
    lgrip = ee.ImageCollection('projects/sat-io/open-datasets/GFSAD/LGRIP30').mosaic()
    irrig_layers['lgrip30_irrigated'] = lgrip.eq(2)
    irrig_layers['lgrip30_rainfed']   = lgrip.eq(3)
    print('LGRIP30 loaded')
except Exception as e:
    print('LGRIP30 not available:', str(e)[:80])
try:   # GFSAD1000 v1: 1 km; 1 = irrigation major, 2 = irrigation minor, 3/4/5 = rainfed classes
    gf = ee.Image('USGS/GFSAD1000_V1').select('landcover')
    irrig_layers['gfsad1000_irrigated'] = gf.eq(1).Or(gf.eq(2))
    print('GFSAD1000 loaded')
except Exception as e:
    print('GFSAD1000 not available:', str(e)[:80])""")

md(r"""### 3b · District land-cover / irrigation fraction table

For each district: total area, CORINE-211 share in each vintage, irrigated classes, and — the key number for
İpsala — the share of **2018 class-211 pixels** that an independent 30 m product (LGRIP30) labels as irrigated,
plus the share of 211 pixels that are green in mid-summer (Jul–Aug max NDVI ≥ 0.5, Landsat 8/9 2013–2024) — a
direct, sensor-based signature of summer irrigation (rice/maize) inside the nominally rainfed class.""")
code(r"""def frac_table():
    area = ee.Image.pixelArea()
    layers = {}
    for y in CORINE_YEARS:
        layers[f'corine{y}_211'] = m211[y]
    layers['stable211'] = stable211
    layers['wc40'] = wc40
    layers.update(irrig_layers)
    # irrigation-inside-211 (2018)
    for k in ['lgrip30_irrigated', 'gfsad1000_irrigated', 'corine2018_213_rice']:
        if k in irrig_layers:
            layers[f'{k}_within211'] = irrig_layers[k].And(m211[2018])
    img = ee.Image.cat([area.rename('area_m2')] + [area.updateMask(v).rename(k) for k, v in layers.items()])
    fc = img.reduceRegions(districts, ee.Reducer.sum(), scale=100, tileScale=4)
    df = geemap.ee_to_df(fc)
    cols = [c for c in df.columns if c not in ('name',)]
    out = df[['name'] + cols].copy()
    for c in cols:
        if c != 'area_m2':
            out[c + '_frac'] = out[c] / out['area_m2']
    for k in ['lgrip30_irrigated', 'gfsad1000_irrigated', 'corine2018_213_rice']:
        if f'{k}_within211' in out:
            out[f'{k}_share_of_211'] = out[f'{k}_within211'] / out['corine2018_211']
    out['area_km2'] = out['area_m2'] / 1e6
    return out

frac = frac_table()
show = ['name','area_km2'] + [c for c in frac.columns if c.endswith('_frac') or c.endswith('_share_of_211')]
print(frac[show].round(3).to_string(index=False))""")

md("## 4 · Landsat C2 L2 collection (NDVI + LST) with sensor / SLC-off tags")
code(r"""SR_SCALE,SR_OFF = 0.0000275,-0.2; ST_SCALE,ST_OFF = 0.00341802,149.0
SENSOR_DATES={'L5':('1984-03-01','2012-05-05'),'L7':('1999-05-28','2022-04-06'),
              'L8':('2013-03-18','2025-12-31'),'L9':('2021-10-31','2025-12-31')}
CFG={'L5':('LANDSAT/LT05/C02/T1_L2',['SR_B1','SR_B2','SR_B3','SR_B4','SR_B5','SR_B7','ST_B6','QA_PIXEL']),
     'L7':('LANDSAT/LE07/C02/T1_L2',['SR_B1','SR_B2','SR_B3','SR_B4','SR_B5','SR_B7','ST_B6','QA_PIXEL']),
     'L8':('LANDSAT/LC08/C02/T1_L2',['SR_B2','SR_B3','SR_B4','SR_B5','SR_B6','SR_B7','ST_B10','QA_PIXEL']),
     'L9':('LANDSAT/LC09/C02/T1_L2',['SR_B2','SR_B3','SR_B4','SR_B5','SR_B6','SR_B7','ST_B10','QA_PIXEL'])}
COMMON=['blue','green','red','nir','swir1','swir2','thermal','QA_PIXEL']; OPT=['blue','green','red','nir','swir1','swir2']
ROY_S={'blue':0.8474,'green':0.8483,'red':0.9047,'nir':0.8462,'swir1':0.8937,'swir2':0.9071}
ROY_I={'blue':0.0003,'green':0.0088,'red':0.0061,'nir':0.0412,'swir1':0.0254,'swir2':0.0172}

def mask_qa(img):
    qa=img.select('QA_PIXEL')
    bad=(qa.bitwiseAnd(1<<1).neq(0).Or(qa.bitwiseAnd(1<<2).neq(0)).Or(qa.bitwiseAnd(1<<3).neq(0))
         .Or(qa.bitwiseAnd(1<<4).neq(0)).Or(qa.bitwiseAnd(1<<5).neq(0)))
    return img.updateMask(bad.Not())

def prep(sensor):
    coll_id,bands=CFG[sensor]; s0,s1=SENSOR_DATES[sensor]
    lo=f'{max(int(s0[:4]),START_YEAR)}-01-01'; hi=f'{min(int(s1[:4]),END_YEAR)}-12-31'
    coll=ee.ImageCollection(coll_id).filterBounds(roi).filterDate(lo,hi)
    if CLOUD_BITS: coll=coll.map(mask_qa)
    def _p(img):
        img=img.select(bands,COMMON); opt=img.select(OPT).multiply(SR_SCALE).add(SR_OFF)
        if HARMONIZE and sensor in ('L5','L7'):
            opt=opt.multiply(ee.Image([ROY_S[b] for b in OPT])).add(ee.Image([ROY_I[b] for b in OPT]))
        lst=img.select('thermal').multiply(ST_SCALE).add(ST_OFF).subtract(273.15).rename('LST')
        ndvi=opt.normalizedDifference(['nir','red']).rename('NDVI')
        if sensor == 'L7':   # 1 = acquired after the scan-line-corrector failure (gap-striped scene)
            slc_off = ee.Algorithms.If(ee.Number(img.get('system:time_start')).gte(ee.Date(L7_SLC_OFF).millis()), 1, 0)
        else:
            slc_off = 0
        return (ndvi.addBands(lst)
                .set({'system:time_start': img.get('system:time_start'), 'sensor': sensor,
                      'slc_off': slc_off, 'scene_id': img.get('LANDSAT_PRODUCT_ID'),
                      'wrs_path': img.get('WRS_PATH'), 'wrs_row': img.get('WRS_ROW')}))
    return coll.map(_p)

landsat = prep(SENSORS[0])
for s in SENSORS[1:]: landsat = landsat.merge(prep(s))
landsat = landsat.select(['NDVI','LST'])
print('Collection built (lazy).')""")

md(r"""## 5 · Phenology masks (per year, inside CORINE-211 of 2018) — v2

Per pixel and year: spring (Apr–May) and summer (Jul–Aug) **maximum NDVI** from all cloud-free scenes.
v2 uses a *relative* rule so that a cloudy season does not silently empty a class:
`wheat-like` = spring − summer ≥ 0.15 and spring ≥ 0.40; `sunflower-like` = summer − spring ≥ 0.15 and summer ≥ 0.40;
both require the pixel to be observed in **both** windows (`observed`), and class shares are reported relative to the
observed area, not to the whole 211 area.""")
code(r"""def window_max(year, months):
    start = ee.Date.fromYMD(int(year), months[0], 1); end = ee.Date.fromYMD(int(year), months[1], 1).advance(1, 'month')
    return landsat.filterDate(start, end).select('NDVI').max()

def pheno_masks(year):
    spring = window_max(year, SPRING); summer = window_max(year, SUMMER)
    base = m211[2018]
    observed = spring.mask().And(summer.mask()).And(base).rename('observed')
    d = spring.subtract(summer)
    wheat = d.gte(PH['delta']).And(spring.gte(PH['min_peak'])).And(observed).rename('wheat')
    sun   = d.lte(-PH['delta']).And(summer.gte(PH['min_peak'])).And(observed).rename('sun')
    return wheat, sun, observed

PH = dict(delta=0.15, min_peak=0.40)   # v2 thresholds (relative rule)

# quick number check for one recent year (shares relative to the observed 211 area)
w, s_, o = pheno_masks(2020)
img = ee.Image.cat([ee.Image.pixelArea().updateMask(m211[2018]).rename('a211'),
                    ee.Image.pixelArea().updateMask(o).rename('observed'),
                    ee.Image.pixelArea().updateMask(w).rename('wheat'),
                    ee.Image.pixelArea().updateMask(s_).rename('sun')])
try:
    chk = geemap.ee_to_df(img.reduceRegions(districts, ee.Reducer.sum(), scale=100, tileScale=8))
    chk['observed_share'] = chk.observed / chk.a211; chk['wheat_share'] = chk.wheat / chk.observed; chk['sun_share'] = chk.sun / chk.observed
    print(chk[['name','observed_share','wheat_share','sun_share']].round(3).to_string(index=False))
except Exception as e:
    print('sync check skipped:', str(e)[:80])""")

md(r"""### 5b · Phenology class fractions for every year + September-green irrigation signature (two Drive tasks)

`rev_phenology_fractions.csv`: one row per district-year with `a211, observed, wheat, sun` areas (m²).
`rev_september_green.csv`: share of 211 pixels whose **September** maximum NDVI is ≥ 0.45 in at least half of the
observed years 2016–2024 — sunflower is senescent/harvested by September, so late-season greenness inside the
nominally rainfed class points to irrigated summer crops (rice, maize, second crops).""")
code(r"""def year_fracs(y):
    y = ee.Number(y)
    start_s = ee.Date.fromYMD(y, SPRING[0], 1); end_s = ee.Date.fromYMD(y, SPRING[1], 1).advance(1, 'month')
    start_u = ee.Date.fromYMD(y, SUMMER[0], 1); end_u = ee.Date.fromYMD(y, SUMMER[1], 1).advance(1, 'month')
    spring = landsat.filterDate(start_s, end_s).select('NDVI').max()
    summer = landsat.filterDate(start_u, end_u).select('NDVI').max()
    base = m211[2018]
    observed = spring.mask().And(summer.mask()).And(base)
    d = spring.subtract(summer)
    wheat = d.gte(PH['delta']).And(spring.gte(PH['min_peak'])).And(observed)
    sun   = d.lte(-PH['delta']).And(summer.gte(PH['min_peak'])).And(observed)
    img = ee.Image.cat([ee.Image.pixelArea().updateMask(base).rename('a211'),
                        ee.Image.pixelArea().updateMask(observed).rename('observed'),
                        ee.Image.pixelArea().updateMask(wheat).rename('wheat'),
                        ee.Image.pixelArea().updateMask(sun).rename('sun')])
    return img.reduceRegions(districts, ee.Reducer.sum(), scale=100, tileScale=8).map(lambda f: f.set('year', y))

ph_fc = ee.FeatureCollection(ee.List.sequence(START_YEAR, END_YEAR).map(year_fracs)).flatten()
ee.batch.Export.table.toDrive(collection=ph_fc, description='rev_phenology_fractions', folder=DRIVE_FOLDER,
                              fileNamePrefix='rev_phenology_fractions', fileFormat='CSV',
                              selectors=['name','year','a211','observed','wheat','sun']).start()

SG_YEARS = list(range(2016, END_YEAR+1))
sep_imgs = [window_max(y, (9, 9)) for y in SG_YEARS]
sep_green_cnt = ee.ImageCollection([im.gte(0.45).unmask(0) for im in sep_imgs]).sum()
sep_obs_cnt   = ee.ImageCollection([im.mask() for im in sep_imgs]).sum()
sep_green = sep_green_cnt.gte(sep_obs_cnt.multiply(0.5)).And(sep_obs_cnt.gte(3)).And(m211[2018])
img = ee.Image.cat([ee.Image.pixelArea().updateMask(m211[2018]).rename('a211'),
                    ee.Image.pixelArea().updateMask(sep_obs_cnt.gte(3).And(m211[2018])).rename('observed'),
                    ee.Image.pixelArea().updateMask(sep_green).rename('september_green')])
ee.batch.Export.table.toDrive(collection=img.reduceRegions(districts, ee.Reducer.sum(), scale=100, tileScale=8),
                              description='rev_september_green', folder=DRIVE_FOLDER, fileNamePrefix='rev_september_green',
                              fileFormat='CSV', selectors=['name','a211','observed','september_green']).start()
print('two tasks started: rev_phenology_fractions, rev_september_green')""")

md(r"""## 6 · Per-scene district means under all masks  *(slow cell — whole record)* — v2

For each scene a multi-band image is built: NDVI and LST under `v1990 v2000 v2006 v2012 v2018 stable wc40`,
the year's `wheat`/`sun` phenology masks, plus unmasked (`none`). One `reduceRegions` per scene with a
**mean + count** reducer (count = valid 30 m pixels, needed for the SLC-off analysis).
**v2 fix:** the export lists every column explicitly (`selectors`) — without it GEE takes the schema from the first
feature, and a fully cloudy first scene silently drops all NDVI/LST columns (this is what happened in v1).
Geometry is not exported (files ~10× smaller).""")
code(r"""VARIANTS = list(MASKS.keys()) + ['wheat', 'sun', 'none']
PROPS = ['millis','name','sensor','slc_off','scene_id','wrs_path','wrs_row']
BAND_COLS = [f'{v}_{k}_{stat}' for k in VARIANTS for v in ('NDVI','LST') for stat in ('mean','count')]
SELECTORS = PROPS + BAND_COLS

def masked_bands(img, year_masks):
    bands = []
    for k, m in list(MASKS.items()) + list(year_masks.items()):
        bands.append(img.select('NDVI').updateMask(m).rename(f'NDVI_{k}'))
        bands.append(img.select('LST').updateMask(m).rename(f'LST_{k}'))
    bands.append(img.select('NDVI').rename('NDVI_none')); bands.append(img.select('LST').rename('LST_none'))
    return ee.Image.cat(bands)

REDUCER = ee.Reducer.mean().combine(ee.Reducer.count(), sharedInputs=True)

def year_fc(year, d0=None, d1=None):
    wheat, sun, _ = pheno_masks(year); ym = {'wheat': wheat, 'sun': sun}
    coll = landsat.filterDate(d0 or f'{year}-01-01', d1 or f'{year+1}-01-01')
    def samp(img):
        props = {'millis': img.get('system:time_start'), 'sensor': img.get('sensor'), 'slc_off': img.get('slc_off'),
                 'scene_id': img.get('scene_id'), 'wrs_path': img.get('wrs_path'), 'wrs_row': img.get('wrs_row')}
        return (masked_bands(img, ym).reduceRegions(collection=districts, reducer=REDUCER, scale=SCALE_M, tileScale=8)
                .map(lambda f: f.set(props)))
    return coll.map(samp).flatten()

def start_task(fc, name):
    t = ee.batch.Export.table.toDrive(collection=fc, description=name, folder=DRIVE_FOLDER, fileNamePrefix=name,
                                      fileFormat='CSV', selectors=SELECTORS)
    t.start(); return t

# years that failed as a single task are split into halves
SPLIT_YEARS = {2011}
tasks = {}
for y in range(START_YEAR, END_YEAR+1):
    try:
        if y in SPLIT_YEARS:
            tasks[f'{y}a'] = start_task(year_fc(y, f'{y}-01-01', f'{y}-07-01'), f'rev_scene_means_{y}a')
            tasks[f'{y}b'] = start_task(year_fc(y, f'{y}-07-01', f'{y+1}-01-01'), f'rev_scene_means_{y}b')
        else:
            tasks[str(y)] = start_task(year_fc(y), f'rev_scene_means_{y}')
        print(f'{y}: task started')
    except Exception as e:
        print(f'{y}: FAILED to start ({str(e)[:80]})')
print('\nMonitor in the Tasks tab; a FAILED year -> add it to SPLIT_YEARS and re-run this cell for that year only.')""")

md("### 6b · Monitor the tasks")
code(r"""while True:
    st = {k: t.status()['state'] for k, t in tasks.items()}
    print(time.strftime('%H:%M:%S'), {s_: list(st.values()).count(s_) for s_ in set(st.values())})
    if all(s_ in ('COMPLETED','FAILED','CANCELLED') for s_ in st.values()): break
    time.sleep(120)
print('Failed:', [k for k, t in tasks.items() if t.status()['state'] != 'COMPLETED'] or 'none')""")

md("## 7 · Sanity check + download")
code(r"""import glob
files = sorted(glob.glob('rev_scene_means_*.csv'))
if files:
    scenes = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    scenes['date'] = pd.to_datetime(scenes['millis'], unit='ms'); scenes['year']=scenes.date.dt.year; scenes['month']=scenes.date.dt.month
    print('rows', len(scenes), '| years', scenes.year.min(), '-', scenes.year.max(), '| sensors', scenes.sensor.value_counts().to_dict())
    # paper mask (v2018) monthly composite: NDVI max, LST mean — compare with notebook-01 output
    mo = scenes.groupby(['name','year','month']).agg(NDVI=('NDVI_v2018_mean','max'), LST=('LST_v2018_mean','mean')).reset_index()
    clim = mo.groupby(['name','month'])['NDVI'].mean().unstack()
    print('Amplitude (max-min) of monthly-mean NDVI by district, v2018 mask:'); print((clim.max(axis=1)-clim.min(axis=1)).round(2).to_string())
    scenes.to_csv('rev_scene_means_1985_2024.csv', index=False)
frac.to_csv('rev_district_landcover_fractions.csv', index=False)
# rev_phenology_fractions.csv and (if it ran as a task) rev_summer_green.csv are in Drive/thrace_revision
try:
    from google.colab import files as gfiles
    for f in ['rev_scene_means_1985_2024.csv','rev_district_landcover_fractions.csv','rev_phenology_fractions.csv']:
        if os.path.exists(f): gfiles.download(f)
except Exception:
    print('Not in Colab — download the CSVs manually.')""")

md(r"""## Notes
- **Timeouts in sync mode:** switch `MODE='drive'`; each year becomes one export task (≈ 5–20 min each, they run in parallel).
- **LGRIP30 / GFSAD1000 missing:** the table still works with the CORINE 212/213 and summer-green columns.
- **Phenology thresholds:** if §5b sunflower shares look far from TÜİK (≈ 0.3–0.4 of arable land on the plateau),
  adjust `PH` and re-run §5b only; §6 uses the same function, so re-run §6 after changing thresholds.
- Send the three CSVs back; the compositing, SLC-off, vintage, crop-specific and İpsala analyses are then run offline
  with the paper's own selection code (`revision_analysis.py` / `H2_gee_postprocess.py`).""")

nb['cells'] = C
nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
               "colab": {"provenance": []}}
nbf.write(nb, '09_gee_revision_extractions.ipynb')
print('written')
