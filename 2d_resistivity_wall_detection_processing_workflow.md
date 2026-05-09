# 2D Apparent Resistivity Processing Workflow for Ancient Wall Detection

**Objective:** remove the regional/background resistivity field and enhance compact, linear, high-resistivity anomalies that may be caused by buried ancient walls or foundations.

**Coordinate system:** Pulkovo 1942 / Albania Zone 4  
**Typical input columns:** `Piketa`, `Profili`, `East`, `North`, `Rd`  
**Main target:** positive residual anomalies with linear/compact geometry.

---

## 1. Processing logic

This workflow follows the same logic used in the 3D resistivity-anomaly case, adapted to a 2D apparent-resistivity plan map.

In the 3D case, anomaly confidence can be tested by continuity between depth slices. In this 2D case, the equivalent controls are:

1. spatial continuity in plan view;
2. minimum anomaly area;
3. linear or compact geometry;
4. positive residual amplitude;
5. robust Z-score thresholding;
6. caution for anomalies near survey boundaries.

The complete procedure is:

1. Load the database.
2. Keep valid positive apparent-resistivity values.
3. Convert apparent resistivity to `log10(ρa)`.
4. Apply robust percentile clipping to reduce extreme outliers.
5. Interpolate the data to a regular grid.
6. Mask cells that are too far from measured points.
7. Apply a 3×3 median filter to remove spikes.
8. Estimate the background using Gaussian low-pass filtering.
9. Subtract the background to obtain the residual anomaly map.
10. Calculate robust Z-score using MAD.
11. Keep only positive high-resistivity anomalies.
12. Apply morphological closing and opening.
13. Remove small isolated components.
14. Extract anomaly components and estimate wall-candidate endpoints.
15. Export maps, processed grids, coordinate tables, and a report.

---

## 2. Mathematical basis

### 2.1 Logarithmic transform

Apparent resistivity is transformed to logarithmic scale:

```text
R(x,y) = log10(ρa(x,y))
```

This is useful because apparent resistivity commonly has a skewed distribution.

### 2.2 Robust clipping

The log-resistivity values are clipped between robust percentiles:

```text
Rclip = clip(R, P2, P98)
```

This reduces the influence of extreme isolated values.

### 2.3 Median despiking

A 3×3 median filter suppresses isolated spikes:

```text
Rmed = median_filter(Rclip, 3×3)
```

### 2.4 Background model

The regional background is estimated by Gaussian smoothing:

```text
Rbg = Gaussian_low_pass(Rmed, σ)
```

where the recommended value is:

```text
σ = 2.5 m
```

For a 0.5 m grid:

```text
σcells = 2.5 / 0.5 = 5 cells
```

### 2.5 Residual anomaly map

The residual anomaly map is:

```text
Rres = Rmed − Rbg
```

Positive values represent zones that are more resistive than the local background.

### 2.6 Robust Z-score

The robust Z-score is calculated using median absolute deviation:

```text
Zrobust = (Rres − median(Rres)) / MADnormal
```

where `MADnormal` is the MAD scaled to be comparable with standard deviation.

Recommended interpretation thresholds:

| Threshold | Meaning |
|---:|---|
| `Z ≥ 1.5` | moderate positive anomaly |
| `Z ≥ 2.0` | strong positive anomaly |
| `Z ≥ 2.5` | very strong positive anomaly |

The wall-candidate extraction in this script uses:

```text
Z > 2.0
```

---

## 3. Parameters used in the code

| Parameter | Value | Purpose |
|---|---:|---|
| Grid resolution | 0.5 m | regular interpolation grid |
| Percentile clipping | P2–P98 | robust outlier control |
| Median filter | 3×3 cells | despiking |
| Gaussian sigma | 2.5 m | regional-background removal |
| Maximum extrapolation distance | 2.6 m | avoid false edge interpolation |
| Robust Z threshold | 2.0 | strong positive anomalies |
| Minimum plan component size | 12 cells | remove isolated noise |
| Morphology kernel | 3×3 | close/open anomaly patches |

---

## 4. Required Python packages

Install the required libraries with:

```bash
pip install numpy pandas scipy scikit-learn matplotlib openpyxl xlrd
```

`xlrd` is needed for old `.xls` Excel files.  
`openpyxl` is needed for `.xlsx` Excel files.

---

## 5. Complete Python code

Save this as:

```text
apply_2d_resistivity_wall_workflow.py
```

Then edit the `INPUT` and `OUTDIR` variables inside the script, or modify the script to accept command-line arguments.

```python
import pandas as pd
import numpy as np
from scipy.interpolate import griddata
from scipy.spatial import cKDTree
from scipy.ndimage import median_filter, gaussian_filter, binary_closing, binary_opening, label
from scipy.stats import median_abs_deviation
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import ScalarFormatter, MultipleLocator
from matplotlib.patches import Rectangle
import zipfile, os

INPUT = '/mnt/data/Database Finale Enisi.xls'
OUTDIR = '/mnt/data/processed_2d_same_as_3d_workflow'
os.makedirs(OUTDIR, exist_ok=True)

# Parameters from the previous 3D workflow, adapted to 2D plan-map data
GRID_RESOLUTION_M = 0.5
CLIP_LOW_PCT = 2
CLIP_HIGH_PCT = 98
MEDIAN_SIZE_CELLS = 3
GAUSSIAN_SIGMA_M = 2.5
MAX_EXTRAPOLATION_DISTANCE_M = 2.6
ROBUST_Z_THRESHOLD = 2.0
MIN_PLAN_CELLS = 12
MORPH_STRUCTURE = np.ones((3,3), dtype=bool)

# Load and clean
raw = pd.read_excel(INPUT)
raw = raw.rename(columns={'Rd': 'rho_ohm_m'})
df = raw[['Piketa','Profili','East','North','rho_ohm_m']].copy()
df = df[df['rho_ohm_m'] > 0].copy()

# Local survey coordinates for filtering; final map is exported in Pulkovo coordinates.
x = df['Piketa'].to_numpy(float)
y = df['Profili'].to_numpy(float)
rho = df['rho_ohm_m'].to_numpy(float)
logrho = np.log10(rho)
p2, p98 = np.percentile(logrho, [CLIP_LOW_PCT, CLIP_HIGH_PCT])
logclip = np.clip(logrho, p2, p98)

xi = np.arange(np.floor(x.min()), np.ceil(x.max()) + GRID_RESOLUTION_M/2, GRID_RESOLUTION_M)
yi = np.arange(np.floor(y.min()), np.ceil(y.max()) + GRID_RESOLUTION_M/2, GRID_RESOLUTION_M)
X, Y = np.meshgrid(xi, yi)
points = np.column_stack([x,y])
lin = griddata(points, logclip, (X,Y), method='linear')
near = griddata(points, logclip, (X,Y), method='nearest')
G = np.where(np.isnan(lin), near, lin)
# mask extrapolated cells
tree = cKDTree(points)
dist, _ = tree.query(np.column_stack([X.ravel(), Y.ravel()]), k=1)
D = dist.reshape(X.shape)
valid_mask = D <= MAX_EXTRAPOLATION_DISTANCE_M
filled = np.where(valid_mask, G, near)

# Processing chain
median_log = median_filter(filled, size=MEDIAN_SIZE_CELLS, mode='nearest')
sigma_cells = GAUSSIAN_SIGMA_M / GRID_RESOLUTION_M
background_log = gaussian_filter(median_log, sigma=sigma_cells, mode='nearest')
residual_log = median_log - background_log
valid_residual = residual_log[valid_mask]
residual_median = np.nanmedian(valid_residual)
mad = median_abs_deviation(valid_residual, scale='normal', nan_policy='omit')
if not np.isfinite(mad) or mad == 0:
    mad = np.nanstd(valid_residual)
robust_z = (residual_log - residual_median) / mad
positive_mask_raw = (robust_z > ROBUST_Z_THRESHOLD) & valid_mask
positive_mask_clean = binary_opening(binary_closing(positive_mask_raw, structure=MORPH_STRUCTURE), structure=MORPH_STRUCTURE)
lbl, n_lbl = label(positive_mask_clean)

# Affine transformation from local survey axes to Pulkovo coordinates.
A = np.column_stack([np.ones(len(df)), x, y])
coefE = np.linalg.lstsq(A, df['East'].to_numpy(float), rcond=None)[0]
coefN = np.linalg.lstsq(A, df['North'].to_numpy(float), rcond=None)[0]
def transform(xv, yv):
    return coefE[0] + coefE[1]*xv + coefE[2]*yv, coefN[0] + coefN[1]*xv + coefN[2]*yv
E, N = transform(X, Y)

# Component extraction and line-trace approximation using PCA.
components = []
kept_label_ids = []
for cid in range(1, n_lbl + 1):
    inds = np.argwhere(lbl == cid)
    ncells = len(inds)
    if ncells < MIN_PLAN_CELLS:
        continue
    yy = Y[inds[:,0], inds[:,1]]
    xx = X[inds[:,0], inds[:,1]]
    ee, nn = transform(xx, yy)
    coords = np.column_stack([ee, nn])
    pca = PCA(n_components=2).fit(coords)
    proj = pca.transform(coords)[:,0]
    ep1 = coords[np.argmin(proj)]
    ep2 = coords[np.argmax(proj)]
    length_m = float(np.linalg.norm(ep2 - ep1))
    edge = bool((np.nanmin(xx) <= x.min()+MAX_EXTRAPOLATION_DISTANCE_M) or
                (np.nanmax(xx) >= x.max()-MAX_EXTRAPOLATION_DISTANCE_M) or
                (np.nanmin(yy) <= y.min()+MAX_EXTRAPOLATION_DISTANCE_M) or
                (np.nanmax(yy) >= y.max()-MAX_EXTRAPOLATION_DISTANCE_M))
    zvals = robust_z[lbl == cid]
    rvals = residual_log[lbl == cid]
    components.append({
        'component_id': cid,
        'cells': int(ncells),
        'area_m2': ncells * GRID_RESOLUTION_M**2,
        'start_E': float(ep1[0]),
        'start_N': float(ep1[1]),
        'end_E': float(ep2[0]),
        'end_N': float(ep2[1]),
        'length_m': length_m,
        'local_x_min': float(xx.min()),
        'local_x_max': float(xx.max()),
        'local_y_min': float(yy.min()),
        'local_y_max': float(yy.max()),
        'z_max': float(np.nanmax(zvals)),
        'z_mean': float(np.nanmean(zvals)),
        'residual_log10_max': float(np.nanmax(rvals)),
        'resistivity_ratio_max': float(10**np.nanmax(rvals)),
        'edge_caution': edge,
        'interpretation': 'edge-related resistive candidate' if edge else 'resistive wall/foundation candidate'
    })
    kept_label_ids.append(cid)
components_df = pd.DataFrame(components).sort_values(['z_max','area_m2'], ascending=False).reset_index(drop=True)
components_df['wall_label'] = [f'Wall {i+1}' for i in range(len(components_df))]
# Confidence classification
conf = []
for _, row in components_df.iterrows():
    if row['edge_caution']:
        c = 'High*' if row['z_max'] >= 3.0 else 'Medium*'
    elif row['z_max'] >= 3.0 and row['length_m'] >= 3.0:
        c = 'High'
    else:
        c = 'Medium'
    conf.append(c)
components_df['confidence'] = conf
components_df.to_csv(os.path.join(OUTDIR, 'wall_trace_summary_same_3d_workflow.csv'), index=False)

# Export processed grid table
out_grid = pd.DataFrame({
    'local_x_piketa_m': X.ravel(),
    'local_y_profili_m': Y.ravel(),
    'Easting_Pulkovo1942_Zone4_m': E.ravel(),
    'Northing_Pulkovo1942_Zone4_m': N.ravel(),
    'nearest_data_distance_m': D.ravel(),
    'inside_extrapolation_mask': valid_mask.ravel(),
    'log10_rho_interpolated_clipped': G.ravel(),
    'median_filtered_log10_rho': median_log.ravel(),
    'background_log10_rho': background_log.ravel(),
    'residual_log10_rho': residual_log.ravel(),
    'robust_z': robust_z.ravel(),
    'positive_mask_raw_z_gt_2': positive_mask_raw.ravel(),
    'positive_mask_clean': positive_mask_clean.ravel(),
    'component_id': lbl.ravel()
})
out_grid.to_csv(os.path.join(OUTDIR, 'processed_grid_same_3d_workflow.csv'), index=False)

# Helper functions for plots
def setup_axis(ax):
    fmt = ScalarFormatter(useOffset=False)
    fmt.set_scientific(False)
    ax.xaxis.set_major_formatter(fmt)
    ax.yaxis.set_major_formatter(fmt)
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.yaxis.set_major_locator(MultipleLocator(10))
    ax.xaxis.set_minor_locator(MultipleLocator(5))
    ax.yaxis.set_minor_locator(MultipleLocator(5))
    ax.grid(which='major', color='0.15', alpha=0.18, linewidth=0.5)
    ax.grid(which='minor', color='0.15', alpha=0.08, linewidth=0.35)
    ax.set_xlabel('Easting (m)')
    ax.set_ylabel('Northing (m)')
    ax.set_aspect('equal')

# Prepare masked arrays for plotting
plot_log = np.where(valid_mask, G, np.nan)
plot_median = np.where(valid_mask, median_log, np.nan)
plot_resid = np.where(valid_mask, residual_log, np.nan)
plot_z = np.where(valid_mask, robust_z, np.nan)

# Workflow panels
fig, axs = plt.subplots(2,2,figsize=(13,12),dpi=240,constrained_layout=True)
panel_data = [
    ('A. Original apparent resistivity, log10 clipped P2–P98', plot_log, 'viridis', None, 'log10 apparent resistivity'),
    ('B. 3×3 median-filtered map', plot_median, 'viridis', None, 'log10 apparent resistivity'),
    ('C. Residual anomaly map', plot_resid, 'RdYlBu_r', TwoSlopeNorm(vcenter=0, vmin=np.nanpercentile(plot_resid,2), vmax=np.nanpercentile(plot_resid,98)), 'Residual log10 apparent resistivity'),
    ('D. Robust Z-score anomaly map', plot_z, 'RdYlBu_r', TwoSlopeNorm(vcenter=0, vmin=-3.0, vmax=max(3.0, np.nanpercentile(plot_z,99))), 'Robust Z-score')
]
for ax, (title, data, cmap, norm, cblabel) in zip(axs.ravel(), panel_data):
    im = ax.pcolormesh(E,N,data,shading='auto',cmap=cmap,norm=norm)
    setup_axis(ax)
    ax.set_title(title, fontsize=11.5, weight='bold')
    if 'Residual' in title or 'Z-score' in title:
        cs = ax.contour(E,N,plot_z,levels=[1.5,2.0],colors='k',linewidths=[0.8,1.0],linestyles=['--','-'])
        ax.clabel(cs, fmt={1.5:'Z=1.5',2.0:'Z=2.0'}, fontsize=7)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label(cblabel, fontsize=8.5)
fig.suptitle('Ancient Wall Detection Workflow — Same Procedure as 3D Case\nPulkovo 1942 / Albania Zone 4', fontsize=15, weight='bold')
fig.savefig(os.path.join(OUTDIR, '01_workflow_maps_same_3d_procedure.png'), bbox_inches='tight')
plt.close(fig)

# Final interpreted map with top candidates labeled
fig, ax = plt.subplots(figsize=(10,10), dpi=260)
norm = TwoSlopeNorm(vcenter=0, vmin=np.nanpercentile(plot_resid,2), vmax=np.nanpercentile(plot_resid,98))
im = ax.pcolormesh(E,N,plot_resid,shading='auto',cmap='RdYlBu_r',norm=norm)
setup_axis(ax)
cs = ax.contour(E,N,plot_z,levels=[1.5,2.0],colors='black',linewidths=[1.0,1.25],linestyles=['--','-'])
ax.clabel(cs, fmt={1.5:'Z≥1.5', 2.0:'Z≥2.0'}, fontsize=8)
# mask contours
ax.contour(E,N,positive_mask_clean.astype(float),levels=[0.5],colors='magenta',linewidths=0.9)
# Label all components lightly; strongest 6 with wall labels
for idx, row in components_df.iterrows():
    xs = [row['start_E'], row['end_E']]
    ys = [row['start_N'], row['end_N']]
    color = 'deepskyblue' if row['edge_caution'] else ('cyan' if row['confidence']=='High' else 'lime')
    lw = 3.0 if idx < 6 else 1.5
    alpha = 0.95 if idx < 6 else 0.65
    ax.plot(xs, ys, color=color, lw=lw, alpha=alpha, solid_capstyle='round')
    mx, my = np.mean(xs), np.mean(ys)
    if idx < 6:
        labeltxt = f"{row['wall_label']} ({row['confidence']})"
        ax.text(mx+0.6, my+0.6, labeltxt, color=color, fontsize=8.5, weight='bold',
                bbox=dict(boxstyle='round,pad=0.2', fc='black', ec=color, alpha=0.72))
# scale bar and north arrow
xmin,xmax=ax.get_xlim(); ymin,ymax=ax.get_ylim()
bar_len=10; x0=xmin+3; y0=ymin+2
ax.plot([x0,x0+bar_len],[y0,y0],color='k',lw=4)
ax.plot([x0+bar_len/2,x0+bar_len/2],[y0-0.25,y0+0.25],color='white',lw=4,solid_capstyle='butt')
ax.text(x0+bar_len/2,y0+1.0,'10 m',ha='center',fontsize=8.5)
ax.annotate('N', xy=(xmax-6.5,ymin+12), xytext=(xmax-6.5,ymin+4), ha='center', va='bottom',
            arrowprops=dict(arrowstyle='-|>', lw=1.8, color='k'), fontsize=11, weight='bold')
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.025)
cb.set_label('Residual log10 apparent resistivity\npositive = resistive relative to background', fontsize=9)
ax.set_title('Final Wall-Candidate Map — 2D Adaptation of the 3D Workflow\nlog10 transform, P2–P98 clipping, σ=2.5 m background, robust Z>2.0', fontsize=12.5, weight='bold')
fig.savefig(os.path.join(OUTDIR, '02_final_wall_candidates_same_3d_procedure.png'), bbox_inches='tight')
plt.close(fig)

# A concise markdown report
report = f"""# 2D Apparent Resistivity Wall Detection — Same Processing Logic as 3D Case

Input file: `Database Finale Enisi.xls`  
Coordinate system: **Pulkovo 1942 / Albania Zone 4**  
Rows used after removing non-positive resistivity: **{len(df)}**

## Processing parameters

| Parameter | Value |
|---|---:|
| Working variable | log10(apparent resistivity) |
| Percentile clipping | P2 = {p2:.4f}, P98 = {p98:.4f} |
| Grid resolution | {GRID_RESOLUTION_M:.2f} m |
| Maximum extrapolation distance | {MAX_EXTRAPOLATION_DISTANCE_M:.2f} m |
| Median filter | {MEDIAN_SIZE_CELLS}×{MEDIAN_SIZE_CELLS} cells |
| Gaussian background sigma | {GAUSSIAN_SIGMA_M:.2f} m = {sigma_cells:.1f} grid cells |
| Residual | median-filtered log10(rho) − Gaussian background |
| Robust normalization | (residual − median) / MAD |
| Positive anomaly threshold | robust Z > {ROBUST_Z_THRESHOLD:.1f} |
| Morphology | 3×3 closing followed by 3×3 opening |
| Minimum component size | {MIN_PLAN_CELLS} plan cells = {MIN_PLAN_CELLS*GRID_RESOLUTION_M**2:.2f} m² |

## Results

- Raw positive cells before morphology: **{int(positive_mask_raw.sum())}**
- Clean positive cells after morphology: **{int(positive_mask_clean.sum())}**
- Kept wall/anomaly components: **{len(components_df)}**

The CSV file `wall_trace_summary_same_3d_workflow.csv` contains all extracted components with endpoint coordinates, length, area, maximum robust Z, and edge-caution flag.

## Interpretation rule

- **High**: strong positive residual, robust Z generally ≥ 3, and coherent linear/compact geometry.
- **High\***: same as High, but affected by edge uncertainty.
- **Medium**: coherent anomaly but weaker, shorter, or less persistent in plan view.

Because this is a single 2D plan map, the original 3D criterion of persistence across multiple depth slices cannot be applied. It is replaced here by 2D plan continuity, morphological cleaning, and minimum plan-area filtering.
"""
with open(os.path.join(OUTDIR, 'processing_report_same_3d_workflow.md'), 'w', encoding='utf-8') as f:
    f.write(report)

# Zip all deliverables
zip_path='/mnt/data/2d_resistivity_same_as_3d_workflow_outputs.zip'
with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
    for fn in os.listdir(OUTDIR):
        zf.write(os.path.join(OUTDIR, fn), arcname=fn)
    zf.write('/mnt/data/apply_2d_resistivity_wall_workflow.py', arcname='apply_2d_resistivity_wall_workflow.py')
print('DONE')
print(zip_path)
print(components_df.head(10).to_string(index=False))

```

---

## 6. Output files produced by the script

The script produces a processing folder containing:

| Output | Description |
|---|---|
| `01_workflow_maps_same_3d_procedure.png` | Original, median-filtered, residual, and robust Z-score maps |
| `02_final_wall_candidates_same_3d_procedure.png` | Final interpreted wall-candidate map |
| `wall_trace_summary_same_3d_workflow.csv` | Wall-candidate coordinate and geometry table |
| `processed_grid_same_3d_workflow.csv` | Full processed grid with residual and Z-score values |
| `processing_report_same_3d_workflow.md` | Short processing report |
| `2d_resistivity_same_as_3d_workflow_outputs.zip` | ZIP archive of all outputs |

---

## 7. Interpretation rules

A strong wall candidate should normally satisfy several conditions:

1. positive residual anomaly;
2. robust Z-score above 2.0;
3. compact or linear geometry;
4. continuity over several grid cells;
5. not only a single-cell spike;
6. not only an edge artifact;
7. consistent direction with archaeological/site geometry.

High-resistivity anomalies may be caused by:

- buried stone walls;
- foundations;
- compact masonry;
- rubble zones;
- dry soil or gravel;
- modern debris;
- shallow stones or blocks.

Therefore, the output should be described as a **wall-candidate map**, not as definitive proof of ancient walls.

---

## 8. How to adjust the workflow

### If the map is too noisy

Increase:

```python
ROBUST_Z_THRESHOLD = 2.5
MIN_PLAN_CELLS = 20
GAUSSIAN_SIGMA_M = 3.0
```

### If weak wall traces disappear

Decrease:

```python
ROBUST_Z_THRESHOLD = 1.8
MIN_PLAN_CELLS = 8
```

### If the background is not removed enough

Increase:

```python
GAUSSIAN_SIGMA_M = 3.5
```

### If wall anomalies are over-filtered

Decrease:

```python
GAUSSIAN_SIGMA_M = 2.0
```

### If boundary artifacts are too strong

Decrease:

```python
MAX_EXTRAPOLATION_DISTANCE_M = 1.5
```

---

## 9. Recommended report caption

```text
The 2D apparent resistivity plan-map data were processed using the same anomaly-enhancement logic applied in the 3D workflow. Apparent resistivity was transformed to log10 scale, clipped using robust percentiles, interpolated to a 0.5 m grid, despiked with a 3×3 median filter, and separated into background and residual components using Gaussian low-pass filtering. Positive residual anomalies were standardized using a robust MAD-based Z-score. Connected positive anomalies exceeding Z > 2.0 were morphologically cleaned, filtered by minimum plan area, and extracted as possible high-resistivity wall candidates. Candidate traces are reported in Pulkovo 1942 / Albania Zone 4 coordinates.
```

---

## 10. Important limitations

- A 2D plan map cannot confirm vertical continuity.
- Edge anomalies may be affected by interpolation and survey-boundary effects.
- High resistivity is not unique to archaeological walls.
- Confirmation should use excavation, GPR, magnetic data, or repeated resistivity surveying.
- The final result is best used as a guide for targeted archaeological verification.
