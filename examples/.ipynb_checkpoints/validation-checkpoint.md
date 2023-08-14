---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.7
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

```python
from gammapy.datasets import MapDataset                                                                                                                                     
from gammapy.irf import load_cta_irfs                                                                                                                                       
from gammapy.maps import MapAxis, WcsGeom, WcsNDMap                                                                                                                         
from gammapy.modeling.models import (                                                                                                                                       
    FoVBackgroundModel,                                                                                                                                                     
    SkyModel,                                                                                                                                                               
    Models,                                                                                                                                                                 
    TemplateSpatialModel,                                                                                                                                                   
)                                                                                                                                                                           
from gammapy.data import Observation, observatory_locations                                                                                                                 
from gammapy.makers import MapDatasetMaker, SafeMaskMaker                                                                                                                   
from gammapy.astro.darkmatter import profiles, JFactory, DarkMatterAnnihilationSpectralModel                                                                                
from astropy.coordinates import SkyCoord                                                                                                                                    
import astropy.units as u                                                                                                                                                   
import numpy as np                                                                                                                                                          
from joblib import Parallel, delayed                                                                                                                                        
import matplotlib.pyplot as plt       
from scipy.stats import norm
from rich.progress import track
from scipy.interpolate import interp1d
from scipy.optimize import brentq

import astropy.constants as const
                                                                                                                                                                            
from pyphd.datasets import AsimovMapDataset                                                                                                                                 
from pyphd.tests import QMuTestStatistic, QTildeMuTestStatistic                                                                                                             
from pyphd.utils import calc_ts_toyMC, copy_dataset_with_models
```

# UpperLimits using the approximation

```python
# load IRFs                                                                                                                                                                 
irfs = load_cta_irfs(                                                                                                                                                       
    "$GAMMAPY_DATA/cta-1dc/caldb/data/cta/1dc/bcf/South_z20_50h/irf_file.fits"                                                                                              
)                                                                                                                                                                           
                                                                                                                                                                            
# observation                                                                                                                                                               
livetime = 50 * u.hr                                                                                                                                                        
pointing = SkyCoord(0, 0, unit="deg", frame="galactic")                                                                                                                     
location = observatory_locations["cta_south"]                                                                                                                               
obs = Observation.create(                                                                                                                                                   
    pointing=pointing, livetime=livetime, irfs=irfs, location=location                                                                                                      
)  
```

```python
# energy bins                                                                                                                                                               
energy_axis = MapAxis.from_energy_bounds("0.1 TeV", "100 TeV", nbin=10, per_decade=True)
energy_axis_true = MapAxis.from_energy_bounds(
    "0.03 TeV", "300 TeV", nbin=20, per_decade=True, name="energy_true"
)
migra_axis = MapAxis.from_bounds(0.5, 2, nbin=150, node_type="edges", name="migra")

geom = WcsGeom.create(
    skydir=pointing,
    width=(2, 2),
    binsz=0.02,
    frame="galactic",
    axes=[energy_axis],
)

empty_asimov = AsimovMapDataset.create(
    geom,
    energy_axis_true=energy_axis_true,
    migra_axis=migra_axis,
    name="asimov",
)
```

```python
# Ursa Major II
rhos = 10**-1.1331 * const.M_sun / u.pc ** 3
rhos = rhos / (1.8e-27 * u.kg) * u.GeV
rs = 10**3.6317 * u.pc
```

```python
profile = profiles.NFWProfile(r_s=rs, rho_s=rhos)
position = SkyCoord(0.0, 0.0, frame="galactic", unit="deg")
geom = WcsGeom.create(binsz=0.05, skydir=position, width=3.0, frame="galactic")
jfactory = JFactory(geom=geom, profile=profile, distance=32*u.kpc)
jfactor = jfactory.compute_differential_jfactor()
jfact_map = WcsNDMap(geom=geom, data=jfactor.value, unit=jfactor.unit)
spatial_model = TemplateSpatialModel(jfact_map, normalize=False)

spectral_model = DarkMatterAnnihilationSpectralModel(mass=50*u.TeV, channel='b')

model_simu = SkyModel(
    spatial_model=spatial_model,
    spectral_model=spectral_model,
    name="asimov-simu",
)

bkg_model = FoVBackgroundModel(dataset_name="asimov")
models = Models([model_simu, bkg_model])
```

```python
np.max(jfactor.value)
```

```python
jfactor.unit
```

```python
maker = MapDatasetMaker(selection=["exposure", "background", "psf", "edisp"])
maker_safe_mask = SafeMaskMaker(methods=["offset-max"], offset_max=4.0 * u.deg)

asimov_dataset = maker.run(empty_asimov, obs)
asimov_dataset = maker_safe_mask.run(asimov_dataset, obs)

asimov_dataset.models = models
```

```python
# create normal measurement dataset
energy_reco = MapAxis.from_edges(
    np.logspace(-1.0, 1.0, 10), unit="TeV", name="energy", interp="log"
)
geom = WcsGeom.create(
    skydir=pointing,
    binsz=0.02,
    width=(6, 6),
    frame="galactic",
    axes=[energy_reco],
)

empty = MapDataset.create(
    geom,
    energy_axis_true=energy_axis_true,
    name="measured",
)
measured_dataset = maker.run(empty, obs)
measured_dataset = maker_safe_mask.run(measured_dataset, obs)

model_simu = SkyModel(
    spatial_model=spatial_model,
    spectral_model=spectral_model,
    name="measured-simu",
)

bkg_model = FoVBackgroundModel(dataset_name="measured")

models = Models([model_simu, bkg_model])

measured_dataset.models = models
```

```python
measured_dataset.fake()
print(measured_dataset)
```

```python
ts_meas = QMuTestStatistic(measured_dataset, 'scale')
```

```python
np.array(interp_pvalues).ravel()
```

```python
poi_ul = 0
while ts_meas.pvalue(poi_ul) > 0.05:
    poi_ul += 1
    
interp_ul_points = np.linspace(poi_ul-1, poi_ul, 10)
interp_pvalues = np.array([ts_meas.pvalue(poi_ul) for poi_ul in interp_ul_points]).ravel()

interpolation = interp1d(interp_ul_points, interp_pvalues-0.05)
poi_ul = brentq(interpolation, poi_ul-1, poi_ul)
poi_ul
```

```python

```

```python
poi_ul * spectral_model.THERMAL_RELIC_CROSS_SECTION
```

```python
ts_meas.pvalue(poi_ul-0.1)
```

```python
asimov_dataset.models.parameters['scale'].value = 0
```

```python
asimov_dataset.fake()
```

```python
ts_asimov = QMuTestStatistic(asimov_dataset, 'scale')
```

```python
sigma = np.sqrt(asimov_dataset.models.covariance.data[0,0])
sigma
```

## Loop

```python
poi_uls = []
sigmas = []
masses = np.logspace(-1,2, 10)
for m in track(masses):
    spectral_model = DarkMatterAnnihilationSpectralModel(mass=m*u.TeV, channel='b')

    model_simu = SkyModel(
        spatial_model=spatial_model,
        spectral_model=spectral_model,
        name="measured-simu",
    )

    bkg_model = FoVBackgroundModel(dataset_name="measured")
    models = Models([model_simu, bkg_model])
    measured_dataset.models = models
    
    ts_meas = QMuTestStatistic(measured_dataset, 'scale')
    
    poi_ul = 1e-1
    pb = ts_meas.pvalue(0)
    while ts_meas.pvalue(poi_ul) / (1 - pb) > 0.05:
        poi_ul *= 10

    interp_ul_points = np.linspace(poi_ul/10, poi_ul, 10)
    interp_pvalues = np.array([ts_meas.pvalue(poi_ul) / (1 - pb) for poi_ul in interp_ul_points]).ravel()

    interpolation = interp1d(interp_ul_points, interp_pvalues-0.05)
    poi_ul = brentq(interpolation, poi_ul/10, poi_ul)
    
    poi_uls.append(poi_ul)
    
    model_simu = SkyModel(
        spatial_model=spatial_model,
        spectral_model=spectral_model,
        name="asimov-simu",
    )

    bkg_model = FoVBackgroundModel(dataset_name="asimov")
    models = Models([model_simu, bkg_model])
    asimov_dataset.models = models
    asimov_dataset.models.parameters['scale'].value = 0
    asimov_dataset.fake()
    
    ts_asimov = QMuTestStatistic(asimov_dataset, 'scale')
    sigma = np.sqrt(asimov_dataset.models.covariance.data[0,0])
    sigmas.append(sigma)
    print(poi_ul,sigma)
```

```python
masses
```

```python
poi_uls
```

```python
plt.plot(masses[:8], np.array(poi_uls)*spectral_model.THERMAL_RELIC_CROSS_SECTION.value)
plt.plot(masses[:8], np.array(sigmas) * norm.ppf(1-0.05*norm.cdf(0))*spectral_model.THERMAL_RELIC_CROSS_SECTION.value)
plt.fill_between(masses[:8], np.array(sigmas) * norm.ppf(1-0.05*norm.cdf(0))*spectral_model.THERMAL_RELIC_CROSS_SECTION.value, np.array(sigmas) * (norm.ppf(1-0.05*norm.cdf(2))+2)*spectral_model.THERMAL_RELIC_CROSS_SECTION.value, color='yellow')
plt.fill_between(masses[:8], np.array(sigmas) * norm.ppf(1-0.05*norm.cdf(0))*spectral_model.THERMAL_RELIC_CROSS_SECTION.value, np.array(sigmas) * (norm.ppf(1-0.05*norm.cdf(-2))-2)*spectral_model.THERMAL_RELIC_CROSS_SECTION.value, color='yellow')
plt.fill_between(masses[:8], np.array(sigmas) * norm.ppf(1-0.05*norm.cdf(0))*spectral_model.THERMAL_RELIC_CROSS_SECTION.value, np.array(sigmas) * (norm.ppf(1-0.05*norm.cdf(1))+1)*spectral_model.THERMAL_RELIC_CROSS_SECTION.value, color='lime')
plt.fill_between(masses[:8], np.array(sigmas) * norm.ppf(1-0.05*norm.cdf(0))*spectral_model.THERMAL_RELIC_CROSS_SECTION.value, np.array(sigmas) * (norm.ppf(1-0.05*norm.cdf(-1))-1)*spectral_model.THERMAL_RELIC_CROSS_SECTION.value, color='lime')
plt.hlines(spectral_model.THERMAL_RELIC_CROSS_SECTION.value, 1e-1, 2e1, linestyle='--', color='red')
plt.xscale('log')
plt.yscale('log')
```

```python
np.array(sigmas) * norm.ppf(0.95)
```

```python
np.array(sigmas) * (norm.ppf(0.95)-1)
```

```python
np.array(sigmas) * (norm.ppf(1-0.05*norm.cdf(-2))-2)
```

```python

```
