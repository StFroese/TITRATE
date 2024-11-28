import astropy.units as u
from astropy import constants as const
from astropy.coordinates import SkyCoord
from gammapy.astro.darkmatter import JFactory, profiles
from gammapy.datasets import MapDataset
from gammapy.maps import WcsGeom, WcsNDMap

from titrate.upperlimits import ULFactory

if __name__ == "__main__":
    filename = "$GAMMAPY_DATA/cta-1dc-gc/cta-1dc-gc.fits.gz"
    dataset = MapDataset.read(filename, name="cta-dataset")

    def geometry2d():
        geom = WcsGeom.create(
            skydir=SkyCoord(0, 0, unit="deg", frame="galactic"),
            width=(4, 4),
            binsz=0.04,
            frame="galactic",
        )

        return geom

    def ursa_major_ii_profile():
        rhos = (
            10**-1.1331
            * const.M_sun.to(u.GeV, equivalencies=u.mass_energy())
            / u.pc**3
        )
        rs = 10**3.6317 * u.pc

        profile = profiles.NFWProfile(r_s=rs, rho_s=rhos)
        profile.DISTANCE_GC = 32 * u.kpc

        return profile

    jfactory = JFactory(
        geom=geometry2d(),
        profile=ursa_major_ii_profile(),
        distance=ursa_major_ii_profile().DISTANCE_GC,
    )
    jfactor = jfactory.compute_differential_jfactor()
    jfactor_map = WcsNDMap(geom=geometry2d(), data=jfactor.value, unit=jfactor.unit)

    ul_factory = ULFactory(
        dataset,
        ["b", "W", "tau", "mu"],
        0.1 * u.TeV,
        100 * u.TeV,
        20,
        jfactor_map,
        max_workers=8,
    )
    ul_factory.compute()
