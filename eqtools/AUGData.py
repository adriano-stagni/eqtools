# This program is distributed under the terms of the GNU General Purpose License (GPL).
# Refer to http://www.gnu.org/licenses/gpl.txt
#
# This file is part of EqTools.
#
# EqTools is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# EqTools is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with EqTools.  If not, see <http://www.gnu.org/licenses/>.

"""This module provides classes inheriting :py:class:`eqtools.Equilibrium` for 
working with ASDEX Upgrade experimental data.
"""

import numpy
from scipy.interpolate import interp1d
from scipy.constants import mu_0

from .core import PropertyAccessMixin, ModuleWarning, Equilibrium, inPolygon
from joblib import Parallel, delayed
from tqdm import tqdm

import warnings

from collections import namedtuple

try:
    import aug_sfutils as sf

    _has_sf = True

except Exception as _e_sf:
    if isinstance(_e_sf, ImportError):
        warnings.warn(
            "aug_sfutils module could not be loaded -- classes that use "
            "aug_sfutils for data access will not work.",
            ModuleWarning,
        )
    else:
        warnings.warn(
            "aug_sfutils module could not be loaded -- classes that use "
            "aug_sfutils for data access will not work. Exception raised "
            "was of type %s, message was '%s'." % (_e_sf.__class__, _e_sf.message),
            ModuleWarning,
        )
    _has_sf = False

try:
    import dd
    from dd import PyddError

    _has_dd = True

except Exception as _e_dd:
    if isinstance(_e_dd, ImportError):
        warnings.warn(
            "dd module could not be loaded -- classes that use "
            "dd for data access will not work.",
            ModuleWarning,
        )
    else:
        warnings.warn(
            "dd module could not be loaded -- classes that use "
            "dd for data access will not work. Exception raised "
            "was of type %s, message was '%s'." % (_e_dd.__class__, _e_dd.message),
            ModuleWarning,
        )
    _has_dd = False

try:
    import matplotlib.pyplot as plt
    from contourpy import contour_generator as cntr

    _has_plt = True
except:
    warnings.warn(
        "Matplotlib.pyplot module could not be loaded -- classes that "
        "use pyplot will not work.",
        ModuleWarning,
    )
    _has_plt = False

class AUGSFData(Equilibrium):
    """Inherits :py:class:`eqtools.Equilibrium` class. Machine-specific data
    handling class for ASDEX Upgrade. Pulls AFS data from selected location
    and shotfile, stores as object attributes. Each data variable or set of
    variables is recovered with a corresponding getter method. Essential data
    for mapping are pulled on initialization (e.g. psirz grid). Additional
    data are pulled at the first request and stored for subsequent usage.
    
    Intializes ASDEX Upgrade version of the Equilibrium object.  Pulls data to 
    storage in instance attributes.  Core attributes are populated from the AFS 
    data on initialization.  Additional attributes are initialized as None, 
    filled on the first request to the object.

    Args:
        shot (integer): ASDEX Upgrade shot index.
    
    Keyword Args:
        shotfile (string): Optional input for alternate shotfile, defaults to 'EQH'
            (i.e., CLISTE results are in EQH,EQI with other reconstructions
            Available (FPP, EQE, ect.).
        edition (integer): Describes the edition of the shotfile to be used
        shotfile2 (string): Describes companion 0D equilibrium data, will automatically
            reference based off of shotfile, but can be manually specified for 
            unique reconstructions, etc.
        length_unit (string): Sets the base unit used for any quantity whose
            dimensions are length to any power. Valid options are:
                
                ===========  ===========================================================================================
                'm'          meters
                'cm'         centimeters
                'mm'         millimeters
                'in'         inches
                'ft'         feet
                'yd'         yards
                'smoot'      smoots
                'cubit'      cubits
                'hand'       hands
                'default'    whatever the default in the tree is (no conversion is performed, units may be inconsistent)
                ===========  ===========================================================================================
                
            Default is 'm' (all units taken and returned in meters).
        tspline (Boolean): Sets whether or not interpolation in time is
            performed using a tricubic spline or nearest-neighbor
            interpolation. Tricubic spline interpolation requires at least
            four complete equilibria at different times. It is also assumed
            that they are functionally correlated, and that parameters do
            not vary out of their boundaries (derivative = 0 boundary
            condition). Default is False (use nearest neighbor interpolation).
        monotonic (Boolean): Sets whether or not the "monotonic" form of time
            window finding is used. If True, the timebase must be monotonically
            increasing. Default is False (use slower, safer method).
        experiment: Used to describe the work space that the shotfile is located
            It defaults to 'AUGD' but can be set to other values
    """

    # its like relating g files to a files
    _relatedSVFile = {"EQI": "GQI", "EQH": "GQH", "EQE": "GQE", "FPP": "GPI"}

    def __init__(
        self,
        shot,
        shotfile="EQH",
        edition=0,
        shotfile2=None,
        length_unit="m",
        tspline=False,
        monotonic=True,
        experiment="AUGD",
    ):

        if not _has_sf:
            print("aug_sfutils module did not load properly")
            print("Most functionality will not be available!")

        super(AUGSFData, self).__init__(
            length_unit=length_unit, tspline=tspline, monotonic=monotonic
        )

        self._shot = shot
        self._tree = shotfile
        print(self._shot, self._tree, edition, experiment)
        self._shotFile = sf.SFREAD(
            self._shot, self._tree, ed=edition, exp=experiment
        )
        self._equ = sf.EQU(
            self._shot, diag=self._tree, eq=edition, exp=experiment
        )

        try:
            if shotfile2 is None:
                shotfile2 = self._relatedSVFile[self._tree]

            # Overwrite getSSQ with a shotfile with same capabilities
            self.getSSQ = sf.SFREAD(
                self._shot, shotfile2, ed=edition, exp=experiment
            )
        except (KeyError, PyddError):
            warnings.warn(
                "Companion SV not valid, extracting from " + self._tree + ":SSQ",
                RuntimeWarning,
            )

        self._defaultUnits = {}

        # initialize None for non-essential data

        # grad-shafranov related parameters
        self._fpol = None
        self._fluxPres = None  # pressure on flux surface (psi,t)
        self._ffprim = None
        self._pprime = None  # pressure derivative on flux surface (t,psi)

        # fields
        self._btaxp = None  # Bt on-axis, with plasma (t)
        self._btaxv = None  # Bt on-axis, vacuum (t)
        self._bpolav = None  # avg poloidal field (t)
        self._BCentr = None  # Bt at RCentr, vacuum (for gfiles) (t)

        # plasma current
        self._IpCalc = None  # calculated plasma current (t)
        self._IpMeas = None  # measured plasma current (t)
        self._Jp = None  # grid of current density (r,z,t)
        self._currentSign = None  # sign of current for entire shot (calculated in moderately kludgey manner)

        # safety factor parameters
        self._q0 = None  # q on-axis (t)
        self._q95 = None  # q at 95% flux (t)
        self._qLCFS = None  # q at LCFS (t)
        self._rq1 = None  # outboard-midplane minor radius of q=1 surface (t)
        self._rq2 = None  # outboard-midplane minor radius of q=2 surface (t)
        self._rq3 = None  # outboard-midplane minor radius of q=3 surface (t)

        # shaping parameters
        self._kappa = None  # LCFS elongation (t)
        self._dupper = None  # LCFS upper triangularity (t)
        self._dlower = None  # LCFS lower triangularity (t)

        # (dimensional) geometry parameters
        self._rmag = None  # major radius, magnetic axis (t)
        self._zmag = None  # Z magnetic axis (t)
        self._aLCFS = None  # outboard-midplane minor radius (t)
        self._RmidLCFS = None  # outboard-midplane major radius (t)
        self._areaLCFS = None  # LCFS surface area (t)
        self._RLCFS = None  # R-positions of LCFS (t,n)
        self._ZLCFS = None  # Z-positions of LCFS (t,n)
        self._RCentr = None  # Radius for BCentr calculation (for gfiles) (t)

        # machine geometry parameters
        self._Rlimiter = None  # R-positions of vacuum-vessel wall (t)
        self._Zlimiter = None  # Z-positions of vacuum-vessel wall (t)

        # calc. normalized-pressure values
        self._betat = None  # calc toroidal beta (t)
        self._betap = None  # calc avg. poloidal beta (t)
        self._Li = None  # calc internal inductance (t)

        # diamagnetic measurements
        self._diamag = None  # diamagnetic flux (t)
        self._betatd = None  # diamagnetic toroidal beta (t)
        self._betapd = None  # diamagnetic poloidal beta (t)
        self._WDiamag = None  # diamagnetic stored energy (t)
        self._tauDiamag = None  # diamagnetic energy confinement time (t)

        # energy calculations
        self._WMHD = None  # calc stored energy (t)
        self._tauMHD = None  # calc energy confinement time (t)
        self._Pinj = None  # calc injected power (t)
        self._Wbdot = None  # d/dt magnetic stored energy (t)
        self._Wpdot = None  # d/dt plasma stored energy (t)

        # load essential mapping data
        # Set the variables to None first so the loading calls will work right:
        self._time = None  # timebase
        self._psiRZ = None  # flux grid (r,z,t)
        self._rGrid = None  # R-axis (t)
        self._zGrid = None  # Z-axis (t)
        self._psiLCFS = None  # flux at LCFS (t)
        self._psiAxis = None  # flux at magnetic axis (t)
        self._psiLabel = None # poloidal flux label (t,psi)
        self._psiNLabel = None # normalized poloidal flux label (t,psi)
        self._fluxVol = None  # volume within flux surface (t,psi)
        self._volLCFS = None  # volume within LCFS (t)
        self._qpsi = None  # q profile (psi,t)
        self._RmidPsi = None  # max major radius of flux surface (t,psi)
        self._xlow = None # lower x-point coordinates (t,2)
        self._xup = None # upper x-point coordinates (t,2)

        # AUG SV file flag
        self._SSQ = None

        # Call the get functions to preload the data. Add any other calls you
        # want to preload here.
        self.getTimeBase()  # check
        self._timeidxend = self.getTimeBase().size
        self.getFluxGrid()  # loads _psiRZ, _rGrid and _zGrid at once. check
        self.getFluxLCFS()  # check
        self.getFluxAxis()  # check
        self.getFluxLabel() # check
        self.getNormFluxLabel() # check
        self.getFluxVol()  # check
        self.getVolLCFS()  # check
        self.getQProfile()  #
        self._ygcauginterface()  # needed to initialize Vessel properties
        self.getBtVac()
        self.remapLCFS()

    def __str__(self):
        """string formatting for ASDEX Upgrade Equilibrium class.
        """
        try:
            nt = len(self._time)
            nr = len(self._rGrid)
            nz = len(self._zGrid)

            mes = (
                "AUG data for shot "
                + str(self._shot)
                + " from shotfile "
                + str(self._tree.upper())
                + "\n"
                + "timebase "
                + str(self._time[0])
                + "-"
                + str(self._time[-1])
                + "s in "
                + str(nt)
                + " points\n"
                + str(nr)
                + "x"
                + str(nz)
                + " spatial grid"
            )
            return mes
        except TypeError:
            return "tree has failed data load."
        
    def getInfo(self):
        """returns namedtuple of shot information
        
        Returns:
            namedtuple containing
                
                =====   ===============================
                shot    ASDEX Upgrage shot index (long)
                tree    shotfile (string)
                nr      size of R-axis for spatial grid
                nz      size of Z-axis for spatial grid
                nt      size of timebase for flux grid
                =====   ===============================
        """
        try:
            nt = len(self._time)
            nr = len(self._rGrid)
            nz = len(self._zGrid)
        except TypeError:
            nt, nr, nz = 0, 0, 0
            print("tree has failed data load.")

        data = namedtuple("Info", ["shot", "tree", "nr", "nz", "nt"])
        return data(shot=self._shot, tree=self._tree, nr=nr, nz=nz, nt=nt)
    
    def getTimeBase(self):
        """returns time base vector.

        Returns:
            time (array): [nt] array of time points.

        Raises:
            ValueError: if module cannot retrieve data from the AUG AFS system.
        """
        if self._time is None:
            try:
                self._time = self._shotFile.getobject("time").copy()
                self._defaultUnits["_time"] = self._time.phys_unit
            except:
                raise ValueError("data retrieval failed.")
        return self._time.copy()
    
    def getFluxGrid(self):
        """returns flux grid.
        
        Note that this method preserves whatever sign convention is used in AFS.
        
        Returns:
            psiRZ (Array): [nt,nz,nr] array of (non-normalized) flux on grid.

        Raises:
            ValueError: if module cannot retrieve data from the AUG AFS system.
        """
        if self._psiRZ is None:
            try:
                self._psiRZ = self._equ.pfm.transpose((2,1,0)) / (2* numpy.pi) # Correct for a factor 2*pi (verified with Bp values)
                self._defaultUnits["_psiRZ"] = self._psiRZ.phys_unit  # HARDCODED DUE TO CALIBRATED=FALSE
                self._rGrid = self._equ.Rmesh.copy()
                self._defaultUnits["_rGrid"] = self._rGrid.phys_unit
                self._zGrid = self._equ.Zmesh.copy()
                self._defaultUnits["_zGrid"] = self._zGrid.phys_unit
            except:
                raise ValueError("data retrieval failed.")
        return self._psiRZ.copy()
    
    def getRGrid(self, length_unit=1):
        """returns R-axis.

        Returns:
            rGrid (Array): [nr] array of R-axis of flux grid.

        Raises:
            ValueError: if module cannot retrieve data from the AUG AFS system.
        """
        if self._rGrid is None:
            raise ValueError("data retrieval failed.")

        # Default units should be 'm'
        unit_factor = self._getLengthConversionFactor(
            self._defaultUnits["_rGrid"], length_unit
        )
        return unit_factor * self._rGrid.copy()
    
    def getZGrid(self, length_unit=1):
        """returns Z-axis.

        Returns:
            zGrid (Array): [nz] array of Z-axis of flux grid.

        Raises:
            ValueError: if module cannot retrieve data from the AUG AFS system.
        """
        if self._zGrid is None:
            raise ValueError("data retrieval failed.")

        # Default units should be 'm'
        unit_factor = self._getLengthConversionFactor(
            self._defaultUnits["_zGrid"], length_unit
        )
        return unit_factor * self._zGrid.copy()
    
    def getFluxAxis(self):
        """returns psi on magnetic axis.

        Returns:
            psiAxis (Array): [nt] array of psi on magnetic axis.

        Raises:
            ValueError: if module cannot retrieve data from the AUG AFS system.
        """
        if self._psiAxis is None:
            try:
                self._psiAxis = self._equ.psi0.copy() / (2 * numpy.pi)
                self._defaultUnits["_psiAxis"] = self._equ.psi0.phys_unit
            except:
                raise ValueError("data retrieval failed.")
        return self._psiAxis.copy()
    
    def getFluxLCFS(self):
        """returns psi at separatrix.

        Returns:
            psiLCFS (Array): [nt] array of psi at LCFS.

        Raises:
            ValueError: if module cannot retrieve data from the AUG AFS system.
        """
        if self._psiLCFS is None:
            try:
                self._psiLCFS = self._equ.psi_lcfs.copy() / (2 * numpy.pi)
                self._defaultUnits["_psiLCFS"] = self._equ.psi_lcfs.phys_unit
            except PyddError:
                raise ValueError("data retrieval failed.")
        return self._psiLCFS.copy()
    
    def getFluxLabel(self):
        """returns time-dependent poloidal flux label array.
        
        Returns:
            psiLabel (Array) [nt,npfl]: time-dependent array of poloidal flux labels
        
        Raises:
            ValueError: if module cannot retrieve data from the AUG shotfile system.
        """
        if self._psiLabel is None:
            try:
                self._psiLabel = self._equ.pfl.copy() / (2 * numpy.pi)
                self._defaultUnits["_psiLabel"] = self._equ.pfl.phys_unit
            except:
                raise ValueError("data retrieval failed.")
        return self._psiLabel.copy()
    
    def getNormFluxLabel(self):
        """returns time-dependent normalized poloidal flux label array.
        
        Returns:
            psiNLabel (Array) [nt,npfl]: time-dependent array of normalized poloidal flux labels
        
        Raises:
            ValueError: if module cannot retrieve data from the AUG shotfile system.
        """
        if self._psiNLabel is None:
            try:
                self._psiNLabel = self._equ.psiN.copy()
                self._defaultUnits["_psiNLabel"] = '' # The version stored in aug_sfutils.EQU uses 'Vs' as units, which is obviously wrong for a normalized quantity
            except:
                raise ValueError("data retrieval failed.")
        return self._psiNLabel.copy()

    def getFluxVol(self, length_unit=3):
        """returns volume within flux surface.

        Keyword Args:
            length_unit (String or 3): unit for plasma volume.  Defaults to 3, 
                indicating default volumetric unit (typically m^3).

        Returns:
            fluxVol (Array): [nt,npsi] array of volume within flux surface.

        Raises:
            ValueError: if module cannot retrieve data from the AUG AFS system.
        """
        if self._fluxVol is None:
            try:
                self._fluxVol = self._equ.vol.copy()
                self._defaultUnits["_fluxVol"] = self._equ.vol.phys_unit
            except:
                raise ValueError("data retrieval failed.")
        unit_factor = self._getLengthConversionFactor(
            self._defaultUnits["_fluxVol"], length_unit
        )
        return self._fluxVol.copy()

    def getVolLCFS(self, length_unit=3):
        """returns volume within LCFS.

        Keyword Args:
            length_unit (String or 3): unit for LCFS volume.  Defaults to 3, 
                denoting default volumetric unit (typically m^3).

        Returns:
            volLCFS (Array): [nt] array of volume within LCFS.

        Raises:
            ValueError: if module cannot retrieve data from the AUG AFS system.
        """
        if self._volLCFS is None:
            try:
                self._volLCFS = self._equ.Vol.copy()
                self._defaultUnits["_volLCFS"] = 'm^3' # Units not stored in the SFOBJ onstance
            except:
                raise ValueError("data retrieval failed.")
        # Default units should be 'cm^3':
        unit_factor = self._getLengthConversionFactor(
            self._defaultUnits["_volLCFS"], length_unit
        )
        return unit_factor * self._volLCFS.copy()
    
    def getRmidPsi(self, length_unit=1):
        """returns maximum major radius of each flux surface.

        Keyword Args:
            length_unit (String or 1): unit of Rmid.  Defaults to 1, indicating 
                the default parameter unit (typically m).

        Returns:
            Rmid (Array): [nt,npsi] array of maximum (outboard) major radius of 
            flux surface psi.
       
        Raises:
            NotImplementedError: Not implemented on ASDEX-Upgrade reconstructions.
        """
        if self._RmidPsi is None:
            try:
                psiLevels = numpy.asarray(self.getFluxLabel())
                psiRZ = numpy.asarray(self.getFluxGrid())  # [nt,nZ,nR]
                R = numpy.asarray(self.getRGrid())
                Z = numpy.asarray(self.getZGrid())
                Rmag, Zmag = numpy.asarray(self.getMagR()), numpy.asarray(self.getMagZ())
                RZxpo, RZxpu = numpy.asarray(self.getUpperXpoint()), numpy.asarray(self.getLowerXpoint())

                def process_timestep(args):
                    it, R, Z, psiRZ_it, psiLevels_it, Rmag_it, Zmag_it, RZxpo_it, RZxpu_it = args
                    from contourpy import contour_generator as cntr # Import contourpy inside the function to ensure it's available in each process
                    levels = psiLevels_it # Get levels for this timestep
                    results = numpy.full(len(levels), numpy.nan)
                    cg = cntr(x=R, y=Z, z=psiRZ_it, name='serial') # Create contour generator for this timestep
                    cs = numpy.array(
                        [min(
                            cg.lines(level),
                            key=lambda c: numpy.hypot(c[:,0] - Rmag_it, c[:,1] - Zmag_it).min(),
                            default=numpy.array([[numpy.nan, numpy.nan]]))
                         for level in levels], dtype='object')
                    if numpy.sign(RZxpo_it[1]/RZxpu_it[1]) == -1:
                        mask = (cs[-1][:,1] > RZxpu_it[1]) & (cs[-1][:,1] < RZxpo_it[1])
                        cs[-1] = cs[-1][mask,:]
                        if not numpy.allclose(cs[-1][0,:], cs[-1][-1,:]):
                            cs[-1] = numpy.vstack((cs[-1], cs[-1][0,:]))
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        results = numpy.vectorize(lambda c: numpy.nanmax(c[:,0]))(cs)
                    return it, results
                
                nt = psiLevels.shape[1]
                n_workers = 16
                _RmidPsi = numpy.tile(numpy.nan, psiLevels.shape)
                results = Parallel(n_jobs=n_workers)(
                    delayed(process_timestep)(
                        (it, R, Z, psiRZ[it], psiLevels[it,:], Rmag[it], Zmag[it], RZxpo[it,:], RZxpu[it,:])
                    ) for it in tqdm(range(psiLevels.shape[0]), desc="Processing equilibrium times (RmidPsi)", ascii='-##')
                )

                for it, r_values in results:
                    _RmidPsi[it,:] = r_values

                self._RmidPsi = _RmidPsi.copy()
                self._defaultUnits["_RmidPsi"] = "m"
            except:
                raise ValueError("data retrieval failed.")
        unit_factor = self._getLengthConversionFactor(
            self._defaultUnits["_RmidPsi"], length_unit
        )
        return unit_factor * self._RmidPsi.copy()

    def getRLCFS(self, length_unit=1):
        """returns R-values of LCFS position.

        Returns:
            RLCFS (Array): [nt,n] array of R of LCFS points.

        Raises:
            ValueError: if module cannot retrieve data from the AUG AFS system.
        """
        if self._RLCFS is None:
            try:
                rgeo = self._equ.Rgeo.copy()
                ray_names = [r for r in self._equ.ssqnames if r.startswith("ray__")]
                rays = numpy.array([getattr(self._equ, r) for r in ray_names]).T
                RLCFStemp = numpy.hstack(
                    (numpy.atleast_2d(rays[:,-1]).T, rays)
                )
                templen = rays.shape

                self._RLCFS = numpy.tile(
                    rgeo.data, (templen[1] + 1, 1)
                ).T + RLCFStemp * numpy.cos(
                    numpy.tile(
                        (numpy.linspace(0, 2 * numpy.pi, templen[1] + 1)),
                        (templen[0], 1),
                    )
                )  # construct a 2d grid of angles, take cos, multiply by radius
                self._defaultUnits["_RLCFS"] = 'm'
            except KeyError:
                self.remapLCFS()
                self._defaultUnits["_RLCFS"] = str("m")
                self._defaultUnits["_ZLCFS"] = str("m")
            except:
                raise ValueError("data retrieval failed.")
        unit_factor = self._getLengthConversionFactor(
            self._defaultUnits["_RLCFS"], length_unit
        )
        return unit_factor * self._RLCFS.copy()

    def getZLCFS(self, length_unit=1):
        """returns Z-values of LCFS position.

        Returns:
            ZLCFS (Array): [nt,n] array of Z of LCFS points.

        Raises:
            ValueError: if module cannot retrieve data from the AUG AFS system.
        """
        if self._ZLCFS is None:
            try:
                zgeo = self.getSSQ("Zgeo")
                ray_names = [r for r in self._equ.ssqnames if r.startswith("ray__")]
                rays = numpy.array([getattr(self._equ, r) for r in ray_names]).T
                ZLCFStemp = numpy.hstack(
                    (numpy.atleast_2d(rays[:,-1]).T, rays)
                )
                templen = rays.shape

                self._ZLCFS = numpy.tile(
                    zgeo.data, (templen[1] + 1, 1)
                ).T + ZLCFStemp * numpy.sin(
                    numpy.tile(
                        (numpy.linspace(0, 2 * numpy.pi, templen[1] + 1)),
                        (templen[0], 1),
                    )
                )  # construct a 2d grid of angles, take sin, multiply by radius
                self._defaultUnits["_ZLCFS"] = 'm'
            except KeyError:
                self.remapLCFS()
                self._defaultUnits["_RLCFS"] = str("m")
                self._defaultUnits["_ZLCFS"] = str("m")
            except PyddError:
                raise ValueError("data retrieval failed.")
        unit_factor = self._getLengthConversionFactor(
            self._defaultUnits["_ZLCFS"], length_unit
        )
        return unit_factor * self._ZLCFS.copy()

    def remapLCFS(self, mask=False):
        """Overwrites RLCFS, ZLCFS values pulled with explicitly-calculated 
        contour of psinorm=1 surface.  This is then masked down by the limiter
        array using core.inPolygon, restricting the contour to the closed
        plasma surface and the divertor legs.

        Keyword Args:
            mask (Boolean): Default False.  Set True to mask LCFS path to 
                limiter outline (using inPolygon).  Set False to draw full 
                contour of psi = psiLCFS.

        Raises:
            NotImplementedError: if :py:mod:`matplotlib.pyplot` is not loaded.
            ValueError: if limiter outline is not available.
        """
        if not _has_plt:
            raise NotImplementedError(
                "Requires matplotlib.pyplot for contour calculation."
            )

        try:
            Rlim, Zlim = self.getMachineCrossSection()
        except:
            raise ValueError(
                "Limiter outline (self.getMachineCrossSection) must be available."
            )

        # plt.ioff() # Obsolete, given the change from matplotlib._cntr to skimage.measure.find_contours

        psiRZ = self.getFluxGrid()  # [nt,nZ,nR]
        R = self.getRGrid()
        Z = self.getZGrid()
        psiLCFS = self.getFluxLCFS()

        RLCFS_stores = []
        ZLCFS_stores = []
        maxlen = 0
        nt = len(psiRZ)
        #        fig = plt.figure()
        for i in range(nt):
            cs = cntr(x=R, y=Z, z=psiRZ[i], name='serial').lines(psiLCFS[i])
            RLCFS_frame = []
            ZLCFS_frame = []
            for v in cs:
                RLCFS_frame.extend(v[:, 0])
                ZLCFS_frame.extend(v[:, 1])
                RLCFS_frame.append(numpy.nan)
                ZLCFS_frame.append(numpy.nan)
            RLCFS_frame = numpy.array(RLCFS_frame)
            ZLCFS_frame = numpy.array(ZLCFS_frame)

            # generate masking array to vessel
            if mask:
                maskarr = numpy.array([False for i in range(len(RLCFS_frame))])
                for i, x in enumerate(RLCFS_frame):
                    y = ZLCFS_frame[i]
                    maskarr[i] = inPolygon(Rlim, Zlim, x, y)

                RLCFS_frame = RLCFS_frame[maskarr]
                ZLCFS_frame = ZLCFS_frame[maskarr]

            if len(RLCFS_frame) > maxlen:
                maxlen = len(RLCFS_frame)
            RLCFS_stores.append(RLCFS_frame)
            ZLCFS_stores.append(ZLCFS_frame)

        RLCFS = numpy.zeros((nt, maxlen))
        ZLCFS = numpy.zeros((nt, maxlen))
        for i in range(nt):
            RLCFS_frame = RLCFS_stores[i]
            ZLCFS_frame = ZLCFS_stores[i]
            ni = len(RLCFS_frame)
            RLCFS[i, 0:ni] = RLCFS_frame
            ZLCFS[i, 0:ni] = ZLCFS_frame

        # store final values
        self._RLCFS = RLCFS
        self._ZLCFS = ZLCFS

        # set default unit parameters, based on RZ grid
        rUnit = self._defaultUnits["_rGrid"]
        zUnit = self._defaultUnits["_zGrid"]
        self._defaultUnits["_RLCFS"] = rUnit
        self._defaultUnits["_ZLCFS"] = zUnit

    def getF(self):
        """returns F=RB_{\Phi}(\Psi), often calculated for grad-shafranov 
        solutions.
        
        Returns:
            F (Array): [nt,npsi] array of F=RB_{\Phi}(\Psi)

        Raises:
            ValueError: if module cannot retrieve data from the AUG AFS system.
        """
        if self._fpol is None:
            try:
                self._fpol = self._equ.jpol.copy() * 2e-7
                self._defaultUnits["_fpol"] = str("Tm")
            except:
                raise ValueError("data retrieval failed.")
        return self._fpol.copy()

    def getFluxPres(self):
        """returns pressure at flux surface.

        Returns:
            p (Array): [nt,npsi] array of pressure on flux surface psi.

        Raises:
            ValueError: if module cannot retrieve data from AUG AFS system.
        """
        if self._fluxPres is None:
            try:
                self._fluxPres = self._equ.pres.copy()
                self._defaultUnits["_fluxPres"] = self._equ.pres.phys_unit
            except:
                raise ValueError("data retrieval failed.")
        return self._fluxPres.copy()

    def getFPrime(self):
        """returns F', often calculated for grad-shafranov 
        solutions.

        Returns:
            F (Array): [nt,npsi] array of F=RB_{\Phi}(\Psi)

        Raises:
            ValueError: if module cannot retrieve data from the AUG AFS system.
        """
        if self._fprime is None:
            try:
                self._fprime = self._equ.djpol.copy() * 2e-7
                self._defaultUnits["_fpol"] = str("Tm")
            except:
                raise ValueError("data retrieval failed.")
        return self._fprime.copy()

    def getFFPrime(self):
        """returns FF' function used for grad-shafranov solutions.

        Returns:
            FFprime (Array): [nt,npsi] array of FF' fromgrad-shafranov solution.

        Raises:
            ValueError: if module cannot retrieve data from the AUG AFS system.
        """
        if self._ffprim is None:
            try:
                self._ffprim = self._equ.ffp.copy()
                self._defaultUnits["_ffprim"] = self._equ.ffp.phys_unit
            except:
                raise ValueError("data retrieval failed.")
        return self._ffprim.copy()

    def getPPrime(self):
        """returns plasma pressure gradient as a function of psi.

        Returns:
            pprime (Array): [nt,npsi] array of pressure gradient on flux surface 
            psi from grad-shafranov solution.

        Raises:
            ValueError: if module cannot retrieve data from the AUG AFS system.
        """
        if self._pprime is None:
            try:
                self._pprime = self._equ.dpres.copy()
                self._defaultUnits["_pprime"] = self._equ.dpres.phys_unit
            except:
                raise ValueError("data retrieval failed.")
        return self._pprime.copy()

    def getElongation(self):
        """returns LCFS elongation.

        Returns:
            kappa (Array): [nt] array of LCFS elongation.

        Raises:
            ValueError: if module cannot retrieve data from AFS.
        """
        if self._kappa is None:
            try:
                self._kappa = self._equ.k.copy()
                self._defaultUnits["_kappa"] = self._equ.k.phys_unit
            except:
                raise ValueError("data retrieval failed.")
        return self._kappa.copy()

    def getUpperTriangularity(self):
        """returns LCFS upper triangularity.

        Returns:
            deltau (Array): [nt] array of LCFS upper triangularity.

        Raises:
            ValueError: if module cannot retrieve data from the AUG AFS system.
        """
        if self._dupper is None:
            try:
                self._dupper = self._equ.delRoben.copy()
                self._defaultUnits["_dupper"] = self._equ.delRoben.phys_unit
            except:
                raise ValueError("data retrieval failed.")
        return self._dupper.copy()

    def getLowerTriangularity(self):
        """returns LCFS lower triangularity.

        Returns:
            deltal (Array): [nt] array of LCFS lower triangularity.

        Raises:
            ValueError: if module cannot retrieve data from the AUG AFS system.
        """
        if self._dlower is None:
            try:
                self._dlower = self._equ.delRuntn.copy()
                self._defaultUnits["_dlower"] = self._equ.delRuntn.phys_unit
            except:
                raise ValueError("data retrieval failed.")
        return self._dlower.copy()

    def getShaping(self):
        """pulls LCFS elongation and upper/lower triangularity.
        
        Returns:
            namedtuple containing (kappa, delta_u, delta_l)

        Raises:
            ValueError: if module cannot retrieve data from the AUG AFS system.
        """
        try:
            kap = self.getElongation()
            du = self.getUpperTriangularity()
            dl = self.getLowerTriangularity()
            data = namedtuple("Shaping", ["kappa", "delta_u", "delta_l"])
            return data(kappa=kap, delta_u=du, delta_l=dl)
        except ValueError:
            raise ValueError("data retrieval failed.")

    def getMagR(self, length_unit=1):
        """returns magnetic-axis major radius.

        Returns:
            magR (Array): [nt] array of major radius of magnetic axis.

        Raises:
            ValueError: if module cannot retrieve data from the AUG AFS system.
        """
        if self._rmag is None:
            try:
                self._rmag = self._equ.Rmag.copy()
                self._defaultUnits["_rmag"] = str("m")
            except AttributeError:
                raise ValueError("data retrieval failed.")
        unit_factor = self._getLengthConversionFactor(
            self._defaultUnits["_rmag"], length_unit
        )
        return unit_factor * self._rmag.copy()

    def getMagZ(self, length_unit=1):
        """returns magnetic-axis Z.

        Returns:
            magZ (Array): [nt] array of Z of magnetic axis.

        Raises:
            ValueError: if module cannot retrieve data from the AUG AFS system.
        """
        if self._zmag is None:
            try:
                self._zmag = self._equ.Zmag.copy()
                self._defaultUnits["_zmag"] = str("m")
            except:
                raise ValueError("data retrieval failed.")
        unit_factor = self._getLengthConversionFactor(
            self._defaultUnits["_zmag"], length_unit
        )
        return unit_factor * self._zmag.copy()

    def getAreaLCFS(self, length_unit=2):
        """returns LCFS cross-sectional area.

        Keyword Args:
            length_unit (String or 2): unit for LCFS area.  Defaults to 2, 
                denoting default areal unit (typically m^2).

        Returns:
            areaLCFS (Array): [nt] array of LCFS area.

        Raises:
            ValueError: if module cannot retrieve data from the AUG AFS system.
        """
        if self._areaLCFS is None:
            try:
                self._areaLCFS = self._equ.area.copy()
                self._defaultUnits["_areaLCFS"] = self._equ.area.phys_unit
            except:
                raise ValueError("data retrieval failed.")
        # Units should be cm^2:
        unit_factor = self._getLengthConversionFactor(
            self._defaultUnits["_areaLCFS"], length_unit
        )
        return unit_factor * self._areaLCFS.copy()

    def getAOut(self, length_unit=1):
        """returns outboard-midplane minor radius at LCFS.

        Keyword Args:
            length_unit (String or 1): unit for minor radius.  Defaults to 1, 
                denoting default length unit (typically m).

        Returns:
            aOut (Array): [nt] array of LCFS outboard-midplane minor radius.

        Raises:
            ValueError: if module cannot retrieve data from the AUG AFS system.
        """
        if self._aLCFS is None:
            try:
                self._aLCFS = self._equ.ahor.copy()
                self._defaultUnits["_aLCFS"] = "m" # not in sf.EQU, must hard code this
            except:
                raise ValueError("data retrieval failed.")
        unit_factor = self._getLengthConversionFactor(
            self._defaultUnits["_aLCFS"], length_unit
        )
        return unit_factor * self._aLCFS.copy()

    def getRmidOut(self, length_unit=1):
        """returns outboard-midplane major radius.

        Keyword Args:
            length_unit (String or 1): unit for major radius.  Defaults to 1, 
                denoting default length unit (typically m).
       
        Raises:
            NotImplementedError: Not implemented on ASDEX-Upgrade reconstructions.
        """
        if self._RmidLCFS is None:
            try:
                self._RmidLCFS = self._equ.Raus.copy()
                self._defaultUnits["_RmidLCFS"] = "m" # not in sf.EQU, must hard code this
            except:
                raise ValueError("data retrieval failed.")
        unit_factor = self._getLengthConversionFactor(
            self._defaultUnits["_aLCFS"], length_unit
        )
        return unit_factor * self._aLCFS.copy()

    def getGeometry(self, length_unit=None):
        """pulls dimensional geometry parameters.
        
        Returns:
            namedtuple containing (magR,magZ,areaLCFS,aOut,RmidOut)

        Raises:
            ValueError: if module cannot retrieve data from the AUG AFS system.
        """
        try:
            Rmag = self.getMagR(
                length_unit=(length_unit if length_unit is not None else 1)
            )
            Zmag = self.getMagZ(
                length_unit=(length_unit if length_unit is not None else 1)
            )
            AreaLCFS = self.getAreaLCFS(
                length_unit=(length_unit if length_unit is not None else 2)
            )
            aOut = self.getAOut(
                length_unit=(length_unit if length_unit is not None else 1)
            )
            RmidOut = self.getRmidOut(
                length_unit=(length_unit if length_unit is not None else 1)
            )
            data = namedtuple(
                "Geometry", ["Rmag", "Zmag", "AreaLCFS", "aOut", "RmidOut"]
            )
            return data(
                Rmag=Rmag, Zmag=Zmag, AreaLCFS=AreaLCFS, aOut=aOut, RmidOut=RmidOut
            )
        except ValueError:
            raise ValueError("data retrieval failed.")

    def getQProfile(self):
        """returns profile of safety factor q.

        Returns:
            qpsi (Array): [nt,npsi] array of q on flux surface psi.

        Raises:
            ValueError: if module cannot retrieve data from the AUG AFS system.
        """
        if self._qpsi is None:
            try:
                self._qpsi = self._equ.q.copy()
                self._defaultUnits["_qpsi"] = self._equ.q.phys_unit
            except:
                raise ValueError("data retrieval failed.")
        return self._qpsi.copy()

    def getQ0(self):
        """returns q on magnetic axis,q0.

        Returns:
            q0 (Array): [nt] array of q(psi=0).

        Raises:
            ValueError: if module cannot retrieve data from the AUG AFS system.
        """
        if self._q0 is None:
            try:
                self._q0 = self._equ.q0.copy()
                self._defaultUnits["_q0"] = self._equ.q0.phys_unit
            except AttributeError:
                raise ValueError("data retrieval failed.")
        return self._q0.copy()

    def getQ95(self):
        """returns q at 95% flux surface.

        Returns:
            q95 (Array): [nt] array of q(psi=0.95).

        Raises:
            ValueError: if module cannot retrieve data from the AUG AFS system.
        """
        if self._q95 is None:
            try:
                self._q95 = self._equ.q95.copy()
                self._defaultUnits["_q95"] = self._equ.q95.phys_unit
            except AttributeError:
                raise ValueError("data retrieval failed.")
        return self._q95.copy()

    def getQLCFS(self):
        """returns q on LCFS (interpolated).
       
        Raises:
            NotImplementedError: Not implemented on ASDEX-Upgrade reconstructions.
        """
        raise NotImplementedError("self.getQLCFS not implemented.")

    def getQ1Surf(self, length_unit=1):
        """returns outboard-midplane minor radius of q=1 surface.
       
        Raises:
            NotImplementedError: Not implemented on ASDEX-Upgrade reconstructions.
        """
        raise NotImplementedError("self.getQ1Surf not implemented.")

    def getQ2Surf(self, length_unit=1):
        """returns outboard-midplane minor radius of q=2 surface.
       
        Raises:
            NotImplementedError: Not implemented on ASDEX-Upgrade reconstructions.
        """
        raise NotImplementedError("self.getQ2Surf not implemented.")

    def getQ3Surf(self, length_unit=1):
        """returns outboard-midplane minor radius of q=3 surface.
       
        Raises:
            NotImplementedError: Not implemented on ASDEX-Upgrade reconstructions.
        """
        raise NotImplementedError("self.getQ3Surf not implemented.")

    def getQs(self, length_unit=1):
        """pulls q values.
        
        Raises:
            NotImplementedError: Not implemented on ASDEX-Upgrade reconstructions.
        """
        raise NotImplementedError("self.getQs not implemented.")

    def getBtVac(self):
        """Returns vacuum toroidal field on-axis. THIS MAY BE INCORRECT

        Returns:
            BtVac (Array): [nt] array of vacuum toroidal field.

        Raises:
            ValueError: if module cannot retrieve data from the AUG AFS system.
        """
        if self._btaxv is None:
            try:
                _btaxv = sf.SFREAD(self._shot, "MBI")
                self._btaxv = interp1d(_btaxv.gettimebase("BTF"), _btaxv.getobject("BTF").data())(self._time)
                self._defaultUnits["_btaxv"] = 'T'
            except AttributeError:
                raise ValueError("data retrieval failed.")
        return self._btaxv.copy()

    def getBtPla(self):
        """returns on-axis plasma toroidal field.

        Raises:
            NotImplementedError: Not implemented on ASDEX-Upgrade reconstructions.
        """
        if self._btaxp is None:
            try:
                self._btaxp = self._equ.bave[:,0].copy()
                self._defaultUnits["_btaxv"] = 'T' # Not in sf.EQU, must hard code this
            except AttributeError:
                raise ValueError("data retrieval failed.")
        return self._btaxp.copy()

    def getBpAvg(self):
        """returns average poloidal field.

        Raises:
            NotImplementedError: Not implemented on ASDEX-Upgrade reconstructions.
        """
        raise NotImplementedError("self.getFields not implemented.")

    def getFields(self):
        """pulls vacuum and plasma toroidal field, avg poloidal field.

        Raises:
            NotImplementedError: Not implemented on ASDEX-Upgrade reconstructions.
        """
        raise NotImplementedError("self.getFields not implemented.")

    def getIpCalc(self):
        """returns Plasma Current from FPcurr parametrization (ask M. Dunne, I have no idea...)

        Returns:
            IpCalc (Array): [nt] array of the reconstructed plasma current.

        Raises:
            ValueError: if module cannot retrieve data from the AUG AFS system.
        """
        if self._IpCalc is None:
            try:
                _IpCalc = sf.SFREAD(self._shot, "FPC")
                self._IpCalc = interp1d(_IpCalc.gettimebase("IpiFP"), _IpCalc.getobject("IpiFP"))(self._time)
                self._defaultUnits["_IpCalc"] = 'A'
            except AttributeError:
                raise ValueError("data retrieval failed.")
        return self._IpCalc.copy()

    def getIpMeas(self):
        """returns magnetics-measured plasma current.

        Returns:
            IpMeas (Array): [nt] array of measured plasma current.

        Raises:
            ValueError: if module cannot retrieve data from the AUG AFS system.
        """
        if self._IpMeas is None:
            try:
                _IpMeas = sf.SFREAD(self._shot, "MAG")
                self._IpMeas = interp1d(_IpMeas.gettimebase("Ipa"), _IpMeas.getobject("Ipa"))(self._time)
                self._defaultUnits["_IpMeas"] = 'A'
            except AttributeError:
                raise ValueError("data retrieval failed.")
        return self._IpMeas.copy()

    def getJp(self):
        """returns the calculated plasma current density Jp on flux grid.

        Returns:
            Jp (Array): [nt,nz,nr] array of current density.

        Raises:
            ValueError: if module cannot retrieve data from the AUG AFS system.
        """
        raise NotImplementedError("self.getJp not implemented for AUG reconstructions.")

    def getBetaT(self):
        """returns the calculated toroidal beta.
        It is not saved in any shotfile (that I know...)
        so I calculate it from Wmhd, volume and Btor

        Raises:
            NotImplementedError: Not implemented on ASDEX-Upgrade reconstructions.
        """
        if self._betat is None:
            try:
                _WMHD, _Vol = self.getWMHD(), self.getVolLCFS()
                _pAvg = _WMHD/_Vol
                _bt0 = self.getBtPla()
                self._betat = 2*mu_0 * _pAvg / _bt0**2
            except:
                raise ValueError("data retrieval failed.")
        return self._betat.copy()

    def getBetaP(self):
        """returns the calculated poloidal beta.

        Returns:
            BetaP (Array): [nt] array of the calculated average poloidal beta.

        Raises:
            ValueError: if module cannot retrieve data from the AUG AFS system.
        """
        if self._betap is None:
            try:
                self._betap = self._equ.betpol.copy()
                self._defaultUnits["_betap"] = self._equ.betpol.phys_unit
            except AttributeError:
                raise ValueError("data retrieval failed.")
        return self._betap.copy()

    def getLi(self):
        """returns the calculated internal inductance.

        Returns:
            Li (Array): [nt] array of the calculated internal inductance.

        Raises:
            ValueError: if module cannot retrieve data from the AUG afs system.
        """
        if self._Li is None:
            try:
                self._Li = self._equ.li.copy()
                self._defaultUnits["_Li"] = self._equ.li.phys_unit
            except AttributeError:
                raise ValueError("data retrieval failed.")
        return self._Li.copy()

    def getBetas(self):
        """pulls calculated betap, betat, internal inductance.

        Raises:
            NotImplementedError: Not implemented on ASDEX-Upgrade reconstructions.
        """
        try:
            betat = self.getBetaT()
            betap = self.getBetaP()
            Li = self.getLi()
            data = namedtuple("Betas", ["betat", "betap", "Li"])
            return data(betat=betat, betap=betap, Li=Li)
        except ValueError:
            raise ValueError("data retrieval failed.")        

    def getDiamagFlux(self):
        """returns the measured diamagnetic-loop flux.

        Raises:
            NotImplementedError: Not implemented on ASDEX-Upgrade reconstructions.
        """
        raise NotImplementedError("self.getDiamagFlux not implemented.")

    def getDiamagBetaT(self):
        """returns diamagnetic-loop toroidal beta.

        Raises:
            NotImplementedError: Not implemented on ASDEX-Upgrade reconstructions.
        """
        raise NotImplementedError("self.getDiamagBetaT not implemented.")

    def getDiamagBetaP(self):
        """returns diamagnetic-loop avg poloidal beta.

        Raises:
            NotImplementedError: Not implemented on ASDEX-Upgrade reconstructions.
        """
        raise NotImplementedError("self.getDiamagBetaP not implemented.")

    def getDiamagTauE(self):
        """returns diamagnetic-loop energy confinement time.

        Raises:
            NotImplementedError: Not implemented on ASDEX-Upgrade reconstructions.
        """
        raise NotImplementedError("self.getDiamagTauE not implemented.")

    def getDiamagWp(self):
        """returns diamagnetic-loop plasma stored energy.
        
        Raises:
            NotImplementedError: Not implemented on ASDEX-Upgrade reconstructions.
        """
        raise NotImplementedError("self.getDiamagWp not implemented.")

    def getDiamag(self):
        """pulls diamagnetic flux measurements, toroidal and poloidal beta, 
        energy confinement time and stored energy.
        
        Raises:
            NotImplementedError: Not implemented on ASDEX-Upgrade reconstructions.
        """
        raise NotImplementedError("self.getDiamag not implemented.")

    def getWMHD(self):
        """returns calculated MHD stored energy.

        Returns:
            WMHD (Array): [nt] array of the calculated stored energy.

        Raises:
            ValueError: if module cannot retrieve data from the AUG afs system.
        """
        if self._WMHD is None:
            try:
                self._WMHD = self._equ.Wmhd.copy()
                self._defaultUnits["_WMHD"] = self._equ.Wmhd.phys_unit
            except AttributeError:
                raise ValueError("data retrieval failed.")
        return self._WMHD.copy()

    def getTauMHD(self):
        """returns the calculated MHD energy confinement time.

        Raises:
            NotImplementedError: Not implemented on ASDEX-Upgrade reconstructions.
        """
        raise NotImplementedError("self.getTauMHD not implemented.")

    def getPinj(self):
        """returns the injected power.

        Raises:
            NotImplementedError: Not implemented on ASDEX-Upgrade reconstructions.
            .
        """
        raise NotImplementedError("self.getPinj not implemented.")

    def getWbdot(self):
        """returns the calculated d/dt of magnetic stored energy.

        Raises:
            NotImplementedError: Not implemented on ASDEX-Upgrade reconstructions.
        """
        raise NotImplementedError("self.getWbdot not implemented.")

    def getWpdot(self):
        """returns the calculated d/dt of plasma stored energy.

        Raises:
            NotImplementedError: Not implemented on ASDEX-Upgrade reconstructions.
        """
        raise NotImplementedError("self.getWpdot not implemented.")

    def getBCentr(self):
        """returns Vacuum toroidal magnetic field at center of plasma

        Returns:
            B_cent (Array): [nt] array of B_t at center [T]

        Raises:
            ValueError: if module cannot retrieve data from the AUG afs system.
        """
        if self._BCentr is None:
            try:
                try:
                    temp = sf.SFREAD(self._shot, "MBI")# self._mdsaugdiag("MBI", "BTFABB")
                    BCentr = temp.getobject("BTFABB")
                    self._BCentr = BCentr[
                        self._getNearestIdx(
                            self.getTimeBase(), temp.gettimebase("BTFABB")
                        )
                    ]
                    self._defaultUnits["_BCentr"] = BCentr.phys_unit
                except:
                    temp = sf.SFREAD(self._shot, "MBI")
                    BCentr = temp.getobject("BTF")
                    self._BCentr = BCentr[
                        self._getNearestIdx(
                            self.getTimeBase(), temp.gettimebase("BTF")
                        )
                    ]
                    self._defaultUnits["_BCentr"] = BCentr.phys_unit
            except AttributeError:
                raise ValueError("data retrieval failed.")

        return self._BCentr

    def getRCentr(self, length_unit=1):
        """Returns Radius of BCenter measurement

        Returns:
            R: Radial position where Bcent calculated [m]
        """
        if self._RCentr is None:
            self._RCentr = 1.65  # Hardcoded from MAI file description of BTF
            self._defaultUnits["_RCentr"] = "m"
        unit_factor = self._getLengthConversionFactor(
            self._defaultUnits["_RCentr"], length_unit
        )
        return self._RCentr * unit_factor

    def getLowerXpoint(self):
        """Returns (R, Z) values of the lower x-points as a function of time

        Returns:
            R, Z: of the lower X-point as a function of time
        """
        if self._xlow is None:
            try:
                self._xlow = numpy.zeros((self._time.size, 2))
                self._xlow[0, :] = self._equ.Rxpu.copy()
                self._xlow[1, :] = self._equ.Zxpu.copy()
                self._defaultUnits["_xlow"] = "m"
            except AttributeError:
                raise ValueError("data retrieval failed.")
        return self._xlow.copy()

    def getUpperXpoint(self):
        """Returns (R, Z) values of the upper x-points as a function of time

        Returns:
            R, Z: of the upper X-point as a function of time
        """
        if self._xup is None:
            try:
                self._xup = numpy.zeros((self._time.size, 2))
                self._xup[0, :] = self._equ.Rxpo.copy()
                self._xup[1, :] = self._equ.Zxpo.copy()
                self._defaultUnits["_xup"] = "m"
            except AttributeError:
                raise ValueError("data retrieval failed.")
        return self._xup.copy()

    def getEnergy(self):
        """pulls the calculated energy parameters - stored energy, tau_E,
            injected power, d/dt of magnetic and plasma stored energy.

            Raises:
                NotImplementedError: Not implemented on ASDEX-Upgrade reconstructions.
            """
        raise NotImplementedError("self.getEnergy not implemented.")

    def getMachineCrossSection(self):
        """Returns R,Z coordinates of vacuum-vessel wall for masking, plotting 
        routines.
        
        Returns:
            (`R_limiter`, `Z_limiter`)

            * **R_limiter** (`Array`) - [n] array of x-values for machine cross-section.
            * **Z_limiter** (`Array`) - [n] array of y-values for machine cross-section.
        """
        if self._Rlimiter is None or self._Zlimiter is None:
            try:
                self._Rlimiter, self._Zlimiter = self._VesselgetMachineCrossSection()

            except AttributeError:
                raise ValueError("data retrieval failed.")
        return (self._Rlimiter, self._Zlimiter)

    def getMachineCrossSectionFull(self):
        """Returns R,Z coordinates of vacuum-vessel wall for plotting routines.
        
        Absent additional vector-graphic data on machine cross-section, returns
        :py:meth:`getMachineCrossSection`.
        
        Returns:
            result from getMachineCrossSection().
        """
        x, y = self._VesselgetMachineCrossSectionFull()
        x[x > self.getRGrid().max()] = self.getRGrid().max()

        return (x, y)

    def getCurrentSign(self):
        """Returns the sign of the current, based on the check in Steve Wolfe's 
        IDL implementation efit_rz2psi.pro.

        Returns:
            currentSign (Integer): 1 for positive-direction current, -1 for negative.
        """
        if self._currentSign is None:
            self._currentSign = numpy.nanmedian(numpy.sign(self.getIpMeas()))
        return self._currentSign
    
    def remapLCFS(self, mask=False):
        """Overwrites RLCFS, ZLCFS values pulled with explicitly-calculated 
        contour of psinorm=1 surface.  This is then masked down by the limiter
        array using core.inPolygon, restricting the contour to the closed
        plasma surface and the divertor legs.

        Keyword Args:
            mask (Boolean): Default False.  Set True to mask LCFS path to 
                limiter outline (using inPolygon).  Set False to draw full 
                contour of psi = psiLCFS.

        Raises:
            NotImplementedError: if :py:mod:`matplotlib.pyplot` is not loaded.
            ValueError: if limiter outline is not available.
        """
        if not _has_plt:
            raise NotImplementedError(
                "Requires matplotlib.pyplot for contour calculation."
            )

        try:
            Rlim, Zlim = self.getMachineCrossSection()
        except:
            raise ValueError(
                "Limiter outline (self.getMachineCrossSection) must be available."
            )

        psiRZ = numpy.asarray(self.getFluxGrid())  # [nt,nZ,nR]
        R = numpy.asarray(self.getRGrid())
        Z = numpy.asarray(self.getZGrid())
        psiLCFS = numpy.asarray(self.getFluxLCFS())

        RLCFS_stores = []
        ZLCFS_stores = []
        maxlen = 0
        nt = len(self.getTimeBase())
        for i in range(nt):
            cs = cntr(x=R, y=Z, z=psiRZ[i], name='serial').lines(psiLCFS[i])
            RLCFS_frame = []
            ZLCFS_frame = []
            for v in cs:
                RLCFS_frame.extend(v[:, 0])
                ZLCFS_frame.extend(v[:, 1])
                RLCFS_frame.append(numpy.nan)
                ZLCFS_frame.append(numpy.nan)
            RLCFS_frame = numpy.array(RLCFS_frame)
            ZLCFS_frame = numpy.array(ZLCFS_frame)

            # generate masking array to vessel
            if mask:
                maskarr = numpy.array([False for i in range(len(RLCFS_frame))])
                for i, x in enumerate(RLCFS_frame):
                    y = ZLCFS_frame[i]
                    maskarr[i] = inPolygon(Rlim, Zlim, x, y)

                RLCFS_frame = RLCFS_frame[maskarr]
                ZLCFS_frame = ZLCFS_frame[maskarr]

            if len(RLCFS_frame) > maxlen:
                maxlen = len(RLCFS_frame)
            RLCFS_stores.append(RLCFS_frame)
            ZLCFS_stores.append(ZLCFS_frame)

        RLCFS = numpy.zeros((nt, maxlen))
        ZLCFS = numpy.zeros((nt, maxlen))
        for i in range(nt):
            RLCFS_frame = RLCFS_stores[i]
            ZLCFS_frame = ZLCFS_stores[i]
            ni = len(RLCFS_frame)
            RLCFS[i, 0:ni] = RLCFS_frame
            ZLCFS[i, 0:ni] = ZLCFS_frame

        # store final values
        self._RLCFS = RLCFS
        self._ZLCFS = ZLCFS

        # set default unit parameters, based on RZ grid
        rUnit = self._defaultUnits["_rGrid"]
        zUnit = self._defaultUnits["_zGrid"]
        self._defaultUnits["_RLCFS"] = rUnit
        self._defaultUnits["_ZLCFS"] = zUnit
    
    def getParam(self, path):
        """Backup function, applying a direct path input for tree-like data 
        storage access for parameters not typically found in 
        :py:class:`Equilbrium <eqtools.core.Equilbrium>` object.  
        Directly calls attributes read from g/a-files in copy-safe manner.

        Args:
            name (String): Parameter name for value stored in EqdskReader 
                instance.

        Raises:
            NotImplementedError: Not implemented on ASDEX-Upgrade reconstructions.
        """
        raise NotImplementedError("self.getEnergy not implemented.")

    def getMachineCrossSection(self):
        """Returns R,Z coordinates of vacuum-vessel wall for masking, plotting 
        routines.
        
        Returns:
            (`R_limiter`, `Z_limiter`)

            * **R_limiter** (`Array`) - [n] array of x-values for machine cross-section.
            * **Z_limiter** (`Array`) - [n] array of y-values for machine cross-section.
        """
        if self._Rlimiter is None or self._Zlimiter is None:
            try:
                self._Rlimiter, self._Zlimiter = self._VesselgetMachineCrossSection()

            except AttributeError:
                raise ValueError("data retrieval failed.")
        return (self._Rlimiter, self._Zlimiter)

    def getMachineCrossSectionFull(self):
        """Returns R,Z coordinates of vacuum-vessel wall for plotting routines.
        
        Absent additional vector-graphic data on machine cross-section, returns
        :py:meth:`getMachineCrossSection`.
        
        Returns:
            result from getMachineCrossSection().
        """
        x, y = self._VesselgetMachineCrossSectionFull()
        x[x > self.getRGrid().max()] = self.getRGrid().max()

        return (x, y)

    def _ygcauginterface(self):
        self._vessel_components = {
            0: (
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                0,
                1,
                1,
                1,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ),
            948: (
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ),
            8650: (
                1,
                1,
                0,
                0,
                1,
                0,
                1,
                0,
                1,
                1,
                1,
                1,
                0,
                0,
                0,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                0,
            ),
            9401: (
                1,
                1,
                0,
                0,
                1,
                0,
                1,
                0,
                1,
                1,
                1,
                1,
                0,
                0,
                0,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
            ),
            12751: (
                1,
                1,
                0,
                0,
                1,
                0,
                1,
                0,
                1,
                1,
                1,
                1,
                0,
                0,
                0,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                0,
                1,
            ),
            14051: (
                1,
                1,
                0,
                0,
                1,
                0,
                1,
                0,
                1,
                1,
                1,
                1,
                0,
                0,
                0,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
            ),
            14601: (
                1,
                1,
                0,
                0,
                1,
                0,
                0,
                0,
                1,
                1,
                1,
                1,
                0,
                0,
                0,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
            ),
            16315: (
                1,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                1,
                1,
                1,
                0,
                0,
                0,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                0,
                0,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
            ),
            18204: (
                1,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                1,
                1,
                1,
                0,
                0,
                0,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                0,
                0,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
            ),
            19551: (
                1,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                1,
                1,
                1,
                0,
                0,
                0,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                0,
                1,
                1,
                1,
                0,
                0,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
            ),
            21485: (
                1,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                1,
                1,
                1,
                0,
                0,
                0,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                0,
                0,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
            ),
            25891: (
                1,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                1,
                1,
                1,
                0,
                0,
                0,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                0,
                1,
                1,
                1,
                1,
                1,
                0,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
            ),
            30136: (
                1,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                1,
                1,
                1,
                0,
                0,
                0,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                0,
                1,
                1,
                1,
                1,
                1,
                0,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
            ),
        }
        # counter-clockwise from the inner wall order of components
        self._order = {
            0: (9, 8, 7, 5, 2, 1, 14, 13, 12, 0, 4, 6),
            948: (9, 8, 7, 5, 2, 1, 10, 0, 4, 6),
            8650: (9, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 1, 10, 0, 4, 6),
            9401: (9, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 10, 0, 4, 6),
            12751: (9, 29, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 10, 0, 4, 6),
            14051: (
                9,
                29,
                15,
                16,
                17,
                18,
                19,
                20,
                21,
                22,
                23,
                24,
                25,
                26,
                28,
                10,
                0,
                4,
                6,
            ),
            14601: (
                9,
                29,
                15,
                16,
                17,
                18,
                19,
                20,
                21,
                22,
                23,
                24,
                25,
                26,
                28,
                10,
                0,
                4,
                30,
                31,
                32,
                33,
                34,
            ),
            16315: (
                9,
                15,
                16,
                17,
                18,
                19,
                20,
                21,
                22,
                23,
                24,
                25,
                26,
                27,
                1,
                10,
                0,
                35,
                36,
                37,
                38,
                39,
                30,
                31,
                32,
                33,
                34,
            ),
            18204: (
                9,
                15,
                16,
                17,
                18,
                19,
                20,
                21,
                22,
                23,
                24,
                25,
                26,
                27,
                1,
                10,
                0,
                35,
                36,
                37,
                38,
                39,
                30,
                31,
                32,
                33,
                34,
            ),
            19551: (
                9,
                15,
                16,
                17,
                18,
                19,
                20,
                21,
                22,
                23,
                24,
                25,
                26,
                27,
                1,
                10,
                0,
                35,
                36,
                37,
                38,
                39,
                30,
                31,
                32,
                33,
                34,
            ),
            21485: (
                9,
                15,
                16,
                17,
                18,
                19,
                20,
                21,
                22,
                23,
                24,
                25,
                26,
                27,
                1,
                10,
                0,
                35,
                36,
                37,
                38,
                39,
                30,
                31,
                32,
                33,
                34,
            ),
            25891: (
                9,
                15,
                16,
                17,
                18,
                19,
                20,
                21,
                22,
                23,
                24,
                25,
                26,
                27,
                10,
                41,
                42,
                43,
                36,
                37,
                38,
                39,
                30,
                31,
                32,
                33,
                34,
            ),
            30136: (
                9,
                15,
                16,
                17,
                18,
                19,
                20,
                21,
                22,
                23,
                24,
                25,
                26,
                27,
                10,
                41,
                42,
                43,
                36,
                37,
                38,
                39,
                30,
                31,
                32,
                33,
                34,
            ),
        }
        # start location in array of values for given object closest to plasma
        self._start = {
            0: (21, 0, 3, 3, 1, 4, 2, 3, 3, 9, 0, 1),
            948: (21, 0, 3, 3, 1, 4, 2, 9, 0, 1),
            8650: (21, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 2, 9, 0, 1),
            9401: (21, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 9, 0, 1),
            12751: (21, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 9, 0, 1),
            14051: (21, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 9, 0, 1),
            14601: (
                21,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                2,
                9,
                0,
                0,
                0,
                0,
                0,
                0,
            ),
            16315: (
                21,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                13,
                4,
                2,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ),
            18204: (
                21,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                13,
                4,
                2,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ),
            19551: (
                21,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                13,
                4,
                2,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ),
            21485: (
                21,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                4,
                2,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ),
            25891: (
                21,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                2,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ),
            30136: (
                21,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                2,
                0,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ),
        }
        # end location in array of values for given object closest to plasma
        self._end = {
            0: (42, 2, 7, 7, 5, 6, 10, 5, 10, 13, 4, 5),
            948: (42, 2, 7, 7, 5, 6, 34, 13, 4, 5),
            8650: (42, 4, 26, 25, 32, 2, 9, 8, 3, 35, 22, 28, 3, 5, 34, 13, 4, 5),
            9401: (42, 4, 26, 22, 32, 5, 9, 8, 5, 35, 22, 26, 5, 27, 13, 4, 5),
            12751: (39, 2, 4, 26, 25, 32, 2, 9, 8, 3, 35, 22, 28, 5, 25, 13, 4, 5),
            14051: (39, 2, 5, 5, 20, 26, 8, 11, 7, 5, 3, 4, 14, 4, 5, 25, 13, 4, 10),
            14601: (
                39,
                2,
                5,
                5,
                20,
                26,
                8,
                11,
                7,
                5,
                3,
                4,
                14,
                4,
                5,
                25,
                13,
                4,
                2,
                2,
                2,
                2,
                4,
            ),
            16315: (
                42,
                5,
                5,
                20,
                26,
                8,
                11,
                7,
                5,
                3,
                4,
                14,
                4,
                16,
                5,
                34,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                4,
            ),
            18204: (
                42,
                5,
                5,
                20,
                26,
                8,
                11,
                7,
                5,
                3,
                4,
                14,
                4,
                16,
                5,
                34,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                7,
            ),
            19551: (
                42,
                5,
                5,
                20,
                26,
                8,
                11,
                7,
                5,
                3,
                4,
                14,
                4,
                16,
                5,
                34,
                2,
                2,
                7,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                7,
            ),
            21485: (
                42,
                6,
                5,
                15,
                6,
                6,
                6,
                7,
                6,
                6,
                3,
                12,
                6,
                3,
                5,
                34,
                2,
                2,
                7,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                7,
            ),
            25891: (
                42,
                6,
                5,
                15,
                6,
                6,
                6,
                7,
                6,
                6,
                3,
                18,
                6,
                3,
                57,
                2,
                2,
                18,
                7,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                7,
            ),
            30136: (
                42,
                6,
                5,
                15,
                6,
                6,
                6,
                7,
                6,
                9,
                2,
                18,
                4,
                2,
                57,
                2,
                2,
                17,
                7,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                7,
            ),
        }
        # Which objects are stored reverse of the counter-clockwise motion as described in order
        self._rev = {
            0: (0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
            948: (0, 1, 1, 1, 1, 1, 1, 1, 1, 1),
            8650: (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1),
            9401: (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1),
            12751: (0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1),
            14051: (0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1),
            14601: (
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                1,
                1,
                0,
                0,
                0,
                0,
                0,
            ),
            16315: (
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ),
            18204: (
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ),
            19551: (
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ),
            21485: (
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ),
            25891: (
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ),
            30136: (
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ),
        }
        # ONLY CERTAIN YGC FILES EXIST I MEAN CMON ITS NOT THAT MUCH MEMORY
        self._ygc_shotfiles = numpy.array(
            [
                0,
                948,
                8650,
                9401,
                12751,
                14051,
                14601,
                16315,
                18204,
                19551,
                21485,
                25891,
                30136,
            ]
        )

    def _getDataVessel(self, shot):
        try:
            self._ygc_shot = self._ygc_shotfiles[
                numpy.searchsorted(self._ygc_shotfiles, [shot], "right") - 1
            ][
                0
            ]  # find nearest shotfile which is the before it

            if self._ygc_shot < 8650:
                ccT = sf.SFREAD(self._ygc_shotfiles[2], "YGC")  # This is because of shots <8650 not having RrGC zzGC or inxbeg
            else:
                ccT = sf.SFREAD(self._ygc_shot, "YGC")
            xvctr = ccT.getobject("RrGC")
            yvctr = ccT.getobject("zzGC")
            nvctr = ccT.getobject("inxbeg")
            nvctr = nvctr.astype(int) - 1
        except (AttributeError):
            raise ValueError("data retrieval failed.")
        except:
            raise ValueError("data load failed.")
        return xvctr, yvctr, nvctr

    def _VesselgetMachineCrossSection(self):
        """Returns R,Z coordinates of vacuum-vessel wall for masking, plotting 
        routines.
        
        Returns:
            (`R_limiter`, `Z_limiter`)

            * **R_limiter** (`Array`) - [n] array of x-values for machine cross-section.
            * **Z_limiter** (`Array`) - [n] array of y-values for machine cross-section.
        """
        xvctr, yvctr, nvctr = self._getDataVessel(self._shot)
        x = []
        y = []

        # by reference to simplify coding
        start = self._start[self._ygc_shot]
        end = self._end[self._ygc_shot]
        rev = self._rev[self._ygc_shot]
        order = self._order[self._ygc_shot]

        for i in range(len(order)):
            idx = nvctr[order[i]]
            xseg = xvctr[idx + start[i] : idx + end[i]]
            yseg = yvctr[idx + start[i] : idx + end[i]]

            if rev[i]:
                xseg = xseg[::-1]
                yseg = yseg[::-1]

            x.extend(xseg)
            y.extend(yseg)

        x.extend([x[0]])
        y.extend([y[0]])

        return (x[::-1], y[::-1])

    def _VesselgetMachineCrossSectionFull(self):
        """Returns R,Z coordinates of vacuum-vessel wall for plotting routines.
        
        Absent additional vector-graphic data on machine cross-section, returns
        :py:meth:`getMachineCrossSection`.
        
        Returns:
            result from getMachineCrossSection().
        """

        xvctr, yvctr, nvctr = self._getDataVessel(self._shot)

        # get valid components which is in the data structure for some shots, but not all and had to be hardcoded
        temp = self._vessel_components[self._ygc_shot]

        x = []
        y = []

        for i in range(len(nvctr) - 1):
            if temp[i]:
                xseg = xvctr[nvctr[i] : nvctr[i + 1]]
                yseg = yvctr[nvctr[i] : nvctr[i + 1]]
                x.extend(xseg)
                y.extend(yseg)
                x.append(None)
                y.append(None)

        x = numpy.array(x[:-1])
        y = numpy.array(y[:-1])
        return (x, y)