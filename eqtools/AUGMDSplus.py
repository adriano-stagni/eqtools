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

from scipy.interpolate import interp1d
from scipy.constants import mu_0
import numpy
from .core import PropertyAccessMixin, ModuleWarning, Equilibrium, inPolygon
import os
import warnings
import numpy as np
from collections import namedtuple
from tqdm import tqdm
from joblib import Parallel, delayed

try:
    import MDSplus

    try:
        from MDSplus._treeshr import TreeException
    except:
        from MDSplus.mdsExceptions import TreeException
    _has_MDS = True
except Exception as _e_MDS:
    if isinstance(_e_MDS, ImportError):
        warnings.warn(
            "MDSplus module could not be loaded -- classes that use "
            "MDSplus for data access will not work.",
            ModuleWarning,
        )
    else:
        warnings.warn(
            "MDSplus module could not be loaded -- classes that use"
            "MDSplus for data access will not work. Exception raised"
            "was of type %s, message was '%s'." % (_e_MDS.__class__, _e_MDS.message),
            ModuleWarning,
        )
    _has_MDS = False

try:
    import matplotlib.pyplot as plt

    _has_plt = True

except:
    warnings.warn(
        "Matplotlib.pyplot module could not be loaded -- classes that "
        "use pyplot will not work.",
        ModuleWarning,
    )
    _has_plt = False
    
# Removed old import call to matplotlib._cntr and legacycontour._cntr
# since they refere to older versions of matplotlib and python and they are deprecated now.
try:
    import matplotlib.pyplot as plt
    from contourpy import contour_generator as cntr
except:
    warnings.warn(
        "Matplotlib.pyplot module could not be loaded -- classes that "
        "use Matplotlib will not work.",
        ModuleWarning,
    )
    _has_cntr = False


class AUGMDSTree(Equilibrium):
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
        tree="EQH",
        edition=0,
        shotfile2=None,
        length_unit="m",
        tspline=False,
        monotonic=True,
        experiment="AUGD",
        server="ssh://localhost",
        connection=None,
    ):

        super(AUGMDSTree, self).__init__(
            length_unit=length_unit, tspline=tspline, monotonic=monotonic
        )

        if isinstance(connection, MDSplus.connection.Connection):
            self._MDSTree = connection
            self.server = self._MDSTree.hostspec
        else:
            self.server = server
            self._MDSTree = MDSplus.Connection(self.server)
        self._shot = shot
        self._tree = tree
        self._edition = 0
        self._experiment = experiment
        print(self._shot, self._tree, edition, experiment)
        try:
            if shotfile2 is None:
                self._treessq = self._relatedSVFile[self._tree]
            # Overwrite getSSQ with a shotfile with same capabilities
            # self.getSSQ = dd.shotfile(
            #     shotfile2, self._shot, edition=edition, experiment=experiment
            # )
        except (KeyError):
            warnings.warn(
                "Companion SV not valid, extracting from " + self._tree + ":SSQ",
                RuntimeWarning,
            )

        self._defaultUnits = {}

        # initialize None for non-essential data

        # grad-shafranov related parameters
        self._fpol = None
        self._fprime = None
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
        self._fluxLabel = None # time-dependent flux label array (t,psi)
        self._fluxVol = None  # volume within flux surface (t,psi)
        self._volLCFS = None  # volume within LCFS (t)
        self._qpsi = None  # q profile (psi,t)
        self._RmidPsi = None  # max major radius of flux surface (t,psi)
        self._xlow = None  # (r, z) position of the lower x-point (r, z, t)
        self._xup = None  # (r, z) position of the upper x-point (r, z, t)

        # AUG SV file flag
        self._SSQ = None

        # Call the get functions to preload the data. Add any other calls you
        # want to preload here.
        # I need first to make a call to conn.get otherwise does not work
        _s = (
            "augdiag({},".format(self._shot)
            + '"'
            + self._tree
            + '","time","'
            + self._experiment
            + '",{}'.format(self._edition)
            + ")"
        )
        _dummy = self._MDSTree.get(_s).data()
        self.getTimeBase()  # check
        self._timeidxend = self.getTimeBase().size
        self.getFluxGrid()  # loads _psiRZ, _rGrid and _zGrid at once. check
        self.getFluxLCFS()  # check
        self.getFluxAxis()  # check
        self.getFluxLabel() # check
        self.getFluxVol()  # check
        self.getVolLCFS()  # check
        self.getQProfile()  #
        self._ygcauginterface()  # needed to initialize Vessel properties
        self.getBtVac()
        self.remapLCFS()

    #
    def _mdsaugdiag(self, shotfile, signal):
        """ wrapper for the augdiag TDI function data"""

        # augdiag (_shot, _diag, _signame, _experiment, _edition, _t1, _t2, _oshot, _oedition)
        _s = (
            "augdiag({},".format(self._shot)
            + '"'
            + shotfile
            + '","'
            + signal
            + '","'
            + self._experiment
            + '",{}'.format(self._edition)
            + ")"
        )
        return self._MDSTree.getObject(_s, timeout=3000000)
    
    def _mdsaugvessel(self, shot, shotfile, signal):
        """ wrapper for the augdiag TDI function data time"""
        _s = (
            "augdiag({},".format(shot)
            + '"'
            + shotfile
            + '","'
            + signal
            + '","'
            + self._experiment
            + '")'
        )
        return self._MDSTree.getObject(_s, timeout=1000000)

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
                timeNode = self._MDSTree.get(f"dim_of(augdiag({self._shot},'{self._tree}','PFxx','{self._experiment}',{self._edition}) ,0)")
                self._time = timeNode.data()
                self._defaultUnits["_time"] = str("s")
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
                psinode = self._mdsaugdiag(self._tree, "PFM")
                self._psiRZ = np.moveaxis(psinode.data(), -1, 0)
                self._defaultUnits["_psiRZ"] = "Vs"  # HARDCODED DUE TO CALIBRATED=FALSE
                self._rGrid = self._mdsaugdiag(self._tree, "Ri").data()[:,0]
                self._defaultUnits["_rGrid"] = str("m")
                self._zGrid = self._mdsaugdiag(self._tree, "Zj").data()[:,0]
                self._defaultUnits["_zGrid"] = str("m")

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
                psiAxisNode = self._mdsaugdiag(self._tree, "PFxx")
                self._psiAxis = psiAxisNode.data()[0, : self._timeidxend]
                self._defaultUnits["_psiAxis"] = str(psiAxisNode.units).strip()
            except:
                raise ValueError("data retrieval failed.")
        return self._psiAxis.copy()

    def getFluxLCFS(self):
        """returns psi at separatrix.

        Returns:
            psiLCFS (Array): [nt] array of psi at LCFS.

        Raises:
            ValueError: if module cannot retrieve data from the AUG shotfile system.
        """
        if self._psiLCFS is None:
            try:
                psiLCFSNode = self._mdsaugdiag(self._tree, "PFL")
                self._psiLCFS = psiLCFSNode.data()[0, : self._timeidxend]
                self._defaultUnits["_psiLCFS"] = str(psiLCFSNode.units).strip()
            except:
                raise ValueError("data retrieval failed.")
        return self._psiLCFS.copy()
    
    def getFluxLabel(self):
        """returns time-dependent poloidal flux label array.
        
        Returns:
            psiLabel (Array) [nt,npfl]: time-dependent array of poloidal flux labels.
            idxunique (Array) [npfl]: indices of the unique positions of the saved psiLabel values.
            idxsorted (Array) [npfl]: indices of sorted unique psiLabel values.
        
        Raises:
            ValueError: if module cannot retrieve data from the AUG shotfile system.
        """
        if self._fluxLabel is None:
            try:
                PFL = self._mdsaugdiag(self._tree, "PFL").data().transpose() # Poloidal flux label
                _, idxunique = np.unique(PFL, axis=1, return_index=True) # Eliminate double elements in PFL
                idxsorted = np.argsort(PFL[0,idxunique]) # Sort PFL elements
                _fluxLabel = PFL[:,idxunique[idxsorted]]
                self._fluxLabel = (_fluxLabel, idxunique, idxsorted)
            except:
                raise ValueError("data retrieval failed.")
        _fluxLabel, idxunique, idxsorted = self._fluxLabel
        return _fluxLabel.copy(), idxunique, idxsorted
    
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
                fluxVolNode = self._mdsaugdiag(self._tree, "Vol")
                _fluxVol = fluxVolNode.data().transpose()[:,::2] # Every second item is dVol/dpsi
                _, idxunique, idxsorted = self.getFluxLabel()
                self._fluxVol = np.array(_fluxVol[:self._timeidxend, idxunique[idxsorted]])
                self._defaultUnits["_fluxVol"] = "m^3"
            except:
                raise ValueError("data retrieval failed.")
        # Default units are m^3, but aren't stored in the tree!
        unit_factor = self._getLengthConversionFactor(
            self._defaultUnits["_fluxVol"], length_unit
        )
        return unit_factor * self._fluxVol.copy()

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
                volLCFSNode = self._mdsaugdiag(self._tree, "Vol")
                self._volLCFS = volLCFSNode.data().transpose()[: self._timeidxend, 0]
                self._defaultUnits["_volLCFS"] = "m^3"
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
                psiLevels, _, _ = self.getFluxLabel()
                psiRZ = self.getFluxGrid()  # [nt,nZ,nR]
                R = self.getRGrid()
                Z = self.getZGrid()
                Rmag, Zmag = self.getMagR(), self.getMagZ()

                def process_timestep(args):
                    it, R, Z, psiRZ_it, psiLevels_it, Rmag_it, Zmag_it = args
                    from contourpy import contour_generator as cntr # Import contourpy inside the function to ensure it's available in each process
                    levels = psiLevels_it # Get levels for this timestep
                    results = np.full(len(levels), np.nan)
                    cg = cntr(x=R, y=Z, z=psiRZ_it, name='serial') # Create contour generator for this timestep
                    cs = np.array(
                        [min(
                            cg.lines(level),
                            key=lambda c: np.hypot(c[:,0] - Rmag_it, c[:,1] - Zmag_it).min(),
                            default=np.array([[np.nan, np.nan]]))
                         for level in levels], dtype='object')
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        results = np.vectorize(lambda c: np.nanmax(c[:,0]))(cs)
                    return it, results
                
                nt = psiLevels.shape[1]
                n_workers = 16 # max(1, os.cpu_count()-1)
                _RmidPsi = np.tile(np.nan, psiLevels.shape)
                results = Parallel(n_jobs=n_workers)(
                    delayed(process_timestep)(
                        (it, R, Z, psiRZ[it], psiLevels[it,:], Rmag[it], Zmag[it])
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
                rgeo = self._mdsaugdiag(self._treessq, "Rgeo")
                RLCFSNode = self._mdsaugdiag(self._treessq, "rays")
                RLCFStemp = np.hstack(
                    (
                        np.atleast_2d(RLCFSNode.data().transpose()[:, -1]).T,
                        RLCFSNode.data().transpose(),
                    )
                )
                templen = RLCFSNode.data().transpose().shape

                self._RLCFS = np.tile(
                    rgeo.data(), (templen[1] + 1, 1)
                ).T + RLCFStemp * np.cos(
                    np.tile(
                        (np.linspace(0, 2 * numpy.pi, templen[1] + 1)), (templen[0], 1),
                    )
                )  # construct a 2d grid of angles, take cos, multiply by radius
                self._defaultUnits["_RLCFS"] = str("m")
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
                zgeo = self._mdsaugdiag(self._treessq, "Zgeo")
                ZLCFSNode = self._mdsaugdiag(self._treessq, "rays")
                ZLCFStemp = np.hstack(
                    (
                        numpy.atleast_2d(ZLCFSNode.data().transpose()[:, -1]).T,
                        ZLCFSNode.data().transpose(),
                    )
                )
                templen = ZLCFSNode.data().transpose().shape

                self._ZLCFS = np.tile(
                    zgeo.data(), (templen[1] + 1, 1)
                ).T + ZLCFStemp * np.sin(
                    np.tile(
                        (np.linspace(0, 2 * np.pi, templen[1] + 1)), (templen[0], 1),
                    )
                )  # construct a 2d grid of angles, take sin, multiply by radius
                self._defaultUnits["_ZLCFS"] = str("m")
            except KeyError:
                self.remapLCFS()
                self._defaultUnits["_RLCFS"] = str("m")
                self._defaultUnits["_ZLCFS"] = str("m")
            except:
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
                fNode = self._mdsaugdiag(self._tree, "Jpol") # From definition of F with poloidal current
                _fpol = fNode.data().transpose()[:,::2] * 2e-7 # Every second item is dJpol/dpsi
                _, idxunique, idxsorted = self.getFluxLabel()
                self._fpol = _fpol[:self._timeidxend, idxunique[idxsorted]]
                self._defaultUnits["_fpol"] = str("T m")
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
                fluxPresNode = self._mdsaugdiag(self._tree, "Pres")
                _fluxPres = fluxPresNode.data().transpose()[:,::2] # Every second item is dPres/dpsi
                _, idxunique, idxsorted = self.getFluxLabel()
                self._fluxPres = _fluxPres[:self._timeidxend, idxunique[idxsorted]]
                self._defaultUnits["_fluxPres"] = str(fluxPresNode.units).strip()
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
                fNode = self._mdsaugdiag(self._tree, "Jpol") # From definition of F with poloidal current
                _fprime = fNode.data().transpose()[1::2] * 2e-7 # Every second element is Fprime, take those only
                _, idxunique, idxsorted = self.getFluxLabel()
                self._fprime = _fprime[:self._timeidxend, idxunique[idxsorted]]
                self._defaultUnits["_fpol"] = str("Tm/psi")
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
                FFPrimeNode = self._mdsaugdiag(self._tree, "FFP")
                _, idxunique, idxsorted = self.getFluxLabel()
                self._ffprim = FFPrimeNode.data().transpose()[:self._timeidxend, idxunique[idxsorted]]
                self._defaultUnits["_ffprim"] = str(FFPrimeNode.units).strip()
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
                pPrimeNode = self._mdsaugdiag(self._tree, "Pres")
                _pprime = pPrimeNode.data().transpose()[1::2] # Every second element is pprime, take those only
                _, idxunique, idxsorted = self.getFluxLabel()
                self._pprime = _pprime[:self._timeidxend, idxunique[idxsorted]]
                self._defaultUnits["_pprime"] = str(pPrimeNode.units).strip()
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
                kappaNode = self._mdsaugdiag(self._treessq, "k")
                self._kappa = kappaNode.data()
                # I do not have the slightest idea why I must do this, I just know that I must do it.
                _kappatime = self._MDSTree.get(f"dim_of(augdiag({self._shot},'{self._treessq}','k','{self._experiment}',{self._edition}))").data()
                mask = np.in1d(_kappatime.astype(self.getTimeBase().dtype), self.getTimeBase())
                _kappatime, self._kappa = _kappatime[mask], self._kappa[mask]
                self._defaultUnits["_kappa"] = str(kappaNode.units).strip()
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
                dupperNode = self._mdsaugdiag(self._treessq, "delRoben")
                self._dupper = dupperNode.data()
                _duppertime = self._MDSTree.get(f"dim_of(augdiag({self._shot},'{self._treessq}','delRoben','{self._experiment}',{self._edition}))").data()
                # I do not have the slightest idea why I must do this, I just know that I must do it.
                mask = np.in1d(_duppertime.astype(self.getTimeBase().dtype), self.getTimeBase())
                _duppertime, self._dupper = _duppertime[mask], self._dupper[mask]
                self._defaultUnits["_dupper"] = str(dupperNode.units).strip()
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
                dlowerNode = self._mdsaugdiag(self._treessq, "delRuntn")
                self._dlower = dlowerNode.data()
                _dlowertime = self._MDSTree.get(f"dim_of(augdiag({self._shot},'{self._treessq}','delRuntn','{self._experiment}',{self._edition}))").data()
                # I do not have the slightest idea why I must do this, I just know that I must do it.
                mask = np.in1d(_dlowertime.astype(self.getTimeBase().dtype), self.getTimeBase())
                _dlowertime, self._dlower = _dlowertime[mask], self._dlower[mask]
                self._defaultUnits["_dlower"] = str(dlowerNode.units).strip()
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
                rmagNode = self._mdsaugdiag(self._treessq, "Rmag")
                self._rmag = rmagNode.data()
                # I do not have the slightest idea why I must do this, I just know that I must do it.
                _rmagtime = self._MDSTree.get(f"dim_of(augdiag({self._shot},'{self._treessq}','Rmag','{self._experiment}',{self._edition}))").data()
                mask = np.in1d(_rmagtime.astype(self.getTimeBase().dtype), self.getTimeBase())
                _rmagtime, self._rmag = _rmagtime[mask], self._rmag[mask]
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
                zmagNode = self._mdsaugdiag(self._treessq, "Zmag")
                self._zmag = zmagNode.data()
                # I do not have the slightest idea why I must do this, I just know that I must do it.
                _zmagtime = self._MDSTree.get(f"dim_of(augdiag({self._shot},'{self._treessq}','Zmag','{self._experiment}',{self._edition}))").data()
                mask = np.in1d(_zmagtime.astype(self.getTimeBase().dtype), self.getTimeBase())
                _zmagtime, self._zmag = _zmagtime[mask], self._zmag[mask]
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
                areaLCFSNode = self._mdsaugdiag(self._tree, "Area")
                self._areaLCFS = areaLCFSNode.data().transpose()[: self._timeidxend, 0]
                self._defaultUnits["_areaLCFS"] = str(areaLCFSNode.units).strip()
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
                aLCFSNode = self._mdsaugdiag(self._treessq, "ahor")
                self._aLCFS = aLCFSNode.data()
                # I do not have the slightest idea why I must do this, I just know that I must do it.
                _aLCFStime = self._MDSTree.get(f"dim_of(augdiag({self._shot},'{self._treessq}','Zmag','{self._experiment}',{self._edition}))").data()
                mask = np.in1d(_aLCFStime.astype(self.getTimeBase().dtype), self.getTimeBase())
                _aLCFStime, self._aLCFS = _aLCFStime[mask], self._aLCFS[mask]
                self._defaultUnits["_aLCFS"] = "m"
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
        raise NotImplementedError("self.getRmidOut not implemented.")

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
                qpsiNode = self._mdsaugdiag(self._tree, "Qpsi")
                _, idxunique, idxsorted = self.getFluxLabel()
                self._qpsi = qpsiNode.data().transpose()[:self._timeidxend, idxunique[idxsorted]]
                self._defaultUnits["_qpsi"] = str(qpsiNode.units).strip()
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
                q0Node = self._mdsaugdiag(self._treessq, "q0")
                self._q0 = q0Node.data()
                # I do not have the slightest idea why I must do this, I just know that I must do it.
                _q0time = self._MDSTree.get(f"dim_of(augdiag({self._shot},'{self._treessq}','Zmag','{self._experiment}',{self._edition}))").data()
                mask = np.in1d(_q0time.astype(self.getTimeBase().dtype), self.getTimeBase())
                _q0time, self._q0 = _q0time[mask], self._q0[mask]
                self._defaultUnits["_q0"] = str(q0Node.units).strip()
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
                q95Node = self._mdsaugdiag(self._treessq, "q95")
                self._q95 = q95Node.data()
                # I do not have the slightest idea why I must do this, I just know that I must do it.
                _q95time = self._MDSTree.get(f"dim_of(augdiag({self._shot},'{self._treessq}','Zmag','{self._experiment}',{self._edition}))").data()
                mask = np.in1d(_q95time.astype(self.getTimeBase().dtype), self.getTimeBase())
                _q95time, self._q95 = _q95time[mask], self._q95[mask]
                self._defaultUnits["_q95"] = str(q95Node.units).strip()
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
                btNode = self._mdsaugdiag("MBI", "BTF")
                btTime, bt = btNode.getDimensionAt(0).data(), btNode.data()
                self._btaxv = interp1d(btTime, bt)(self.getTimeBase())
                self._defaultUnits["_btaxv"] = 'T'
            except AttributeError:
                raise ValueError("data retrieval failed.")
        return self._btaxv.copy()

    def getBtPla(self):
        """returns on-axis plasma toroidal field.

        Raises:
            NotImplementedError: Not implemented on ASDEX-Upgrade reconstructions.
        """
        if self._btpla is None:
            try:
                Rmag = self.getMagR().copy()
                self._btpla = np.array([
                    interp1d(pfl, fpol, bounds_error=False)(psi).item()
                    for pfl, fpol, psi in zip(self._fluxLabel[0], self._fpol, self._psiAxis)
                ]) / Rmag
                self._defaultUnits["_btaxv"] = 'T'
            except AttributeError:
                raise ValueError("data retrieval failed.")
        raise NotImplementedError("self.getBtPla not implemented.")

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
                ipCalcNode = self._mdsaugdiag("FPC", "IpiFP")
                ipTime, ip = ipCalcNode.getDimensionAt(0).data(), ipCalcNode.data()
                self._IpCalc = interp1d(ipTime, ip)(self.getTimeBase())
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
                ipMeasNode = self._mdsaugdiag("MAG", "Ipa")
                ipTime, ip = ipMeasNode.getDimensionAt(0).data(), ipMeasNode.data()
                self._IpMeas = interp1d(ipTime, ip)(self.getTimeBase())
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
        if self._Jp is None:
            try:
                JpNode = self._mdsaugdiag(self._tree, "CDM")
                self._Jp = JpNode.data()
                self._defaultUnits["_Jp"] = str(JpNode.units).strip()
            except AttributeError:
                raise ValueError("data retrieval failed.")
        return self._Jp.copy()

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
                betapNode = self._mdsaugdiag(self._treessq, "betpol")
                self._betap = betapNode.data()
                # I do not have the slightest idea why I must do this, I just know that I must do it.
                _betaptime = self._MDSTree.get(f"dim_of(augdiag({self._shot},'{self._treessq}','Zmag','{self._experiment}',{self._edition}))").data()
                mask = np.in1d(_betaptime.astype(self.getTimeBase().dtype), self.getTimeBase())
                _betaptime, self._betap = _betaptime[mask], self._betap[mask]
                self._defaultUnits["_betap"] = str(betapNode.units).strip()
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
                LiNode = self._mdsaugdiag(self._tree, "li")
                self._Li = LiNode.data()
                self._defaultUnits["_Li"] = str(LiNode.units).strip()
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
                WMHDNode = self._mdsaugdiag(self._treessq, "Wmhd")
                self._WMHD = WMHDNode.data()
                # I do not have the slightest idea why I must do this, I just know that I must do it.
                _WMHDtime = self._MDSTree.get(f"dim_of(augdiag({self._shot},'{self._treessq}','Wmhd','{self._experiment}',{self._edition}))").data()
                mask = np.in1d(_WMHDtime.astype(self.getTimeBase().dtype), self.getTimeBase())
                _WMHDtime, self._WMHD = _WMHDtime[mask], self._WMHD[mask]
                self._defaultUnits["_WMHD"] = str(WMHDNode.units).strip()
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
                    temp = self._mdsaugdiag("MBI", "BTFABB")
                    BCentrNode = temp.data()
                    self._BCentr = BCentrNode[
                        self._getNearestIdx(
                            self.getTimeBase(), temp.getDimensionAt().data()
                        )
                    ]
                    self._defaultUnits["_BCentr"] = str(BCentrNode.units).strip()
                except:
                    temp = self._mdsaugdiag("MBI", "BTF")
                    BCentrNode = temp.data()
                    self._BCentr = BCentrNode[
                        self._getNearestIdx(
                            self.getTimeBase(), temp.getDimensionAt().data()
                        )
                    ]
                    self._defaultUnits["_BCentr"] = str(BCentrNode.units).strip()
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
        return self._RCentr

    def getLowerXpoint(self):
        """Returns (R, Z) values of the lower x-points as a function of time

        Returns:
            R, Z: of the lower X-point as a function of time
        """
        if self._xlow is None:
            try:
                self._xlow = np.zeros((2, self._time.size))
                self._xlow[0, :] = self._mdsaugdiag(self._treessq, "Rxpu")
                self._xlow[1, :] = self._mdsaugdiag(self._treessq, "Zxpu")
                # I do not have the slightest idea why I must do this, I just know that I must do it.
                _xlowtime = self._MDSTree.get(f"dim_of(augdiag({self._shot},'{self._treessq}','Rxpu','{self._experiment}',{self._edition}))").data()
                mask = np.in1d(_xlowtime.astype(self.getTimeBase().dtype), self.getTimeBase())
                _xlowtime, self._xlow = _xlowtime[mask], self._xlow[mask]
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
                self._xup = np.zeros((2, self._time.size))
                self._xup[0, :] = self._mdsaugdiag(self._treessq, "Rxpo")
                self._xup[1, :] = self._mdsaugdiag(self._treessq, "Zxpo")
                # I do not have the slightest idea why I must do this, I just know that I must do it.
                _xuptime = self._MDSTree.get(f"dim_of(augdiag({self._shot},'{self._treessq}','Rxpo','{self._experiment}',{self._edition}))").data()
                mask = np.in1d(_xuptime.astype(self.getTimeBase().dtype), self.getTimeBase())
                _xuptime, self._xup = _xuptime[mask], self._xup[mask]
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
            self._currentSign = -1 if numpy.mean(self.getIpMeas()) > 1e5 else 1
        return self._currentSign

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

    # def getSSQ(self, inp, **kwargs):
    #     """returns single value quantities in the case SV file doesn't exist
    #     and coniditions the data in a way that is expected from a dd SV
    #     shotfile. This seamlessly hides the lack of an SV file.
    #
    #     Returns:
    #         signal (dd.signal Object): corresponding data
    #
    #     Raises:
    #         ValueError: if module cannot retrieve data from the AUG AFS system.
    #     """
    #     if self._SSQ is None:
    #         try:
    #             SSQnameNode = self._MDSTree("SSQnam", calibrated=False)
    #             # create a dict mapping the various quantities to positions in the in the data array
    #             self._SSQname = SSQnameNode.data
    #             try:
    #                 self._SSQname = scipy.char.strip(
    #                     SSQnameNode.data.view("S" + str(SSQnameNode.data.shape[1]))
    #                 )  # concatenate and strip blanks
    #             except ValueError:
    #                 self._SSQname = scipy.char.strip(
    #                     SSQnameNode.data.T.view("S" + str(SSQnameNode.data.shape[0]))
    #                 )  # concatenate and strip blanks
    #
    #             self._SSQname = self._SSQname[
    #                 self._SSQname != ""
    #             ]  # remove empty entries
    #             self._SSQname = dict(
    #                 zip(self._SSQname, scipy.arange(self._SSQname.shape[0]))
    #             )  # zip the dict together
    #
    #             self._SSQ = self._MDSTree("SSQ").data
    #
    #         except (PyddError, AttributeError):
    #             raise ValueError("data retrieval failed.")
    #
    #     if inp == "rays":
    #         data = self._SSQ[
    #             : self._timeidxend, self._SSQname["rays015"] : self._SSQname["rays000"]
    #         ]  # really hackish. This line might break at some point
    #         signal = dd.signalGroup(inp, " ", data)
    #     else:
    #         try:
    #             signal = dd.signal(
    #                 inp, " ", self._SSQ[: self._timeidxend, self._SSQname[inp]]
    #             )
    #
    #         except KeyError:
    #             raise ValueError("data retrieval failed.")
    #     return signal

    # class YGCAUGInterface(object):

    # ============================================================================================================
    #
    #                     VESSEL OUTLINE HARDCODE VALUES DUE TO ASDEX UPGRADE INCONSISTENCIES
    #
    # ============================================================================================================

    # Rather than use another dependency in code, this stores all the necessary interfacing from the data structure
    # the only necessary implementation is the data handler (dd python package)

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
                _shot = self._ygc_shotfiles[2]
            else:
                _shot = self._ygc_shot
            xvctr = self._mdsaugvessel(_shot, "YGC", "RrGC")
            yvctr = self._mdsaugvessel(_shot, "YGC", "zzGC")
            nvctr = self._mdsaugvessel(_shot, "YGC", "inxbeg")
            nvctr = nvctr.data().astype(int) - 1

        except (AttributeError):
            raise ValueError("data retrieval failed.")

        except:
            raise ValueError("data load failed.")

        return xvctr.data(), yvctr.data(), nvctr

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


# if _has_MDS:
#     AUGVessel = YGCAUGInterface()  # import setting necessary to get the vacuum vessel


# class AUGDDDataProp(AUGDDData, PropertyAccessMixin):
#     """AUGDDData with the PropertyAccessMixin added to enable property-style
#     access. This is good for interactive use, but may drag the performance down.
#     """
#
#     pass
