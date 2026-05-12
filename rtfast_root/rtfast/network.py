"""
This program comprises the neural network structure used as a base for the
rtfast emulator.

The reflection spectrum is now produced by a FiLM-MLP emulator that
predicts the spectrum directly on a user-supplied energy grid (built-in
interpolation) instead of the previous PCA + ensemble-on-fixed-grid
approach. The surrounding physics in `RTFAST.forward` (lensing, redshift,
comptonised continuum, tbabs absorption) is unchanged and is identical to
that of the original fortran model.

Architectural building blocks for the emulator (`RFFFeaturizer`,
`FiLM`, `FiLMMLPBlock`, `TrendHead`, `FiLM_MLP_Emulator`) are defined in
this module so that callers can instantiate the network from scratch if
they wish; the runtime `RTFAST` wrapper itself uses the `torch.export`
exported program produced by `export_emulator.py`.
"""
from itertools import batched
import math
import sys
from sys import platform
import os
import ctypes as ct
from typing import Optional, Tuple

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from joblib import load
from scipy.interpolate import interp1d, RegularGridInterpolator
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from importlib.resources import files

emudir = files("rtfast.models")
scalerdir = files("rtfast.scalers")
fortrandir = files("rtfast.fortran")

type_double_p = ct.POINTER(ct.c_double)

if platform == "darwin":
    lensing = ct.cdll.LoadLibrary(str(fortrandir.joinpath("lensing.dylib")))
elif platform == "linux" or sys.platform == "linux2":
    lensing = ct.cdll.LoadLibrary(str(fortrandir.joinpath("lensing.so")))
else:
    raise RuntimeError("Unsupported platform for Fortran library loading.")

get_lens = lensing.getlens_
get_lens.argtypes = [type_double_p, type_double_p, type_double_p, type_double_p]
get_lens.restype = None

import ndspec.XspecInterface as XSModels

def nthcomp(ear, params):
    pass

def tbabs(ear, params):
    pass

lib = XSModels.FortranInterface()

lib.add_model(nthcomp, symbol="donthcomp_")

headas_path = os.environ.get("HEADAS")
if platform == "linux" or platform == "linux2":
    lib_path = headas_path + f"/../Xspec/{os.path.basename(headas_path)}/lib/libXSFunctions.so"
elif platform == "darwin":
    lib_path = headas_path + f"/../Xspec/{os.path.basename(headas_path)}/lib/libXSFunctions.dylib"
pars_path =  headas_path + f"/../Xspec/src/manager/model.dat" 

lib_tbabs = XSModels.CInterface(lib_path, pars_path)
lib_tbabs.add_model(tbabs)


# =============================================================================
# Constants
# =============================================================================

LN10 = math.log(10.0)


# =============================================================================
# FiLM-MLP architecture
#
# These classes mirror the implementation in the training programs. They are
# kept here so this module fully documents the architecture; the runtime 
# wrapper below uses an `torch.export`-ed program.
# =============================================================================

class RFFFeaturizer(nn.Module):
    """
    Random Fourier Features featurizer with frequency annealing.
    forward(x) -> [B, L, 2*bands + 2]
    """
    def __init__(self, bands: int = 128, f_max: float = 64.0, device=None):
        super().__init__()
        self.bands = int(bands)
        self.f_max = float(f_max)
        if self.bands > 0:
            freqs = torch.empty(self.bands, device=device).uniform_(0.0, self.f_max)
            self.register_buffer("omega_base", (2.0 * math.pi * freqs).view(1, 1, self.bands))
        else:
            self.register_buffer("omega_base", None)
        self.register_buffer("freq_scale", torch.tensor(1.0))

    @property
    def out_dim(self) -> int:
        return 2 * max(self.bands, 0) + 2

    def set_freq_scale(self, s: float):
        if self.omega_base is None:
            self.freq_scale = torch.tensor(float(s), device=self.freq_scale.device)
        else:
            self.freq_scale = torch.tensor(float(s), device=self.omega_base.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.view(-1, 1)
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        if self.bands > 0 and self.omega_base is not None:
            omega = self.omega_base * self.freq_scale
            arg = x * omega
            cos = torch.cos(arg)
            sin = torch.sin(arg)
            feats = torch.cat([cos, sin, x, torch.ones_like(x)], dim=-1)
        else:
            feats = torch.cat([x, torch.ones_like(x)], dim=-1)
        return feats


class FiLM(nn.Module):
    def __init__(self, theta_dim: int, width: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(theta_dim, hidden), nn.GELU(),
                                 nn.Linear(hidden, 2 * width))

    def forward(self, theta):
        gamma, beta = self.net(theta).chunk(2, dim=-1)
        return gamma, beta


class FiLMMLPBlock(nn.Module):
    def __init__(self, d_model: int, theta_dim: int, ffn_mult: int = 4, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.cond = FiLM(theta_dim, d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ffn_mult * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_mult * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, theta: torch.Tensor):
        y = self.norm(x)
        gamma, beta = self.cond(theta)
        y = y * gamma.unsqueeze(1) + beta.unsqueeze(1)
        return x + self.ff(y)


class TrendHead(nn.Module):
    """Predicts a*x + b in the (normalized) input x domain."""
    def __init__(self, theta_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(theta_dim, hidden), nn.GELU(),
                                 nn.Linear(hidden, 2))

    def forward(self, theta):
        a, b = self.net(theta).chunk(2, dim=-1)
        return a, b


class FiLM_MLP_Emulator(nn.Module):
    """
    FiLM-MLP reflection emulator. Output: scaled log10 reflection spectrum
    of the same length as the input energy grid.
    """
    def __init__(self, theta_dim=10, rff_bands=128, rff_fmax=64.0,
                 d_model=256, n_blocks=6, ffn_mult=4, dropout=0.0,
                 use_trend_head: bool = False, trend_hidden: int = 64, device=None):
        super().__init__()
        self.featurizer = RFFFeaturizer(bands=rff_bands, f_max=rff_fmax, device=device)
        self.in_proj = nn.Linear(self.featurizer.out_dim, d_model)
        self.blocks = nn.ModuleList([
            FiLMMLPBlock(d_model, theta_dim, ffn_mult=ffn_mult, dropout=dropout)
            for _ in range(n_blocks)
        ])
        self.head = nn.Linear(d_model, 1)
        self.use_trend = bool(use_trend_head)
        if self.use_trend:
            self.trend = TrendHead(theta_dim, trend_hidden)

    def set_freq_scale(self, s: float):
        self.featurizer.set_freq_scale(s)

    def forward(self, theta, x, return_residual: bool = False):
        feats = self.featurizer(x)
        h = self.in_proj(feats)
        for blk in self.blocks:
            h = blk(h, theta)
        resid = self.head(h).squeeze(-1)

        if self.use_trend:
            a, b = self.trend(theta)
            trend = a * x + b
            y = resid + trend
            if return_residual:
                return y, resid
            return y
        return resid


# =============================================================================
# Energy-grid preprocessing
# =============================================================================

def _transform_x_for_log(x_row: np.ndarray, logx: bool,
                         x_log_floor: float, x_log_shift: float) -> np.ndarray:
    xr = x_row.astype(np.float64, copy=False)
    if not logx:
        return xr
    tiny = np.finfo(np.float64).tiny
    if x_log_shift > 0.0:
        z = xr + x_log_shift
        z = np.clip(z, max(x_log_floor, tiny), None)
        return np.log10(z)
    z = np.clip(xr, max(x_log_floor, tiny), None)
    return np.log10(z)


def _normalize_x(x_prepared: np.ndarray, mode: str,
                 global_minmax: Optional[Tuple[float, float]]) -> np.ndarray:
    if mode == "none":
        return x_prepared.astype(np.float32, copy=False)
    if mode == "per_row":
        xmin = float(np.min(x_prepared))
        xmax = float(np.max(x_prepared))
    elif mode == "global":
        if global_minmax is None:
            raise ValueError("x_norm='global' requires global_minmax to be set.")
        xmin, xmax = global_minmax
        xmin, xmax = np.log10(xmin), np.log10(xmax)
    else:
        raise ValueError(f"Unknown x_norm mode: {mode}")
    eps = 1e-12
    a = 2.0 / max(xmax - xmin, eps)
    return ((x_prepared - xmin) * a - 1.0).astype(np.float32, copy=False)


# =============================================================================
# RTFAST: physics-aware wrapper around the FiLM-MLP reflection emulator
# =============================================================================

class RTFAST(nn.Module):
    """
    Full rtfast model: FiLM-MLP reflection emulator + analytic comptonised
    continuum (nthcomp) + tbabs absorption + relativistic lensing.

    The reflection emulator predicts the spectrum directly on the user's
    energy grid via built-in interpolation, so no PCA round-trip or post-hoc
    `interp1d` is needed.

    Parameters
    ----------
    device : torch.device, optional
        Device on which the emulator runs. Defaults to CPU.
    exported_name : str, optional
        Filename of the exported program inside `rtfast.models`. Defaults
        to `"rtfast_emulator.pt"`.
    config_name : str, optional
        Filename of the companion preprocessing config inside
        `rtfast.scalers`. Defaults to `"rtfast_config.npz"`.

    Companion config file
    ---------------------
    The .npz config must contain at minimum:
      * `scaler_mean`, `scaler_std` : arrays of length M (training spectrum
        length) used to invert the y-scaling.
    Optional fields (defaults are used otherwise):
      * `logx` (0/1), `x_norm` ("none" / "per_row" / "global"),
        `x_log_floor`, `x_log_shift`, `global_xmin`, `global_xmax`.
    """

    DEFAULT_EXPORT_NAME = "rtfast_emulator.pt2"
    DEFAULT_CONFIG_NAME = "rtfast_config.npz"

    def __init__(self, device: torch.device = torch.device("cpu"),
                 exported_name: str = DEFAULT_EXPORT_NAME,
                 config_name: str = DEFAULT_CONFIG_NAME):
        super().__init__()
        self.device = device

        # ---- load the exported FiLM-MLP reflection emulator ----
        try:
            from numpy.core.multiarray import _reconstruct
            torch.serialization.add_safe_globals([_reconstruct, np.ndarray, np.dtype])
        except Exception:
            pass

        exp_path = str(emudir.joinpath(exported_name))
        self.exported_program = torch.export.load(exp_path)
        # `module()` gives a callable nn.Module-like object: forward(theta, x).
        self.emulator = self.exported_program.module()

        # ---- load the preprocessing / inverse-scaling config ----
        cfg_path = str(scalerdir.joinpath(config_name))
        self._load_emulator_config(cfg_path)

        # ---- define NN parameter selection / transform lists ----
        # These mirror ParamSelector in inference_emulator.py and are also
        # used to identify which raw rtdist parameters need to be flipped or
        # logged before the network sees them.
        self.pars_list = [0, 1, 2, 3, 4, 6, 7, 8, 9, 10]
        self.negatives = [3]
        self.logged = [0, 2, 3, 4, 10]

        # kept for backward compatibility with any code that read it directly
        self.powers = [0, 2, 3, 4, 10]

        # ---- prepare nthcomp ----
        self.nthcomp = lib.nthcomp
        self.nthcomp_reltrans_index = [6, 10, 5]
        self.nthcomp_index = [0, 1, 4]
        self.nthcomp_par_base = [2, 40, 0.05, 1, 0, 1]

        # ---- prepare tbabs ----
        self.tbabs = lib_tbabs.tbabs

        # ---- internal energy grids (kept for normalisation / plotting) ----
        self.define_internal_egrid()
        self.define_normalisation_grid()

    # ------------------------------------------------------------------
    # Emulator config loader
    # ------------------------------------------------------------------
    def _load_emulator_config(self, cfg_path: str):
        # defaults match inference_emulator.py
        self.logx = False
        self.x_norm = "per_row"
        self.x_log_floor = 1e-12
        self.x_log_shift = 0.0
        self._global_minmax = None
        scaler_mean = None
        scaler_std = None

        with np.load(cfg_path, allow_pickle=False) as cfg:
            if "scaler_mean" in cfg.files:
                scaler_mean = np.asarray(cfg["scaler_mean"], dtype=np.float64)
            if "scaler_std" in cfg.files:
                scaler_std = np.asarray(cfg["scaler_std"], dtype=np.float64)
            if "logx" in cfg.files:
                self.logx = bool(cfg["logx"])
            if "x_norm" in cfg.files:
                self.x_norm = str(cfg["x_norm"])
            if "x_log_floor" in cfg.files:
                self.x_log_floor = float(cfg["x_log_floor"])
            if "x_log_shift" in cfg.files:
                self.x_log_shift = float(cfg["x_log_shift"])
            if self.x_norm == "global":
                if "global_xmin" not in cfg.files or "global_xmax" not in cfg.files:
                    raise RuntimeError(
                        "x_norm='global' requires 'global_xmin' and "
                        "'global_xmax' in the emulator config file."
                    )
                self._global_minmax = (float(cfg["global_xmin"]),
                                       float(cfg["global_xmax"]))

        if scaler_mean is None or scaler_std is None:
            raise RuntimeError(
                f"y-scaler statistics not found in {cfg_path}. The config "
                f"npz must contain 'scaler_mean' and 'scaler_std'."
            )

        if (np.all(scaler_mean == 0) & np.all(scaler_std == 1)):
            self.scaler_flag = False
        else:
            self.scaler_flag = True
            self.scaler_mean = torch.as_tensor(scaler_mean, dtype=torch.float32,
                                               device=self.device)
            self.scaler_std = torch.as_tensor(scaler_std, dtype=torch.float32,
                                            device=self.device)

    # ------------------------------------------------------------------
    # Internal grids
    # ------------------------------------------------------------------
    def define_internal_egrid(self):
        """
        Defines internal energy grid which RTFAST was built on. This is defined
        between 0.1 and 100.0. Extrapolating outside of this range is at the
        user's own peril.
        """
        Emin = 0.1
        Emax = 100.0
        ne = 1000
        egrid = np.zeros(ne, dtype=np.float32)
        for i in range(ne):
            egrid[i] = Emin * (Emax / Emin) ** (i / ne)
        self.internal_egrid = egrid[:-1]
        self.internal_egrid_edges = egrid
        return

    def define_normalisation_grid(self):
        """
        Defines energy grid for normalisation of the continuum (illuminating
        comptonised spectrum nthcomp).
        """
        nex = 2 ** 12
        Emin = 1e-2
        Emax = 3e3
        self.norm_egrid = np.zeros(nex, dtype=np.float32)
        for i in range(nex):
            self.norm_egrid[i] = Emin * (Emax / Emin) ** (i / nex)
        return

    # ------------------------------------------------------------------
    # Lensing / GR helpers
    # ------------------------------------------------------------------
    def dgsofac(self, a, h):
        """
        Calculates the blue shift experienced by a photon travelling from an
        on-axis point source to a distant, stationary observer (works for both
        prograde and retrograde spins).

        Parameters
        ----------
        a : float
            spin of the black hole.
        h : float
            height (in Rg) of the source over the black hole.
        """
        Dh = h ** 2 - 2 * h + a ** 2
        dgsofac = Dh / (h ** 2 + a ** 2)
        dgsofac = np.sqrt(dgsofac)
        return dgsofac

    def lensing_factor(self, a, h, muobs):
        a = ct.c_double(a)
        h = ct.c_double(h)
        muobs = ct.c_double(muobs)
        lens = ct.c_double(1)
        a_p = ct.byref(a)
        h_p = ct.byref(h)
        muobs_p = ct.byref(muobs)
        lens_p = ct.byref(lens)
        get_lens(a_p, h_p, muobs_p, lens_p)
        return lens_p._obj.value

    # ------------------------------------------------------------------
    # Reflection-emulator preprocessing
    # ------------------------------------------------------------------
    def pars_shift(self, pars):
        """
        Converts rtdist parameters into FiLM-MLP-friendly form: selects the
        10 indices the network was trained on, negates index 3, and log10s
        indices {0, 2, 3, 4, 10}. Returns a torch float32 tensor of shape
        ``[10]`` (or ``[B, 10]`` if a batch is supplied).
        """
        if pars.ndim > 1:
            pars = pars[:,self.pars_list]
        else:
            pars = pars[self.pars_list]
        for i, parameter in enumerate(self.pars_list):
            if parameter in self.negatives:
                if pars.ndim > 1:
                    pars[:, i] = -pars[:, i]
                else:
                    pars[i] = -pars[i]
            if parameter in self.logged:
                if pars.ndim > 1:
                    pars[:, i] = np.log10(np.clip(pars[:, i], 1e-30, None))
                else:
                    pars[i] = np.log10(np.clip(pars[i], 1e-30, None))
        return torch.as_tensor(pars, dtype=torch.float32, device=self.device)

    def _prepare_x(self, egrid: np.ndarray) -> torch.Tensor:
        """Apply the log/normalisation transforms used during training."""
        x_prep = _transform_x_for_log(np.asarray(egrid, dtype=np.float64),
                                      self.logx, self.x_log_floor, self.x_log_shift)
        x_normed = _normalize_x(x_prep, self.x_norm, self._global_minmax)
        return torch.as_tensor(x_normed, dtype=torch.float32,
                               device=self.device).unsqueeze(0)  # [1, L]

    def _inverse_scale_y(self, y_scaled: torch.Tensor) -> np.ndarray:
        """Convert scaled log10 output to a linear spectrum (1D numpy)."""
        if self.scaler_flag == True:
            ylog = y_scaled * self.scaler_std + self.scaler_mean
        else:
            ylog = y_scaled
        spectrum = torch.pow(10.0, ylog)
        spectrum = spectrum.detach().cpu().numpy()
        if spectrum.ndim == 2 and spectrum.shape[0] == 1:
            spectrum = spectrum[0]
        return spectrum

    # ------------------------------------------------------------------
    # Reflection-spectrum prediction
    # ------------------------------------------------------------------
    def reflection_spectrum_prediction(self, egrid, NN_pars, batched=False):
        """
        Predict the reflection spectrum on the supplied energy grid.

        Parameters
        ----------
        egrid : np.ndarray, shape (M,)
            User-supplied energy grid.
        NN_pars : torch.Tensor, shape (10,)
            Already-processed network parameters (output of
            ``pars_shift``).

        Returns
        -------
        np.ndarray, shape (M,)
            Linear-space reflection spectrum on ``egrid``.
        """
        if NN_pars.dim() == 1:
            theta = NN_pars.unsqueeze(0).to(self.device)
        else:
            theta = NN_pars.to(self.device)

        x = self._prepare_x(self.internal_egrid_edges)
        with torch.no_grad():
            if batched == True:
                x = torch.tile(x, (theta.shape[0], 1))
            y_scaled = self.emulator(theta, x)
        
        if self.scaler_flag:
            ylog = y_scaled * self.scaler_std + self.scaler_mean
        else:
            ylog = y_scaled
        
        spectrum = self._inverse_scale_y(ylog)

        if batched == True:
            f = RegularGridInterpolator(np.tile(self.internal_egrid_edges, (theta.shape[0], 1)), 
                                        spectrum, bounds_error=False, fill_value="extrapolate")
            x_new = np.tile(egrid, (theta.shape[0], 1))
            spectrum = f(x_new)
        else:
            f = interp1d(self.internal_egrid_edges, spectrum, bounds_error=False, 
                         fill_value="extrapolate")
            spectrum = f(egrid)

        spectrum_binned = 0.5 * (spectrum[:-1] + spectrum[1:])
        return spectrum_binned

    # ------------------------------------------------------------------
    # Continuum
    # ------------------------------------------------------------------
    def calculate_normalisation(self, pars):
        earx = self.norm_egrid                                # length nex (edges)
        norm_dens = self.nthcomp(earx, pars)                  # length nex-1 (density at midpoints)
        bin_widths = np.diff(earx.astype(np.float64))         # length nex-1
        Icomp = 0.0
        for i in range(len(norm_dens)):
            E_mid = 0.5 * (earx[i] + earx[i+1])
            if 0.1 <= E_mid <= 1e3:
                N_i = norm_dens[i] * bin_widths[i]            # bin-integrate density → photons
                Icomp += E_mid * N_i                          # = E_mid × N_i, matches Fortran convention
        return Icomp

    def comptonized_continuum(self, egrid, pars, logxi, logne):
        egrid, pars = egrid.astype(np.float32), pars.astype(np.float32)
        comp = self.nthcomp(egrid, pars)
        Icomp = self.calculate_normalisation(pars)
        # incident flux in units [keV/cm^2/s]
        inc_flux = (10 ** (logne + logxi)) / (4.0 * np.pi * 1.602197e-9)
        # renormalise to correct local continuum
        get_norm_cont_local = inc_flux / Icomp / 1e20
        # return renormalised compton spectrum
        comp = comp * get_norm_cont_local / (10 ** (logxi + logne - 15))
        return comp

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
    def forward(self, egrid, theta, batched=False):
        if batched == False:
            # split input parameters into appropriate parameters for each component
            dgsofac = self.dgsofac(theta[1], theta[0])
            inc = theta[2]
            muobs = np.cos(inc * np.pi / 180)
            boost = theta[12]
            logxi = theta[7]
            logne = theta[9]
            z = theta[5]
            kTe = theta[10]
            lens = self.lensing_factor(theta[1], theta[0], muobs)
            norm = theta[-1]

            # redefine temperature to be observed electron temperature for the NN
            theta_NN = theta.copy()
            theta_NN[10] = (theta_NN[10] * dgsofac) / (1 + z)
            NN_pars = self.pars_shift(theta_NN)

            tbabs_pars = theta[11]
            nthcomp_pars = np.copy(self.nthcomp_par_base)
            nthcomp_pars[self.nthcomp_index] = theta[self.nthcomp_reltrans_index]
            nthcomp_pars[1] = kTe
            nthcomp_pars[4] = (1.0 / dgsofac) - 1.0

            # reflection: emulator predicts photon flux density (per keV) at
            # the points of egrid. Convert to per-bin integrated photon flux
            # via the trapezoidal rule so it can be summed with nthcomp's
            # bin-integrated output and multiplied by tbabs' similar output.
            refl_spect = self.reflection_spectrum_prediction(egrid, NN_pars)  # length M
            spectrum = np.abs(boost) * refl_spect                              # length M-1
            spectrum = spectrum.astype(np.float32)

            # add nthcomp component (already bin-integrated, length M-1)
            if boost >= 0:
                comp = self.comptonized_continuum(egrid, nthcomp_pars, logxi, logne)
                comp = lens * (dgsofac / (1 + z)) * comp
                spectrum += comp

            # multiply by tbabs absorption (per-bin transmission, length M-1)
            tbabs_res = self.tbabs(egrid.astype(np.float32),
                                np.array([tbabs_pars.astype(np.float32)], dtype=np.float32))
            spectrum *= tbabs_res
            spectrum = spectrum * norm
            return spectrum
        else:
            # thetas: numpy [B, 22]
            B = theta.shape[0]
            
            # ---- scalar-per-walker physics (vectorize over B) ----
            a    = theta[:, 1]
            h    = theta[:, 0]
            inc  = theta[:, 2]
            z    = theta[:, 5]
            logxi = theta[:, 7]
            logne = theta[:, 9]
            kTe  = theta[:, 10]
            boost = theta[:, 12]
            norm  = theta[:, -1]

            muobs = np.cos(inc * np.pi / 180)
            dgsofac = np.sqrt((h**2 - 2*h + a**2) / (h**2 + a**2))   # [B]
            
            # ---- lensing: this is the painful one ----
            # get_lens is a scalar ctypes call. Loop in Python, or write a 
            # vectorized Fortran wrapper. For B≈50 the loop is fine.
            lens = np.array([self.lensing_factor(a[i], h[i], muobs[i]) 
                            for i in range(B-1)])                     # [B]
            
            # ---- emulator: batched ----
            thetas_NN = theta.copy()
            thetas_NN[:, 10] = (thetas_NN[:, 10] * dgsofac) / (1 + z)
            NN_pars = self.pars_shift(thetas_NN)                     # [B, 10] torch
            with torch.inference_mode():
                refl_density = self.reflection_spectrum_prediction(egrid, NN_pars)  # [B, L]
            
            # ---- trapezoidal bin integration ----
            refl_per_bin = 0.5 * (refl_density[:, :-1] + refl_density[:, 1:])  # [B, M-1]
            spectrum = np.abs(boost)[:, None] * refl_per_bin
            
            # ---- nthcomp + tbabs: also scalar Fortran. Loop. ----
            for i in range(B-1):
                if boost[i] >= 0:
                    nthcomp_pars = np.copy(self.nthcomp_par_base)
                    nthcomp_pars[self.nthcomp_index] = theta[i, self.nthcomp_reltrans_index]
                    nthcomp_pars[1] = kTe[i]
                    nthcomp_pars[4] = (1.0 / dgsofac[i]) - 1.0

                    comp = self.comptonized_continuum(egrid, nthcomp_pars, logxi[i], logne[i])
                    comp = lens[i] * (dgsofac[i] / (1 + z[i])) * comp
                    spectrum[i] += comp

                tbabs_res = self.tbabs(egrid.astype(np.float32),
                            np.array([tbabs_pars.astype(np.float32)], dtype=np.float32))
                spectrum[i] *= tbabs_res

            spectrum *= norm[:, None]
            return spectrum
