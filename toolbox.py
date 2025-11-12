# toolbox.py>
import numpy as np
import xarray as xr
from scipy.optimize import fsolve

### Toolbox of atmospheric science/meterological funcations ###
## Compiled by Walter Shen
# updated October 2025

#########################
####### functions #######
#########################

def helloWorld():
    print("hello world!")

def es(T):
    """
    Calculate the saturation pressure of water vapor.
    
    Parameters:
    T (float): Temperature in Kelvin
    
    Returns:
    float: Saturation pressure in hPa
    
    Notes:
    Following (Bolton, 1980, Monthly Weather Review, 108, 1046-1053)
    """
    Pa_to_hPa = 100 # 100 Pa = 1 hPa; original formula outputs as Pa
    return 611.2*np.exp(17.67*(T-273)/(T-29.5)) / Pa_to_hPa

def T_d(p, r):
    """
    Calculate the dew point temperature given pressure and mixing ratio.
    
    Parameters:
    p (float): Pressure in hPa
    r (float): Vapor mixing ratio (kg/kg)
    
    Returns:
    float: Dew point temperature in Kelvin
    """
    # returned by matlabFunction(finverse(es(x)))
    x = r/18*29*p*100
    return (np.log(x*1.636125654450262e-3)*(5.9e+1/2.0)-4.82391e+3)/(np.log(x*1.636125654450262e-3)-1.767e+1)
    

def q_t(q_v, q_l):
    """
    Calculate total specific humidity
    
    Parameters:
    q_v (float): Specific humidity of water vapor (kg/kg)
    q_l (float): Specific humidity of liquid water (kg/kg)
    
    Returns:
    float: total specific humidity (kg/kg)

    """
    return q_l + q_v

def q_from_r(r):
    """
    Convert mixing ratio to specific humidity
    
    Parameters:
    r (float): mixing ratio (e.g. of water vapor) (kg/kg)
    
    Returns:
    q: absolute humidity (kg/kg)
    """
    return r/(1+r)

def r_from_q(q):
    """
    Convert specific humidity to mixing ratio
    
    Parameters:
    q: absolute humidity (kg/kg)
    
    Returns:
    r (float): mixing ratio (e.g. of water vapor) (kg/kg)
    """
    return q/(1-q)

def q_sat(T, p):
    """
    Saturation specific humidity
    
    Parameters:
    T (float): Temperature in Kelvin
    p (float): Pressure in hPa
    
    Returns:
    q_sat (float): Saturation specific humidity (kg/kg)
    """
    epsilon = 0.622 # Dalton's law of partial pressures; Rd/Rv
    q_sat = epsilon*es(T) / (p - (1-epsilon)*es(T))
    return q_sat
    
def r_sat(T, p):
    """
    Saturation mixing ratio
    
    Parameters:
    T (float): Temperature in Kelvin
    p (float): Pressure in hPa
    
    Returns:
    q_sat (float): Saturation specific humidity (kg/kg)
    """
    return r_from_q(q_sat(T, p))
    
def theta(T, p):
    """
    Calculate the potential temperature
    
    Parameters:
    T (float): Temperature in Kelvin
    p (float): Pressure in hPa
    
    Returns:
    float: potential temperature in Kelvin
    """
    p_ref = 1000 # reference pressure (hPa)
    Rd=287.04; # gas constant for dry air (J K-1 kg-1)
    Cpd=1005.7; # specific heat of dry air, constant pressure (J K-1 kg-1)
    theta = T*(p_ref/p)**(Rd/Cpd)
    return theta
    
def theta_e(T, p, q_v):
    """
    Calculate the equivalent potential temperature
    
    Parameters:
    T (float): Temperature in Kelvin
    p (float): Pressure in hPa
    q_v (float): Specific humidity of water vapor (kg/kg)
    
    Returns:
    float: Equivalent potential temperature in Kelvin
    
    Notes:
    using Alan Betts (1973) definition
    """
    L_v = 2.5e6 # latent heat of vaporization at 0C (J kg-1)
    Cpd=1005.7; # specific heat of dry air, constant pressure (J K-1 kg-1)
    
    theta_e = theta(T, p) * np.exp( (L_v*q_v) / (Cpd*T) )
    
    return theta_e

def theta_l(T, p, q_l):
    """
    Calculate the liquid potential temperature
    
    Parameters:
    T (float): Temperature in Kelvin
    p (float): Pressure in hPa
    q_l (float): Specific humidity of liquid water (kg/kg)
    
    Returns:
    float: Liquid potential temperature in Kelvin
    
    Notes:
    using Alan Betts (1973) definition
    """
    L_v = 2.5e6 # latent heat of vaporization at 0C (J kg-1)
    Cpd=1005.7; # specific heat of dry air, constant pressure (J K-1 kg-1)
    
    theta_l = theta(T, p) * np.exp( -(L_v*q_l) / (Cpd*T) )
    
    return theta_l

def theta_alpha(T, p, q_v, q_l, alpha):
    """
    Calculate the weighted potential temperature from Heus et. al (2008)
    
    Parameters:
    T (float): Temperature in Kelvin
    p (float): Pressure in hPa
    q_v (float): Specific humidity of water vapor (kg/kg)
    q_l (float): Specific humidity of liquid water (kg/kg)
    
    Returns:
    float: Weighted potential temperature in Kelvin

    """

    theta_alpha = alpha*theta_e(T, p, q_v) + (1-alpha)*theta_l(T, p, q_l)
    
    return theta_alpha

def T_v(T, p, q_v, q_l):
    """
    Calculate the virtual temperature.

    Parameters:
    T (float): Temperature in Kelvin
    p (float): Pressure in hPa
    q_v (float): Specific humidity of water vapor (kg/kg)
    q_l (float): Specific humidity of liquid water (kg/kg)

    Returns:
    float: virtual temperature in Kelvin
    
    Notes:
    see Stull (1988), Appendix D.
    """
    r_l = r_from_q(q_l)
    r_v = r_from_q(q_v)
    
    saturated = q_v > q_sat(T, p)
    T_v_if_sat = T * (1 + 0.61 * r_sat(T, p) - r_l)
    T_v_if_unsat = T * (1 + 0.61 * r_v) # unsaturated, r_L = 0, use r_v instead of r_sat
    
    T_v = xr.where(saturated, T_v_if_sat, T_v_if_unsat)
      
    return T_v

def theta_v(T, p, q_v, q_l):
    """
    Calculate the virtual potential temperature.

    Parameters:
    T (float): Temperature in Kelvin
    p (float): Pressure in hPa
    q_v (float): Specific humidity of water vapor (kg/kg)
    q_l (float): Specific humidity of liquid water (kg/kg)

    Returns:
    float: virtual potential temperature in Kelvin
    
    Notes:
    see Stull (1988), Appendix D.
    """
    r_l = r_from_q(q_l)
    r_v = r_from_q(q_v)
    
    saturated = q_v > q_sat(T, p)
    theta_v_if_sat = theta(T, p) * (1 + 0.61 * r_sat(T, p) - r_l)
    theta_v_if_unsat = theta(T, p) * (1 + 0.61 * r_v) # unsaturated, r_L = 0, use r_v instead of r_sat        
    
    theta_v = xr.where(saturated, theta_v_if_sat, theta_v_if_unsat)
     
    return theta_v

def T_for_theta_v_iso(q_t, theta_v_target, p):
    """
    Find value of T, given q_t, that has same theta_v as theta_v_target

    Parameters:
    q_t (float): Total water humidity of water (vapor + liquid) (kg/kg)
    theta_v_target (float): virtual potential temperature (isopleth level)
    p (float): Pressure in hPa

    Returns:
    float: T, such that theta_v(T, p, q_v, q_l) = theta_v_target
    """
    
    # function that takes in T, outputs theta_v
    def theta_v_given_T_minus_theta_v_target(given_T):
        q_v = np.minimum(q_t, q_sat(given_T, p))
        q_l = np.maximum(0, q_t - q_sat(given_T, p))
        return theta_v(given_T, p, q_v, q_l) - theta_v_target
        
    # fsolve to find the T that solves function_above(T)=theta_v
    T_for_theta_v_iso = fsolve(theta_v_given_T_minus_theta_v_target, 250)
    
    return T_for_theta_v_iso

def LWP_from_p_QN(p, q_l, zdim="z"):
    """
    Compute column liquid water path from pressure and mixing ratio.

    Parameters
    ----------
    p : xr.DataArray
        Pressure in mb/hPa with dims (time, z).
    QN : xr.DataArray
        Liquid water mixing ratio in KG/kg with dims (time, z, y, x).
        NOTE: If QN includes ice (water+ice), this yields total condensate path (TWP).
    zdim : str
        Name of the vertical dimension (default "z").

    Returns
    -------
    lwp : xr.DataArray
        Column path in kg m^-2 with dims (time, y, x).
    lwp_mean_t : xr.DataArray
        Domain-mean time series (time) in kg m^-2.
    """
    g = 9.80665  # m s^-2

    # --- unit conversions ---
    # p: mb -> Pa
    p_pa = p * 100.0
    # QN: g/kg -> kg/kg
    ql = q_l #/ 1000.0

    # Broadcast pressure to QN shape
    ql, p_pa = xr.broadcast(ql, p_pa)

    # dp along vertical
    dp = p_pa.diff(zdim)

    # trapezoid average of ql to interfaces
    ql_mid = 0.5 * (
        ql.isel({zdim: slice(None, -1)}) + ql.isel({zdim: slice(1, None)})
    )

    # ensure positive (works whether z increases upward or downward)
    dp = dp.astype("float64")
    dp_abs = abs(dp)

    print(dp)
    
    # LWP = (1/g) * sum( ql_mid * dp )
    lwp = (ql_mid * dp_abs).sum(zdim) / g
    lwp = lwp.assign_attrs(
        long_name="Column liquid water path (pressure integral)",
        units="kg m-2"
    )

    # domain-mean time series
    space_dims = [d for d in lwp.dims if d not in ("time",)]
    lwp_mean_t = lwp.mean(dim=space_dims).assign_attrs(units="kg m-2")

    return lwp, lwp_mean_t

def decode_xyz(xyz):
    """
    Decode packed integer particle positions from LPDM output.

    Each packed value is: packed = x*1e6 + y*1e3 + z (format format XXX,YYY,ZZZ)

    Parameters
    ----------
    xyz : array_like
        Array of packed integer positions (e.g., int32 or int64).
        format XXX,YYY,ZZZ

    Returns
    -------
    x, y, z : ndarray
        Arrays of integer coordinates with the same shape as xyz.
        returns XXX and YYY and ZZZ
    """
    xyz = np.asarray(xyz, dtype=np.int64)
    x = np.floor_divide(xyz, 1e6)
    y = np.floor_divide(xyz - x * 1e6, 1e3)
    z = np.mod(xyz, x * 1e6 + y * 1e3)
    return x, y, z


def bivariate_fit(xi, yi, dxi, dyi, ri=0.0, b0=1.0, maxIter=1e6):
    ### York Fit function
    # https://gist.github.com/mikkopitkanen/ce9cd22645a9e93b6ca48ba32a3c85d0
    
    """Function for fitting York, 2004, bivariate fit.

    Copyright (C) 2019 Mikko Pitkanen

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
    """
    
    """Make a linear bivariate fit to xi, yi data using York et al. (2004).

    This is an implementation of the line fitting algorithm presented in:
    York, D et al., Unified equations for the slope, intercept, and standard
    errors of the best straight line, American Journal of Physics, 2004, 72,
    3, 367-375, doi = 10.1119/1.1632486

    See especially Section III and Table I. The enumerated steps below are
    citations to Section III

    Parameters:
      xi, yi      x and y data points
      dxi, dyi    errors for the data points xi, yi
      ri          correlation coefficient for the weights
      b0          initial guess b
      maxIter     float, maximum allowed number of iterations

    Returns:
      a           y-intercept, y = a + bx
      b           slope
      S           goodness-of-fit estimate
      sigma_a     standard error of a
      sigma_b     standard error of b

    Usage:
    [a, b] = bivariate_fit( xi, yi, dxi, dyi, ri, b0, maxIter)

    """
    # (1) Choose an approximate initial value of b
    b = b0

    # (2) Determine the weights wxi, wyi, for each point.
    wxi = 1.0 / dxi**2.0
    wyi = 1.0 / dyi**2.0

    alphai = (wxi * wyi)**0.5
    b_diff = 999.0

    # tolerance for the fit, when b changes by less than tol for two
    # consecutive iterations, fit is considered found
    tol = 1.0e-8

    # iterate until b changes less than tol
    iIter = 1
    while (abs(b_diff) >= tol) & (iIter <= maxIter):

        b_prev = b

        # (3) Use these weights wxi, wyi to evaluate Wi for each point.
        Wi = (wxi * wyi) / (wxi + b**2.0 * wyi - 2.0*b*ri*alphai)

        # (4) Use the observed points (xi ,yi) and Wi to calculate x_bar and
        # y_bar, from which Ui and Vi , and hence betai can be evaluated for
        # each point
        x_bar = np.sum(Wi * xi) / np.sum(Wi)
        y_bar = np.sum(Wi * yi) / np.sum(Wi)

        Ui = xi - x_bar
        Vi = yi - y_bar

        betai = Wi * (Ui / wyi + b*Vi / wxi - (b*Ui + Vi) * ri / alphai)

        # (5) Use Wi, Ui, Vi, and betai to calculate an improved estimate of b
        b = np.sum(Wi * betai * Vi) / np.sum(Wi * betai * Ui)

        # (6) Use the new b and repeat steps (3), (4), and (5) until successive
        # estimates of b agree within some desired tolerance tol
        b_diff = b - b_prev

        iIter += 1

    # (7) From this final value of b, together with the final x_bar and y_bar,
    # calculate a from
    a = y_bar - b * x_bar

    # Goodness of fit
    S = np.sum(Wi * (yi - b*xi - a)**2.0)

    # (8) For each point (xi, yi), calculate the adjusted values xi_adj
    xi_adj = x_bar + betai

    # (9) Use xi_adj, together with Wi, to calculate xi_adj_bar and thence ui
    xi_adj_bar = np.sum(Wi * xi_adj) / np.sum(Wi)
    ui = xi_adj - xi_adj_bar

    # (10) From Wi , xi_adj_bar and ui, calculate sigma_b, and then sigma_a
    # (the standard uncertainties of the fitted parameters)
    sigma_b = np.sqrt(1.0 / np.sum(Wi * ui**2))
    sigma_a = np.sqrt(1.0 / np.sum(Wi) + xi_adj_bar**2 * sigma_b**2)

    # calculate covariance matrix of b and a (York et al., Section II)
    cov = -xi_adj_bar * sigma_b**2
    # [[var(b), cov], [cov, var(a)]]
    cov_matrix = np.array(
        [[sigma_b**2, cov], [cov, sigma_a**2]])

    if iIter <= maxIter:
        return a, b, S, cov_matrix
    else:
        print("bivariate_fit.py exceeded maximum number of iterations, " +
              "maxIter = {:}".format(maxIter))
        return np.nan, np.nan, np.nan, np.nan