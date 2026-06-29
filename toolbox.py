# toolbox.py>
import numpy as np
import xarray as xr
from scipy.optimize import fsolve
from scipy.special import gamma
import inspect
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

### Toolbox of atmospheric science/meterological funcations ###
## Compiled by Walter Shen ##
# Updated 2026-02

##############################################
####### General Meteological Constants #######
##############################################

# todo add some basic constants

##############################################
####### General Helper Functions #######
##############################################

def describe_vars(objs):
    """
    Pass a list like [X, Y, V_QT] and try to infer names from the caller's namespace.
    Works best in notebooks where variables live in user_ns.
    
    # usage
    describe_vars([X, Y, V_QT, time])
    """
    
    frame = inspect.currentframe().f_back
    namespaces = [frame.f_locals, frame.f_globals]
    try:
        ip = get_ipython()
        if ip is not None:
            namespaces.append(ip.user_ns)
    except NameError:
        pass

    for i, obj in enumerate(objs):
        names = []
        for ns in namespaces:
            names = [k for k, v in ns.items() if v is obj]
            if names:
                break
        name = names[0] if names else f"var{i}"
        if len(names) > 1:
            name = f"{name} (aliases: {names})"

        t = type(obj).__name__
        shape = getattr(obj, "shape", None)
        if shape is None:
            try: shape = np.shape(obj)
            except Exception: shape = None
        dtype = getattr(obj, "dtype", None)

        print(f"{name}: type={t}"
              + (f", shape={shape}" if shape is not None else "")
              + (f", dtype={dtype}" if dtype is not None else ""))

# toolbox.py

# Source - https://stackoverflow.com/a/60007513
# Posted by PetSven
# Retrieved 2026-04-28, License - CC BY-SA 4.0
cm_data = [[0.2422, 0.1504, 0.6603],
[0.2444, 0.1534, 0.6728],
[0.2464, 0.1569, 0.6847],
[0.2484, 0.1607, 0.6961],
[0.2503, 0.1648, 0.7071],
[0.2522, 0.1689, 0.7179],
[0.254, 0.1732, 0.7286],
[0.2558, 0.1773, 0.7393],
[0.2576, 0.1814, 0.7501],
[0.2594, 0.1854, 0.761],
[0.2611, 0.1893, 0.7719],
[0.2628, 0.1932, 0.7828],
[0.2645, 0.1972, 0.7937],
[0.2661, 0.2011, 0.8043],
[0.2676, 0.2052, 0.8148],
[0.2691, 0.2094, 0.8249],
[0.2704, 0.2138, 0.8346],
[0.2717, 0.2184, 0.8439],
[0.2729, 0.2231, 0.8528],
[0.274, 0.228, 0.8612],
[0.2749, 0.233, 0.8692],
[0.2758, 0.2382, 0.8767],
[0.2766, 0.2435, 0.884],
[0.2774, 0.2489, 0.8908],
[0.2781, 0.2543, 0.8973],
[0.2788, 0.2598, 0.9035],
[0.2794, 0.2653, 0.9094],
[0.2798, 0.2708, 0.915],
[0.2802, 0.2764, 0.9204],
[0.2806, 0.2819, 0.9255],
[0.2809, 0.2875, 0.9305],
[0.2811, 0.293, 0.9352],
[0.2813, 0.2985, 0.9397],
[0.2814, 0.304, 0.9441],
[0.2814, 0.3095, 0.9483],
[0.2813, 0.315, 0.9524],
[0.2811, 0.3204, 0.9563],
[0.2809, 0.3259, 0.96],
[0.2807, 0.3313, 0.9636],
[0.2803, 0.3367, 0.967],
[0.2798, 0.3421, 0.9702],
[0.2791, 0.3475, 0.9733],
[0.2784, 0.3529, 0.9763],
[0.2776, 0.3583, 0.9791],
[0.2766, 0.3638, 0.9817],
[0.2754, 0.3693, 0.984],
[0.2741, 0.3748, 0.9862],
[0.2726, 0.3804, 0.9881],
[0.271, 0.386, 0.9898],
[0.2691, 0.3916, 0.9912],
[0.267, 0.3973, 0.9924],
[0.2647, 0.403, 0.9935],
[0.2621, 0.4088, 0.9946],
[0.2591, 0.4145, 0.9955],
[0.2556, 0.4203, 0.9965],
[0.2517, 0.4261, 0.9974],
[0.2473, 0.4319, 0.9983],
[0.2424, 0.4378, 0.9991],
[0.2369, 0.4437, 0.9996],
[0.2311, 0.4497, 0.9995],
[0.225, 0.4559, 0.9985],
[0.2189, 0.462, 0.9968],
[0.2128, 0.4682, 0.9948],
[0.2066, 0.4743, 0.9926],
[0.2006, 0.4803, 0.9906],
[0.195, 0.4861, 0.9887],
[0.1903, 0.4919, 0.9867],
[0.1869, 0.4975, 0.9844],
[0.1847, 0.503, 0.9819],
[0.1831, 0.5084, 0.9793],
[0.1818, 0.5138, 0.9766],
[0.1806, 0.5191, 0.9738],
[0.1795, 0.5244, 0.9709],
[0.1785, 0.5296, 0.9677],
[0.1778, 0.5349, 0.9641],
[0.1773, 0.5401, 0.9602],
[0.1768, 0.5452, 0.956],
[0.1764, 0.5504, 0.9516],
[0.1755, 0.5554, 0.9473],
[0.174, 0.5605, 0.9432],
[0.1716, 0.5655, 0.9393],
[0.1686, 0.5705, 0.9357],
[0.1649, 0.5755, 0.9323],
[0.161, 0.5805, 0.9289],
[0.1573, 0.5854, 0.9254],
[0.154, 0.5902, 0.9218],
[0.1513, 0.595, 0.9182],
[0.1492, 0.5997, 0.9147],
[0.1475, 0.6043, 0.9113],
[0.1461, 0.6089, 0.908],
[0.1446, 0.6135, 0.905],
[0.1429, 0.618, 0.9022],
[0.1408, 0.6226, 0.8998],
[0.1383, 0.6272, 0.8975],
[0.1354, 0.6317, 0.8953],
[0.1321, 0.6363, 0.8932],
[0.1288, 0.6408, 0.891],
[0.1253, 0.6453, 0.8887],
[0.1219, 0.6497, 0.8862],
[0.1185, 0.6541, 0.8834],
[0.1152, 0.6584, 0.8804],
[0.1119, 0.6627, 0.877],
[0.1085, 0.6669, 0.8734],
[0.1048, 0.671, 0.8695],
[0.1009, 0.675, 0.8653],
[0.0964, 0.6789, 0.8609],
[0.0914, 0.6828, 0.8562],
[0.0855, 0.6865, 0.8513],
[0.0789, 0.6902, 0.8462],
[0.0713, 0.6938, 0.8409],
[0.0628, 0.6972, 0.8355],
[0.0535, 0.7006, 0.8299],
[0.0433, 0.7039, 0.8242],
[0.0328, 0.7071, 0.8183],
[0.0234, 0.7103, 0.8124],
[0.0155, 0.7133, 0.8064],
[0.0091, 0.7163, 0.8003],
[0.0046, 0.7192, 0.7941],
[0.0019, 0.722, 0.7878],
[0.0009, 0.7248, 0.7815],
[0.0018, 0.7275, 0.7752],
[0.0046, 0.7301, 0.7688],
[0.0094, 0.7327, 0.7623],
[0.0162, 0.7352, 0.7558],
[0.0253, 0.7376, 0.7492],
[0.0369, 0.74, 0.7426],
[0.0504, 0.7423, 0.7359],
[0.0638, 0.7446, 0.7292],
[0.077, 0.7468, 0.7224],
[0.0899, 0.7489, 0.7156],
[0.1023, 0.751, 0.7088],
[0.1141, 0.7531, 0.7019],
[0.1252, 0.7552, 0.695],
[0.1354, 0.7572, 0.6881],
[0.1448, 0.7593, 0.6812],
[0.1532, 0.7614, 0.6741],
[0.1609, 0.7635, 0.6671],
[0.1678, 0.7656, 0.6599],
[0.1741, 0.7678, 0.6527],
[0.1799, 0.7699, 0.6454],
[0.1853, 0.7721, 0.6379],
[0.1905, 0.7743, 0.6303],
[0.1954, 0.7765, 0.6225],
[0.2003, 0.7787, 0.6146],
[0.2061, 0.7808, 0.6065],
[0.2118, 0.7828, 0.5983],
[0.2178, 0.7849, 0.5899],
[0.2244, 0.7869, 0.5813],
[0.2318, 0.7887, 0.5725],
[0.2401, 0.7905, 0.5636],
[0.2491, 0.7922, 0.5546],
[0.2589, 0.7937, 0.5454],
[0.2695, 0.7951, 0.536],
[0.2809, 0.7964, 0.5266],
[0.2929, 0.7975, 0.517],
[0.3052, 0.7985, 0.5074],
[0.3176, 0.7994, 0.4975],
[0.3301, 0.8002, 0.4876],
[0.3424, 0.8009, 0.4774],
[0.3548, 0.8016, 0.4669],
[0.3671, 0.8021, 0.4563],
[0.3795, 0.8026, 0.4454],
[0.3921, 0.8029, 0.4344],
[0.405, 0.8031, 0.4233],
[0.4184, 0.803, 0.4122],
[0.4322, 0.8028, 0.4013],
[0.4463, 0.8024, 0.3904],
[0.4608, 0.8018, 0.3797],
[0.4753, 0.8011, 0.3691],
[0.4899, 0.8002, 0.3586],
[0.5044, 0.7993, 0.348],
[0.5187, 0.7982, 0.3374],
[0.5329, 0.797, 0.3267],
[0.547, 0.7957, 0.3159],
[0.5609, 0.7943, 0.305],
[0.5748, 0.7929, 0.2941],
[0.5886, 0.7913, 0.2833],
[0.6024, 0.7896, 0.2726],
[0.6161, 0.7878, 0.2622],
[0.6297, 0.7859, 0.2521],
[0.6433, 0.7839, 0.2423],
[0.6567, 0.7818, 0.2329],
[0.6701, 0.7796, 0.2239],
[0.6833, 0.7773, 0.2155],
[0.6963, 0.775, 0.2075],
[0.7091, 0.7727, 0.1998],
[0.7218, 0.7703, 0.1924],
[0.7344, 0.7679, 0.1852],
[0.7468, 0.7654, 0.1782],
[0.759, 0.7629, 0.1717],
[0.771, 0.7604, 0.1658],
[0.7829, 0.7579, 0.1608],
[0.7945, 0.7554, 0.157],
[0.806, 0.7529, 0.1546],
[0.8172, 0.7505, 0.1535],
[0.8281, 0.7481, 0.1536],
[0.8389, 0.7457, 0.1546],
[0.8495, 0.7435, 0.1564],
[0.86, 0.7413, 0.1587],
[0.8703, 0.7392, 0.1615],
[0.8804, 0.7372, 0.165],
[0.8903, 0.7353, 0.1695],
[0.9, 0.7336, 0.1749],
[0.9093, 0.7321, 0.1815],
[0.9184, 0.7308, 0.189],
[0.9272, 0.7298, 0.1973],
[0.9357, 0.729, 0.2061],
[0.944, 0.7285, 0.2151],
[0.9523, 0.7284, 0.2237],
[0.9606, 0.7285, 0.2312],
[0.9689, 0.7292, 0.2373],
[0.977, 0.7304, 0.2418],
[0.9842, 0.733, 0.2446],
[0.99, 0.7365, 0.2429],
[0.9946, 0.7407, 0.2394],
[0.9966, 0.7458, 0.2351],
[0.9971, 0.7513, 0.2309],
[0.9972, 0.7569, 0.2267],
[0.9971, 0.7626, 0.2224],
[0.9969, 0.7683, 0.2181],
[0.9966, 0.774, 0.2138],
[0.9962, 0.7798, 0.2095],
[0.9957, 0.7856, 0.2053],
[0.9949, 0.7915, 0.2012],
[0.9938, 0.7974, 0.1974],
[0.9923, 0.8034, 0.1939],
[0.9906, 0.8095, 0.1906],
[0.9885, 0.8156, 0.1875],
[0.9861, 0.8218, 0.1846],
[0.9835, 0.828, 0.1817],
[0.9807, 0.8342, 0.1787],
[0.9778, 0.8404, 0.1757],
[0.9748, 0.8467, 0.1726],
[0.972, 0.8529, 0.1695],
[0.9694, 0.8591, 0.1665],
[0.9671, 0.8654, 0.1636],
[0.9651, 0.8716, 0.1608],
[0.9634, 0.8778, 0.1582],
[0.9619, 0.884, 0.1557],
[0.9608, 0.8902, 0.1532],
[0.9601, 0.8963, 0.1507],
[0.9596, 0.9023, 0.148],
[0.9595, 0.9084, 0.145],
[0.9597, 0.9143, 0.1418],
[0.9601, 0.9203, 0.1382],
[0.9608, 0.9262, 0.1344],
[0.9618, 0.932, 0.1304],
[0.9629, 0.9379, 0.1261],
[0.9642, 0.9437, 0.1216],
[0.9657, 0.9494, 0.1168],
[0.9674, 0.9552, 0.1116],
[0.9692, 0.9609, 0.1061],
[0.9711, 0.9667, 0.1001],
[0.973, 0.9724, 0.0938],
[0.9749, 0.9782, 0.0872],
[0.9769, 0.9839, 0.0805]]

def parula_cmap(name: str = "parula_cmap"):
    """
    Return MATLAB Parula colormap as a Matplotlib LinearSegmentedColormap.

    Parameters
    ----------
    name : str
        Colormap name.
    register : bool
        If True, register with Matplotlib so you can use cmap=name.

    Returns
    -------
    cmap : matplotlib.colors.LinearSegmentedColormap
    """
    cmap = LinearSegmentedColormap.from_list(name, cm_data, N=len(cm_data))
    return cmap
    
##############################################
####### General Meteological Functions #######
##############################################

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


##############################################
####### Statistical Functions ################
##############################################

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
    
    
##############################################
####### SAM-specific Functions ###############
##############################################

def partition_n(T):
    """
    Calculate the hydrometeor partition fraction (non-precipitating condensate)
    
    Parameters:
    T (float): Temperature in Kelvin
    
    Returns:
    float: Hydrometeor partition fraction
    
    Notes:
    Following SAM model paper Khairoutdinov and Randall (2003)
    ω_n(T) = max(0, min(1, (T - T_00n)/(T_0n - T_00n)))
    """
    T_00n = 253.16
    T_0n  = 273.16
    return np.maximum(0.0, np.minimum(1.0, (T - T_00n) / (T_0n - T_00n)))

def partition_p(T):
    """
    Calculate the hydrometeor partition fraction (precipitating water)
    
    Parameters:
    T (float): Temperature in Kelvin
    
    Returns:
    float: Hydrometeor partition fraction
    
    Notes:
    Following SAM model paper Khairoutdinov and Randall (2003)
    ω_p(T) = max(0, min(1, (T - T_00p)/(T_0p - T_00p)))
    """
    T_00p = 268.16
    T_0p  = 283.16
    return np.maximum(0.0, np.minimum(1.0, (T - T_00p) / (T_0p - T_00p)))

def partition_g(T):
    """
    Calculate the hydrometeor partition fraction (graupel)
    
    Parameters:
    T (float): Temperature in Kelvin
    
    Returns:
    float: Hydrometeor partition fraction
    
    Notes:
    Following SAM model paper Khairoutdinov and Randall (2003)
    ω_g(T) = max(0, min(1, (T - T_00g)/(T_0g - T_00g)))
    """
    T_00g = 223.16
    T_0g  = 283.16
    return np.maximum(0.0, np.minimum(1.0, (T - T_00g) / (T_0g - T_00g)))

def dq_pdt_auto_cloud(q_c):
    """
    Calculate the source of precipitating water due to autoconversion of cloud water into precip
    described following the original Kessler formulation
    
    Parameters:
    q_c (float): cloud water (kg/kg)
    
    Returns:
    float: dq_p/dt_auto, source of precipitating water due to autoconversion of cloud water into rain
    
    Notes:
    Following SAM model paper Khairoutdinov and Randall (2003)
    dq_pdt_auto = max[0, alpha(q_c-q_co)0]
    """
    alpha = 0.001 # autoconversion rate, s-1
    q_co = 1e-3 # threshold cloud water for autoconversion, kg/kg

    return np.maximum(0, alpha * (q_c - q_co))

def dq_pdt_auto_ice(q_i, T):
    """
    Calculate the source of precipitating water due to autoconversion of ice water into precip
    described following the original Kessler formulation
    
    Parameters:
    q_i (float): icea water (kg/kg)
    T (float): temperature (K)
    
    Returns:
    float: dq_p/dt_auto, source of precipitating water due to autoconversion of ice water into precip
    
    Notes:
    Following SAM model paper Khairoutdinov and Randall (2003)
    eqn (A31)
    """
    beta = 0.001 # ice aggregation rate, s-1
    q_io = 1e-4 # threshold ice for autoconversion, kg/kg

    return np.maximum(0, beta * np.exp(0.025*(T-273.16)) * (q_i - q_io))

def dq_rdt_accr(q_l, q_r, RHO):
    """
    Expression for the rate of change of precipitating type m mixing ratio due to collection of condensate type l
    
    Parameters:
    q_l (float): liquid nonprecip (kg/kg)
    q_r (float): rain (kg/kg)
    RHO (float): Air density (kg/m^3).
    
    Returns:
    float: dq_rdt_accr
    
    Notes:
    Following SAM model paper Khairoutdinov and Randall (2003)
    eqn (A28)
    """
    
    # --- rain parameters ---
    a_r   = 842.0     # fall-speed coefficient for rain
    b_r   = 0.8       # fall-speed exponent for rain
    N0r   = 8.0e6     # intercept parameter for rain (m^-4)
    E_rl  = 1.0       # collection efficiency (rain collecting liquid)
    rho_r = 1000.0    # density of rain water (kg/m^3)
    rho_o = 1.29      # reference air density (kg/m^3)

    # exponent (3 + b_m)/4
    pow1 = (3.0 + b_r) / 4.0

    # Eq (A29): A_ar (using m=r, l=liquid)
    A_ar = (np.pi / 4.0) * a_r * N0r * E_rl * gamma(3.0 + b_r) \
           * (rho_o / RHO) ** 0.5 \
           * (RHO / (np.pi * rho_r * N0r)) ** pow1

    # Eq (A28)
    dq_rdt_accr = A_ar * q_l * np.maximum(q_r, 0.0) ** pow1
    return dq_rdt_accr 

def dq_rdt_evap(T, QR, QV, P, RHO):
    """
    Calculate the rate of change of precipitating water type (r)ain mixing ratio due to evaporation
    
    Parameters:
    T, QR, QV, P (hPa), RHO (kg/m3)
    
    Returns:
    float: dq_rdt_evap, rate of change of precipitating water type (r)ain mixing ratio due to evaporation
    
    Notes:
    Following SAM model paper Khairoutdinov and Randall (2003)
    eqn (A24)
    """
    C_r = 1 # rain shape factor
    N_0r = 8e6 # intercept parameter for rain, m-4
    L_c = 2.5104e6 # latent heat condensation J/kg 
    K_a = 2.4e-2 # thermal conductivity of air 0C, J M /k/s
    R_v = 461 # specific gas constant water vapor, J/kg/K
    D_a = 2.210e-5 # diffusion coeff water vapor 0C, m2/s
    
    L = L_c
    A = L/(K_a*T) * (L/(R_v*T)-1)
    R = 287 # spcific gas constant for air J/kg/K
    B = R_v*T / (D_a * 100*es(T))
    
    a_fr = 0.78 #Constant in ventilation factor for rain
    b_fr = 0.31 #Constant in ventilation factor for rain
    rho_r = 1000 # density of rain kg/m3
    a_r = 842 # Constant in fall speed formula for rain
    b_r = 0.8 # Exponent in fall speed formula for rai
    mu = 1.717e-5 # Dynamic viscosity of air at 0C
    rho_o = 1.29 # reference air density kg/m3
    
    A_er = a_fr * (RHO / np.pi/ rho_r / N_0r) ** 0.5
    B_er = b_fr * (RHO * a_r / mu) ** 0.5 * gamma((5+b_r)/2) * (rho_o/RHO)**0.25 * (RHO/np.pi/rho_r/N_0r)**((5+b_r)/8)
    S = QV/q_sat(T, P)
    
    return 2 * np.pi * C_r * N_0r /(RHO*(A+B)) * (A_er * QR**0.5 + B_er * QR**((5+b_r)/8)) * (S-1)

def mean_XY_from_pdf2d(pdf_XY_2d, xbin, ybin):
    """
    Compute mean X and mean Y from a 2D joint PDF over (X-bin, Y-bin).

    Parameters
    ----------
    pdf_XY_2d : array-like, shape (nX, nY)
        Joint PDF or histogram weights on the (X, Y) grid.
        If it's a true PDF, it should sum to 1 over both axes.
    xbin : array-like, shape (nX,)
        Bin centers for the X axis.
    ybin : array-like, shape (nY,)
        Bin centers for the Y axis.

    Returns
    -------
    mean_X : float
        Expected value of X.
    mean_Y : float
        Expected value of Y.
    """
    P = np.asarray(pdf_XY_2d, dtype=float)
    x = np.asarray(xbin, dtype=float)
    y = np.asarray(ybin, dtype=float)

    mean_X = np.nansum(P * x[:, None])
    mean_Y = np.nansum(P * y[None, :])
    return mean_X, mean_Y

def scatter_linreg(x, y, s=10, alpha=0.25, figsize=(6, 5), line_lw=2):
    """
    Make a scatter plot of y vs x, fit a least-squares line, and report slope/intercept/R²/p-value.

    Parameters
    ----------
    x : array-like, shape (n,)
        Predictor values.
    y : array-like, shape (n,)
        Response values.
    s : float, optional
        Scatter marker size.
    alpha : float, optional
        Scatter marker transparency.
    figsize : tuple, optional
        Figure size passed to matplotlib.
    line_lw : float, optional
        Line width for the fitted regression line.

    Returns
    -------
    res : scipy.stats._stats_mstats_common.LinregressResult
        Result object containing slope, intercept, rvalue, pvalue, stderr, intercept_stderr.
    """
    # --- try to infer variable names from caller namespace ---
    frame = inspect.currentframe().f_back
    namespaces = [frame.f_locals, frame.f_globals]
    try:
        ip = get_ipython()  # noqa: F821
        if ip is not None:
            namespaces.append(ip.user_ns)
    except NameError:
        pass

    def _infer_name(obj, default):
        for ns in namespaces:
            for k, v in ns.items():
                if v is obj:
                    return k
        return default

    xname = _infer_name(x, "x")
    yname = _infer_name(y, "y")

    # --- data + fit ---
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]

    res = linregress(x, y)
    slope, intercept, r2, p = res.slope, res.intercept, res.rvalue**2, res.pvalue

    # --- plot ---
    plt.figure(figsize=figsize)
    plt.scatter(x, y, s=s, alpha=alpha)
    xx = np.linspace(x.min(), x.max(), 200)
    plt.plot(xx, slope * xx + intercept, "r", lw=line_lw)
    plt.xlabel(xname)
    plt.ylabel(yname)
    plt.title(f"{yname} vs {xname}\nslope={slope:.3g}, intercept={intercept:.3g}\nR²={r2:.3f}, p={p:.1e}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    print(f"slope     = {slope:.6g}")
    print(f"intercept = {intercept:.6g}")
    print(f"R^2       = {r2:.6g}")
    print(f"p-value   = {p:.6g}")

    return res


