from typing import Callable, Iterator, Tuple
import chex
import jax

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import jaxopt
import h5py
import pandas as pd
from scipy import signal, interpolate
import sxs
import glob
import sys

from math import pi, log
from ripple.typing import Array
from scipy.optimize import minimize, minimize_scalar

from numpy import random, abs
from ripple.waveforms import IMRPhenomD, IMRPhenomD_utils
from ripple.waveforms.IMRPhenomD import *
from ripple.waveforms.IMRPhenomD_utils import get_coeffs, get_transition_frequencies
from ripple import ms_to_Mc_eta, Mc_eta_to_ms
from jax import grad, vmap, scipy
from functools import partial
import time
from tqdm import tqdm
import json

PhenomD_coeff_table = jnp.array(
    [
        [  # rho1 (element 0)
            3931.8979897196696,
            -17395.758706812805,
            3132.375545898835,
            343965.86092361377,
            -1.2162565819981997e6,
            -70698.00600428853,
            1.383907177859705e6,
            -3.9662761890979446e6,
            -60017.52423652596,
            803515.1181825735,
            -2.091710365941658e6,
        ],
        [  # rho2 (element 1)
            -40105.47653771657,
            112253.0169706701,
            23561.696065836168,
            -3.476180699403351e6,
            1.137593670849482e7,
            754313.1127166454,
            -1.308476044625268e7,
            3.6444584853928134e7,
            596226.612472288,
            -7.4277901143564405e6,
            1.8928977514040343e7,
        ],
        [  # rho3 (element 2)
            83208.35471266537,
            -191237.7264145924,
            -210916.2454782992,
            8.71797508352568e6,
            -2.6914942420669552e7,
            -1.9889806527362722e6,
            3.0888029960154563e7,
            -8.390870279256162e7,
            -1.4535031953446497e6,
            1.7063528990822166e7,
            -4.2748659731120914e7,
        ],
        [  # v2 (element 3)
            0.8149838730507785,
            2.5747553517454658,
            1.1610198035496786,
            -2.3627771785551537,
            6.771038707057573,
            0.7570782938606834,
            -2.7256896890432474,
            7.1140380397149965,
            0.1766934149293479,
            -0.7978690983168183,
            2.1162391502005153,
        ],
        [  # gamma1 (element 4)
            0.006927402739328343,
            0.03020474290328911,
            0.006308024337706171,
            -0.12074130661131138,
            0.26271598905781324,
            0.0034151773647198794,
            -0.10779338611188374,
            0.27098966966891747,
            0.0007374185938559283,
            -0.02749621038376281,
            0.0733150789135702,
        ],
        [  # gamma2 (element 5)
            1.010344404799477,
            0.0008993122007234548,
            0.283949116804459,
            -4.049752962958005,
            13.207828172665366,
            0.10396278486805426,
            -7.025059158961947,
            24.784892370130475,
            0.03093202475605892,
            -2.6924023896851663,
            9.609374464684983,
        ],
        [  # gamma3 (element 6)
            1.3081615607036106,
            -0.005537729694807678,
            -0.06782917938621007,
            -0.6689834970767117,
            3.403147966134083,
            -0.05296577374411866,
            -0.9923793203111362,
            4.820681208409587,
            -0.006134139870393713,
            -0.38429253308696365,
            1.7561754421985984,
        ],
        [  # sig1 (element 7)
            2096.551999295543,
            1463.7493168261553,
            1312.5493286098522,
            18307.330017082117,
            -43534.1440746107,
            -833.2889543511114,
            32047.31997183187,
            -108609.45037520859,
            452.25136398112204,
            8353.439546391714,
            -44531.3250037322,
        ],
        [  # sig2 (element 8)
            -10114.056472621156,
            -44631.01109458185,
            -6541.308761668722,
            -266959.23419307504,
            686328.3229317984,
            3405.6372187679685,
            -437507.7208209015,
            1.6318171307344697e6,
            -7462.648563007646,
            -114585.25177153319,
            674402.4689098676,
        ],
        [  # sig3 (element 9)
            22933.658273436497,
            230960.00814979506,
            14961.083974183695,
            1.1940181342318142e6,
            -3.1042239693052764e6,
            -3038.166617199259,
            1.8720322849093592e6,
            -7.309145012085539e6,
            42738.22871475411,
            467502.018616601,
            -3.064853498512499e6,
        ],
        [  # sig4 (element 10)
            -14621.71522218357,
            -377812.8579387104,
            -9608.682631509726,
            -1.7108925257214056e6,
            4.332924601416521e6,
            -22366.683262266528,
            -2.5019716386377467e6,
            1.0274495902259542e7,
            -85360.30079034246,
            -570025.3441737515,
            4.396844346849777e6,
        ],
        [  # beta1 (element 11)
            97.89747327985583,
            -42.659730877489224,
            153.48421037904913,
            -1417.0620760768954,
            2752.8614143665027,
            138.7406469558649,
            -1433.6585075135881,
            2857.7418952430758,
            41.025109467376126,
            -423.680737974639,
            850.3594335657173,
        ],
        [  # beta2 (element 12)
            -3.282701958759534,
            -9.051384468245866,
            -12.415449742258042,
            55.4716447709787,
            -106.05109938966335,
            -11.953044553690658,
            76.80704618365418,
            -155.33172948098394,
            -3.4129261592393263,
            25.572377569952536,
            -54.408036707740465,
        ],
        [  # beta3 (element 13)
            -0.000025156429818799565,
            0.000019750256942201327,
            -0.000018370671469295915,
            0.000021886317041311973,
            0.00008250240316860033,
            7.157371250566708e-6,
            -0.000055780000112270685,
            0.00019142082884072178,
            5.447166261464217e-6,
            -0.00003220610095021982,
            0.00007974016714984341,
        ],
        [  # a1 (element 14)
            43.31514709695348,
            638.6332679188081,
            -32.85768747216059,
            2415.8938269370315,
            -5766.875169379177,
            -61.85459307173841,
            2953.967762459948,
            -8986.29057591497,
            -21.571435779762044,
            981.2158224673428,
            -3239.5664895930286,
        ],
        [  # a2 (element 15)
            -0.07020209449091723,
            -0.16269798450687084,
            -0.1872514685185499,
            1.138313650449945,
            -2.8334196304430046,
            -0.17137955686840617,
            1.7197549338119527,
            -4.539717148261272,
            -0.049983437357548705,
            0.6062072055948309,
            -1.682769616644546,
        ],
        [  # a3 (element 16)
            9.5988072383479,
            -397.05438595557433,
            16.202126189517813,
            -1574.8286986717037,
            3600.3410843831093,
            27.092429659075467,
            -1786.482357315139,
            5152.919378666511,
            11.175710130033895,
            -577.7999423177481,
            1808.730762932043,
        ],
        [  # a4 (element 17)
            -0.02989487384493607,
            1.4022106448583738,
            -0.07356049468633846,
            0.8337006542278661,
            0.2240008282397391,
            -0.055202870001177226,
            0.5667186343606578,
            0.7186931973380503,
            -0.015507437354325743,
            0.15750322779277187,
            0.21076815715176228,
        ],
        [  # a5 (element 18)
            0.9974408278363099,
            -0.007884449714907203,
            -0.059046901195591035,
            1.3958712396764088,
            -4.516631601676276,
            -0.05585343136869692,
            1.7516580039343603,
            -5.990208965347804,
            -0.017945336522161195,
            0.5965097794825992,
            -2.0608879367971804,
        ],
    ]
)


"""
Various constants, all in SI units.
"""

EulerGamma = 0.577215664901532860606512090082402431

MSUN = 1.9884099021470415e30  # kg
"""Solar mass"""

G = 6.67430e-11  # m^3 / kg / s^2
"""Newton's gravitational constant"""

C = 299792458.0  # m / s
"""Speed of light"""

gt = G * MSUN / (C ** 3.0)
"""
G MSUN / C^3 in seconds
"""

m_per_Mpc = 3.085677581491367278913937957796471611e22
"""
Meters per Mpc.
"""

@jax.jit
def _get_coeffs(theta: Array, table: Array) -> Array:
    # Retrives the coefficients needed to produce the waveform

    m1, m2, chi1, chi2 = theta
    m1_s = m1 * gt
    m2_s = m2 * gt
    M_s = m1_s + m2_s
    eta = m1_s * m2_s / (M_s ** 2.0)

    # Definition of chiPN from lalsuite
    chi_s = (chi1 + chi2) / 2.0
    chi_a = (chi1 - chi2) / 2.0
    seta = (1 - 4 * eta) ** (1 / 2)
    chiPN = chi_s * (1 - 76 * eta / 113) + seta * chi_a

    coeff = (
        table[:, 0]
        + table[:, 1] * eta
        + (chiPN - 1.0)
        * (
            table[:, 2]
            + table[:, 3] * eta
            + table[:, 4] * (eta ** 2.0)
            )
        + (chiPN - 1.0) ** 2.0
        * (
            table[:, 5]
            + table[:, 6] * eta
            + table[:, 7] * (eta ** 2.0)
        )
        + (chiPN - 1.0) ** 3.0
        * (
            table[:, 8]
            + table[:, 9] * eta
            + table[:, 10] * (eta ** 2.0)
        )
    )

    # FIXME: Change to dictionary lookup
    return coeff

noise_dataframe = pd.read_csv('/mnt/home/klam1/code/aLIGOZeroDetHighPower_fs.txt', delimiter=' ')
noise_curve = noise_dataframe.values[:, 1]
noise_f = noise_dataframe.values[:, 0]

@jax.jit
def inner(h1: Array, h2: Array, f):
    df = f[1] - f[0]
    # noise = jnp.interp(f, noise_f, noise_curve)
    noise = 1
    cross_multi = jnp.real(h1 * jnp.conj(h2)) / noise
    return 4 * sum(cross_multi * df)

@jax.jit
def mismatch(h1: Array, h2: Array, f):
    return 1 - (inner(h1, h2, f) / jnp.sqrt(inner(h1, h1, f) * inner(h2, h2, f)))

@jax.jit
def loss(lambdas: Array, intrin: Array, extrin: Array, f: Array, NR_complex: Array) -> Array:
    f_sep = int(len(f) / 100)
    
    NR_phase = -jnp.unwrap(jnp.angle(NR_complex))
    IMR = IMRPhenomD._gen_IMRPhenomD(f, intrin, extrin, _get_coeffs(intrin, lambdas * scale))
    IMR_phase = -jnp.unwrap(jnp.angle(IMR))
    phase_diff = NR_phase - IMR_phase
    
    A = jnp.vstack([f, jnp.ones(len(f))]).T
    two_pi_t0, phi0 = jnp.linalg.lstsq(A, phase_diff, rcond=None)[0]
    
    NR_shifted = NR_complex * jnp.exp(1j * (two_pi_t0 * f + phi0))
    
    return mismatch(NR_shifted[0::f_sep], IMR[0::f_sep], f[0::f_sep])

def total_loss(lambdas: Array, intrin_list: Array, extrin: Array) -> Array:
    loss_value = 0
    for i in range(len(catalog_list)):
        data = pd.read_csv('./NR_waveform/NR_'+str(catalog_list[i])+'.txt', sep=" ", header=None)
        f_uniform = data.values[:, 0]
        NR_waveform = data.values[:, 1] + 1j * data.values[:, 2]
        loss_value += loss(lambdas, intrin_list[i], extrin, f_uniform, NR_waveform) #/ original_loss_list[i]
    return loss_value

scale = jnp.array(PhenomD_coeff_table)

catalog_list = ['0304', '0327', '2123', '2128', '2132', '2153', '0045', '0292'] 
M = 50.00
theta_extrinsic = jnp.array([440.0, 0.0, 0.0])

theta_intrinsic_list = []
original_loss_list = []
for i in range(len(catalog_list)):
    with open('/mnt/home/klam1/ceph/NR_waveform/NR_' + str(catalog_list[i]) + '_metadata.json') as file:
        metadata = json.load(file)
        q = round(metadata['reference_mass_ratio'] * 1000) / 1000
        chi1 = metadata['reference_dimensionless_spin1'][2]
        chi2 = metadata['reference_dimensionless_spin2'][2]

        theta_intrinsic = [M * q / (1 + q), M * 1 / (1 + q), chi1, chi2]
        theta_intrinsic_list.append(theta_intrinsic)

    data = pd.read_csv('/mnt/home/klam1/ceph/NR_waveform/NR_'+str(catalog_list[i])+'.txt', sep=" ", header=None)
    f_uniform = data.values[:, 0]
    NR_waveform = data.values[:, 1] + 1j * data.values[:, 2]
    original_loss_value = loss(PhenomD_coeff_table / scale, theta_intrinsic_list[i], theta_extrinsic, f_uniform, NR_waveform)
    original_loss_list.append(original_loss_value)
    
test = jnp.array(PhenomD_coeff_table) 

params = test / scale
loss_jit = jax.jit(jax.value_and_grad(total_loss))
alpha = 1e-6

data_list = np.array([])
start_time = time.time()
original_loss = total_loss(scale / scale, theta_intrinsic_list, theta_extrinsic)
print('---------------------------------------------------------------------')
for i in range(30000):
    value, grad = loss_jit(params, theta_intrinsic_list, theta_extrinsic)
    if i % 3000 == 0:
        print('iteration %d' % i)
        print('loss %.6e' % value)
        print('stepsize %.6e' % alpha)
        print('---------------------------------------------------------------------')
        data = (params * scale).flatten()
        data = np.append(data, original_loss)
        data = np.append(data, value)
        data_list = np.append(data_list, data)

    params -= alpha * grad / jnp.linalg.norm(grad)

print("--- %s seconds ---" % (time.time() - start_time))

data = (params * scale).flatten()
data = np.append(data, original_loss)
data = np.append(data, value)
data_list = np.append(data_list, data)

with open("/mnt/home/klam1/code/quadrant_data/1pos2neg.txt", "w") as file:
    np.savetxt(file, data_list, delimiter=', ')
