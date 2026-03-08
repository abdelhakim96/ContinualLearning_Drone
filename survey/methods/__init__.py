from .nn_model import NNModel
from .pinn_model import PINNModel
from .sindy_model import SINDyModel
from .dmd_model import DMDModel
from .gp_model import GPModel
from .hankel_model import HankelDMDModel
from .classic_model import ClassicModel

ALL_METHODS = {
    'Classic':  ClassicModel,
    'NN':       NNModel,
    'PINN':     PINNModel,
    'SINDy':    SINDyModel,
    'DMD':      DMDModel,
    'GP':       GPModel,
    'Hankel':   HankelDMDModel,
}
