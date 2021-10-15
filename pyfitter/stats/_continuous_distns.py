"""
Created on 00:46, Oct. 15th, 2021
Author: fassial
Filename: _continuous_distns.py
"""
# global dep
import numpy as np
import scipy as sp
import scipy.stats
from scipy._lib._util import _lazywhere
from scipy._lib.doccer import replace_notes_in_docstring
# local dep
from ._distn_infrastructure import get_distribution_names

## define continuous distributions
# TODO

## define tool funcs
# def _check_fit_input_parameters func
def _check_fit_input_parameters(data, args, kwds, fixed_param):
    if len(args) > 0:
        raise TypeError("Too many arguments.")

    _remove_optimizer_parameters(kwds)

    if None not in fixed_param:
        # This check is for consistency with `rv_continuous.fit`.
        # Without this check, this function would just return the
        # parameters that were given.
        raise RuntimeError("All parameters fixed. There is nothing to "
                           "optimize.")

    data = np.asarray(data)
    if not np.isfinite(data).all():
        raise RuntimeError("The data contains non-finite values.")

# collect names of classes and objects in this module.
pairs = list(globals().items())
_distn_names, _distn_gen_names = get_distribution_names(pairs, sp.stats.rv_continuous)

__all__ = _distn_names + _distn_gen_names
