"""
Created on 21:18, Oct. 14th, 2021
Author: fassial
Filename: _discrete_distns.py
"""
# global dep
import numpy as np
import scipy as sp
import scipy.stats
from scipy._lib._util import _lazywhere
from scipy._lib.doccer import replace_notes_in_docstring
# local dep
from ._distn_infrastructure import get_distribution_names

## define discrete distributions
# def poisson_gen class
class poisson_gen(sp.stats.rv_discrete):
    r"""A Poisson discrete random variable.

    %(before_notes)s

    Notes
    -----
    The probability mass function for `poisson` is:

    .. math::

        f(k) = \exp(-\mu) \frac{\mu^k}{k!}

    for :math:`k \ge 0`.

    `poisson` takes :math:`\mu` as shape parameter.

    %(after_notes)s

    %(example)s

    """

    # Override rv_discrete._argcheck to allow mu=0.
    def _argcheck(self, mu):
        return mu >= 0

    def _rvs(self, mu, size=None, random_state=None):
        return random_state.poisson(mu, size)

    def _logpmf(self, k, mu):
        Pk = sp.special.xlogy(k, mu) - sp.special.gammaln(k + 1) - mu
        return Pk

    def _pmf(self, k, mu):
        # poisson.pmf(k) = exp(-mu) * mu**k / k!
        return np.exp(self._logpmf(k, mu))

    def _cdf(self, x, mu):
        k = np.floor(x)
        return sp.special.pdtr(k, mu)

    def _sf(self, x, mu):
        k = np.floor(x)
        return sp.special.pdtrc(k, mu)

    def _ppf(self, q, mu):
        vals = np.ceil(sp.special.pdtrik(q, mu))
        vals1 = np.maximum(vals - 1, 0)
        temp = sp.special.pdtr(vals1, mu)
        return np.where(temp >= q, vals1, vals)

    def _stats(self, mu):
        var = mu
        tmp = np.asarray(mu)
        mu_nonzero = tmp > 0
        g1 = _lazywhere(mu_nonzero, (tmp,), lambda x: np.sqrt(1.0/x), np.inf)
        g2 = _lazywhere(mu_nonzero, (tmp,), lambda x: 1.0/x, np.inf)
        return mu, var, g1, g2

    def fit(self, data, **kwargs):
        # get fixed params
        flambda = kwargs.pop("flambda", None)
        _check_fit_input_parameters(
            data = data,
            kwargs = kwargs,
            fixed_param = (flambda,)
        )

        # MLE for the poisson distribution
        if flambda is None:
            _lambda = np.mean(data)
        else:
            _lambda = flambda

        # Source: Statistical Distributions, 4th Edition. Evans, Hastings,
        # and Peacock (2011), Page 156

        return (_lambda,)

poisson = poisson_gen(name="poisson", longname='A Poisson')

## define tool funcs
# def _remove_optimizer_parameters func
def _remove_optimizer_parameters(kwargs):
    """
    Remove the optimizer-related keyword arguments 'loc', 'scale' and
    'optimizer' from `kwargs`.  Then check that `kwargs` is empty, and
    raise `TypeError("Unknown arguments: %s." % kwargs)` if it is not.

    This function is used in the fit method of distributions that override
    the default method and do not use the default optimization code.

    `kwargs` is modified in-place.
    """
    kwargs.pop("lambda", None)
    kwargs.pop("optimizer", None)
    if kwargs:
        raise TypeError("Unknown arguments: %s." % kwargs)

# def _check_fit_input_parameters func
def _check_fit_input_parameters(data, kwargs, fixed_param):
    _remove_optimizer_parameters(kwargs)

    if None not in fixed_param:
        # This check is for consistency with `rv_discrete.fit`.
        # Without this check, this function would just return the
        # parameters that were given.
        raise RuntimeError("All parameters fixed. There is nothing to "
                           "optimize.")

    data = np.asarray(data)
    if not np.isfinite(data).all():
        raise RuntimeError("The data contains non-finite values.")

# collect names of classes and objects in this module.
pairs = list(globals().items())
_distn_names, _distn_gen_names = get_distribution_names(pairs, sp.stats.rv_discrete)

__all__ = _distn_names + _distn_gen_names
