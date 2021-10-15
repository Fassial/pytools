"""
Created on 02:16, Oct. 15t, 2021
Author: fassial
Filename: _distn_infrastructure.py
"""
__all__ = [
    "get_distribution_names",
]

## define tool funcs
# def get_distribution_names func
def get_distribution_names(namespace_pairs, rv_base_class):
    """
    Collect names of statistical distributions and their generators.

    Parameters
    ----------
    namespace_pairs : sequence
        A snapshot of (name, value) pairs in the namespace of a module.
    rv_base_class : class
        The base class of random variable generator classes in a module.

    Returns
    -------
    distn_names : list of strings
        Names of the statistical distributions.
    distn_gen_names : list of strings
        Names of the generators of the statistical distributions.
        Note that these are not simply the names of the statistical
        distributions, with a _gen suffix added.

    """
    distn_names = []
    distn_gen_names = []
    for name, value in namespace_pairs:
        if name.startswith('_'):
            continue
        if name.endswith('_gen') and issubclass(value, rv_base_class):
            distn_gen_names.append(name)
        if isinstance(value, rv_base_class):
            distn_names.append(name)
    return distn_names, distn_gen_names
