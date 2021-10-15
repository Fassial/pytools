"""
Created on 20:49, Oct. 14th, 2021
Author: fassial
Filename: fitter.py
"""
## import packages
# import fund package
import sys
from datetime import datetime
# import dev package
import logging
import threading
from easydev import Progress
from joblib import Parallel, delayed
# import ds package
import pylab
import numpy as np
import scipy as sp
import scipy.stats
import pandas as pd

# macro
logger = logging.getLogger(__name__)
__all__ = ["get_common_distributions", "get_distributions", "Fitter"]

"""
get default distributions in sp.stats
"""
def get_distributions():
    distributions = []
    for this in dir(sp.stats):
        if "fit" in eval("dir(sp.stats." + this + ")"):
            distributions.append(this)
    return distributions

"""
get common distributions in sp stats
"""
def get_common_distributions():
    distributions = get_distributions()
    # to avoid error due to changes in scipy
    common = [
        "cauchy",
        "chi2",
        "expon",
        "exponpow",
        "gamma",
        "lognorm",
        "norm",
        "powerlaw",
        "rayleigh",
        "uniform",
    ]
    common = [x for x in common if x in distributions]
    return common

class Fitter(object):
    """Fit a data sample to known distributions
    A naive approach often performed to figure out the undelying distribution that
    could have generated a data set, is to compare the histogram of the data with
    a PDF (probability distribution function) of a known distribution (e.g., normal).
    Yet, the parameters of the distribution are not known and there are lots of
    distributions. Therefore, an automatic way to fit many distributions to the data
    would be useful, which is what is implemented here.
    Given a data sample, we use the `fit` method of SciPy to extract the parameters
    of that distribution that best fit the data. We repeat this for all available distributions.
    Finally, we provide a summary so that one can see the quality of the fit for those distributions
    Here is an example where we generate a sample from a gamma distribution.
    ::
        >>> # First, we create a data sample following a Gamma distribution
        >>> from scipy import stats
        >>> data = stats.gamma.rvs(2, loc=1.5, scale=2, size=20000)
        >>> # We then create the Fitter object
        >>> import fitter
        >>> f = fitter.Fitter(data)
        >>> # just a trick to use only 10 distributions instead of 80 to speed up the fitting
        >>> f.distributions = f.distributions[0:10] + ['gamma']
        >>> # fit and plot
        >>> f.fit()
        >>> f.summary()
                sumsquare_error
        gamma          0.000095
        beta           0.000179
        chi            0.012247
        cauchy         0.044443
        anglit         0.051672
        [5 rows x 1 columns]
    Once the data has been fitted, the :meth:`summary` metod returns a sorted dataframe where the
    Looping over the 80 distributions in SciPy could takes some times so you can overwrite the
    :attr:`distributions` with a subset if you want. In order to reload all distributions,
    call :meth:`load_all_distributions`.
    Some distributions do not converge when fitting. There is a timeout of 10 seconds after which
    the fitting procedure is cancelled. You can change this :attr:`timeout` attribute if needed.
    If the histogram of the data has outlier of very long tails, you may want to increase the
    :attr:`bins` binning or to ignore data below or above a certain range. This can be achieved
    by setting the :attr:`xmin` and :attr:`xmax` attributes. If you set xmin, you can come back to
    the original data by setting xmin to None (same for xmax) or just recreate an instance.
    """

    def __init__(
        self,
        data,
        xmin=None,
        xmax=None,
        bins=100,
        distributions=None,
        timeout=30,
        density=True,
    ):
        """.. rubric:: Constructor
        :param list data: a numpy array or a list
        :param float xmin: if None, use the data minimum value, otherwise histogram and
            fits will be cut
        :param float xmax: if None, use the data maximum value, otherwise histogram and
            fits will be cut
        :param int bins: numbers of bins to be used for the cumulative histogram. This has
            an impact on the quality of the fit.
        :param list distributions: give a list of distributions to look at. If none, use
            all scipy distributions that have a fit method. If you want to use
            only one distribution and know its name, you may provide a string (e.g.
            'gamma'). Finally, you may set to 'common' to  include only common
            distributions, which are: cauchy, chi2, expon, exponpow, gamma,
                 lognorm, norm, powerlaw, irayleigh, uniform.
        :param timeout: max time for a given distribution. If timeout is
            reached, the distribution is skipped.
        .. versionchanged:: 1.2.1 remove verbose argument, replacedb by logging module.
        .. versionchanged:: 1.0.8 increase timeout from 10 to 30 seconds.
        """
        self.timeout = timeout
        # USER input
        self._data = None

        # Issue https://github.com/cokelaer/fitter/issues/22 asked for setting
        # the density to False in the fitting and plotting. I first tought it
        # would be possible, but the fitting is performed using the PDF of scipy
        # so one would still need to normalise the data so that it is
        # comparable. Therefore I do not see anyway to do it without using
        # density set to True for now.
        self._density = True

        #: list of distributions to test
        self.distributions = distributions
        if self.distributions == None:
            self._load_all_distributions()
        elif self.distributions == "common":
            self.distributions = get_common_distributions()
        elif not isinstance(distributions, list):
            self.distributions = [distributions]
        self.distributions = [_check_distns(distribution) for distribution in self.distributions]

        self.bins = bins

        self._alldata = np.array(data)
        if xmin == None:
            self._xmin = self._alldata.min()
        else:
            self._xmin = xmin
        if xmax == None:
            self._xmax = self._alldata.max()
        else:
            self._xmax = xmax

        self._trim_data()
        self._update_data_pdf()
        self._update_data_pmf()

        # Other attributes
        self._init()

    def _init(self):
        self.fitted_param = {}
        self.fitted_pdf = {}
        self.fitted_pmf = {}
        self._fitted_errors = {}
        self._aic = {}
        self._bic = {}
        self._kldiv = {}
        self._fit_i = 0  # fit progress
        self.pb = None

    def _update_data_pdf(self):
        # histogram retuns X with N+1 values. So, we rearrange the X output into only N
        self.y_cont, self.bins_cont = np.histogram(self._data, bins=self.bins, density=self._density)
        self.x_cont = [(this + self.bins_cont[i + 1]) / 2.0 for i, this in enumerate(self.bins_cont[0:-1])]

    def _update_data_pmf(self):
        # get bin_min & bin_max
        bin_min = np.floor(np.min(self._data)) + .5
        if np.min(self._data) < bin_min: bin_min -= 1
        bin_max = np.floor(np.max(self._data)) + .5
        if np.max(self._data) >= bin_max: bin_max += 1
        # get bins
        self.bins_disc = [(bin_min + i) for i in range(0, int(bin_max - bin_min) + 1, 1)]

        # histogram retuns X with N+1 values. So, we rearrange the X output into only N
        self.y_disc, self.x_disc = np.histogram(self._data, bins=self.bins_disc, density=self._density)
        self.x_disc = [(this + self.x_disc[i + 1]) / 2.0 for i, this in enumerate(self.x_disc[0:-1])]

    def _trim_data(self):
        self._data = self._alldata[
            np.logical_and(self._alldata >= self._xmin, self._alldata <= self._xmax)
        ]

    def _get_xmin(self):
        return self._xmin

    def _set_xmin(self, value):
        if value == None:
            value = self._alldata.min()
        elif value < self._alldata.min():
            value = self._alldata.min()
        self._xmin = value
        self._trim_data()
        self._update_data_pdf()
        self._update_data_pmf()

    xmin = property(
        _get_xmin, _set_xmin, doc="consider only data above xmin. reset if None"
    )

    def _get_xmax(self):
        return self._xmax

    def _set_xmax(self, value):
        if value == None:
            value = self._alldata.max()
        elif value > self._alldata.max():
            value = self._alldata.max()
        self._xmax = value
        self._trim_data()
        self._update_data_pdf()
        self._update_data_pmf()

    xmax = property(
        _get_xmax, _set_xmax, doc="consider only data below xmax. reset if None "
    )

    def _load_all_distributions(self):
        """Replace the :attr:`distributions` attribute with all scipy distributions"""
        self.distributions = get_distributions()

    def hist(self):
        """Draw normed histogram of the data using :attr:`bins`
        .. plot::
            >>> from scipy import stats
            >>> data = stats.gamma.rvs(2, loc=1.5, scale=2, size=20000)
            >>> # We then create the Fitter object
            >>> import fitter
            >>> fitter.Fitter(data).hist()
        """
        if len(self.fitted_pdf) == 0 and len(self.fitted_pmf) > 0:
            _ = pylab.hist(self._data, bins=self.bins_disc, density=self._density)
        else:
            _ = pylab.hist(self._data, bins=self.bins_cont, density=self._density)
        pylab.grid(True)

    def _fit_single_distribution(self, distribution, progress: bool):
        try:
            # need a subprocess to check time it takes. If too long, skip it
            dist = distribution[0]; iscontinuous = distribution[1]; dist_name = distribution[2]

            # TODO here, dist.fit may take a while or just hang forever
            # with some distributions. So, I thought to use signal module
            # to catch the error when signal takes too long. It did not work
            # presumably because another try/exception is inside the
            # fit function, so I used threading with a recipe from stackoverflow
            # See timed_run function above
            param = self._timed_run(dist.fit, dist_name, args=self._data)

            # with signal, does not work. maybe because another expection is caught
            # hoping the order returned by fit is the same as in pdf & pmf
            if iscontinuous:
                # calcualte pdf
                self.fitted_pdf[dist_name] = dist.pdf(self.x_cont, *param)
                logLik = np.sum(dist.logpdf(self.x_cont, *param))
                # calculate error
                sq_error = pylab.sum((self.fitted_pdf[dist_name] - self.y_cont) ** 2)
                # calcualte kullback leibler divergence
                kullback_leibler = sp.stats.entropy(self.fitted_pdf[dist_name], self.y_cont)
            else:
                # calcualte pmf
                self.fitted_pmf[dist_name] = dist.pmf(self.x_disc, *param)
                logLik = np.sum(dist.logpmf(self.x_disc, *param))
                # calculate error
                sq_error = pylab.sum((self.fitted_pmf[dist_name] - self.y_disc) ** 2)
                # calcualte kullback leibler divergence
                kullback_leibler = sp.stats.entropy(self.fitted_pmf[dist_name], self.y_disc)

            self.fitted_param[dist_name] = param[:]

            # calcualte information criteria
            k = len(param[:])
            n = len(self._data)
            aic = 2 * k - 2 * logLik
            bic = n * np.log(sq_error / n) + k * np.log(n)

            logging.info(
                "Fitted {} distribution with error={})".format(dist_name, sq_error)
            )

            # compute some errors now
            self._fitted_errors[dist_name] = sq_error
            self._aic[dist_name] = aic
            self._bic[dist_name] = bic
            self._kldiv[dist_name] = kullback_leibler
        except Exception:  # pragma: no cover
            logging.warning(
                "SKIPPED {} distribution (taking more than {} seconds)".format(
                    dist_name, self.timeout
                )
            )
            # if we cannot compute the error, set it to large values
            self._fitted_errors[dist_name] = np.inf
            self._aic[dist_name] = np.inf
            self._bic[dist_name] = np.inf
            self._kldiv[dist_name] = np.inf
        if progress:
            self._fit_i += 1
            self.pb.animate(self._fit_i)

    def fit(self, amp=1, progress=False, n_jobs=-1):
        r"""Loop over distributions and find best parameter to fit the data for each
        When a distribution is fitted onto the data, we populate a set of
        dataframes:
            - :attr:`df_errors`  :sum of the square errors between the data and the fitted
              distribution i.e., :math:`\sum_i \left( Y_i - pdf(X_i) \right)^2`
            - :attr:`fitted_param` : the parameters that best fit the data
            - :attr:`fitted_pdf` : the PDF generated with the parameters that best fit the data
            - :attr:`fitted_pmf` : the PMF generated with the parameters that best fit the data
        Indices of the dataframes contains the name of the distribution.
        """
        import warnings

        warnings.filterwarnings("ignore", category=RuntimeWarning)

        if progress:
            self.pb = Progress(len(self.distributions))

        jobs = (
            delayed(self._fit_single_distribution)(dist, progress)
            for dist in self.distributions
        )
        pool = Parallel(n_jobs=n_jobs, backend="threading")
        _ = pool(jobs)
        self.df_errors = pd.DataFrame(
            {
                "sumsquare_error": self._fitted_errors,
                "aic": self._aic,
                "bic": self._bic,
                "kl_div": self._kldiv,
            }
        )

    def plot_pdf(self, names=None, Nbest=5, lw=2, method="sumsquare_error"):
        """Plots Probability density functions of the distributions
        :param str,list names: names can be a single distribution name, or a list
            of distribution names, or kept as None, in which case, the first Nbest
            distribution will be taken (default to best 5)
        """
        assert Nbest > 0
        if Nbest > len(self.distributions):
            Nbest = len(self.distributions)

        if isinstance(names, list):
            for name in names:
                pylab.plot(self.x_cont, self.fitted_pdf[name], lw=lw, label=name)
        elif names:
            pylab.plot(self.x_cont, self.fitted_pdf[names], lw=lw, label=names)
        else:
            try:
                names = self.df_errors.sort_values(by=method).index[0:Nbest]
            except Exception:
                names = self.df_errors.sort(method).index[0:Nbest]

            for name in names:
                if name in self.fitted_pdf.keys():
                    pylab.plot(self.x_cont, self.fitted_pdf[name], lw=lw, label=name)
        pylab.grid(True)
        pylab.legend()

    def plot_pmf(self, names=None, Nbest=5, lw=2, method="sumsquare_error"):
        """Plots Probability mass functions of the distributions
        :param str,list names: names can be a single distribution name, or a list
            of distribution names, or kept as None, in which case, the first Nbest
            distribution will be taken (default to best 5)
        """
        assert Nbest > 0
        if Nbest > len(self.distributions):
            Nbest = len(self.distributions)

        if isinstance(names, list):
            for name in names:
                pylab.plot(self.x_disc, self.fitted_pmf[name], lw=lw, label=name)
        elif names:
            pylab.plot(self.x_disc, self.fitted_pmf[names], lw=lw, label=names)
        else:
            try:
                names = self.df_errors.sort_values(by=method).index[0:Nbest]
            except Exception:
                names = self.df_errors.sort(method).index[0:Nbest]

            for name in names:
                if name in self.fitted_pmf.keys():
                    pylab.plot(self.x_disc, self.fitted_pmf[name], lw=lw, label=name)
                elif name not in self.fitted_pdf.keys():  # pragma: no cover
                    logger.warning("%s was not fitted. no parameters available" % name)
        pylab.grid(True)
        pylab.legend()

    def get_best(self, method="sumsquare_error"):
        """Return best fitted distribution and its parameters
        a dictionary with one key (the distribution name) and its parameters
        """
        # self.df should be sorted, so then us take the first one as the best
        name = self.df_errors.sort_values(method).iloc[0].name
        params = self.fitted_param[name]
        distribution = getattr(sp.stats, name)
        param_names = (
            (distribution.shapes + ", loc, scale").split(", ")
            if distribution.shapes
            else ["loc", "scale"]
        )

        param_dict = {}
        for d_key, d_val in zip(param_names, params):
            param_dict[d_key] = d_val
        return {name: param_dict}

    def summary(self, Nbest=5, lw=2, plot=True, method="sumsquare_error", clf=True):
        """Plots the distribution of the data and Nbest distribution"""
        if plot:
            if clf:
                pylab.clf()
            self.hist()
            self.plot_pdf(Nbest=Nbest, lw=lw, method=method)
            self.plot_pmf(Nbest=Nbest, lw=lw, method=method)
            pylab.grid(True)

        Nbest = min(Nbest, len(self.distributions))
        try:
            names = self.df_errors.sort_values(by=method).index[0:Nbest]
        except:  # pragma: no cover
            names = self.df_errors.sort(method).index[0:Nbest]
        return self.df_errors.loc[names]

    def _timed_run(self, func, distribution, args=(), kwargs={}, default=None):
        """This function will spawn a thread and run the given function
        using the args, kwargs and return the given default value if the
        timeout is exceeded.
        http://stackoverflow.com/questions/492519/timeout-on-a-python-function-call
        """

        class InterruptableThread(threading.Thread):
            def __init__(self):
                threading.Thread.__init__(self)
                self.result = default
                self.exc_info = (None, None, None)

            def run(self):
                try:
                    self.result = func(args, **kwargs)
                except Exception as err:  # pragma: no cover
                    print("func fail!")
                    self.exc_info = sys.exc_info()

            def suicide(self):  # pragma: no cover
                raise RuntimeError("Stop has been called")

        it = InterruptableThread()
        it.start()
        started_at = datetime.now()
        it.join(self.timeout)
        ended_at = datetime.now()
        diff = ended_at - started_at

        if (
            it.exc_info[0] is not None
        ):  # pragma: no cover ;  if there were any exceptions
            a, b, c = it.exc_info
            raise Exception(a, b, c)  # communicate that to caller

        if it.is_alive():  # pragma: no cover
            it.suicide()
            raise RuntimeError
        else:
            return it.result

def _check_distns(distribution):
    # convert to sp obj
    if isinstance(distribution, str):
        dist = eval("sp.stats." + distribution); dist_name = distribution
    elif "fit" in distribution.__dir__():
        dist = distribution; dist_name = distribution.name
    return (dist, isinstance(dist, sp.stats.rv_continuous), dist_name)
