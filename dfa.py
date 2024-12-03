import numpy
from sklearn.linear_model import LinearRegression
from statsmodels.api import OLS, add_constant


def dfa(x: numpy.ndarray, max_window_size: int, min_window_size: int = 3, return_confidence_interval: bool = True,
        return_windows=False):
    """
    Expects X to be a time series of T rows and n columns.
    :param return_confidence_interval:
    :param x:
    :param max_window_size:
    :param min_window_size:
    :return: a list whose first element is the vector of DFA exponents for the n features and whose second element is the r_squared for the log-log regression
    """

    def unbounded(xx):
        return numpy.cumsum(xx - xx.mean(0), axis=0)

    def windows(yt, n):
        return numpy.split(yt, numpy.arange(n, yt.shape[0], step=n))

    def regression_error(window_list, n, return_windows=False):
        filtered_windows = list(filter(lambda x: x.shape[0] == n, window_list))
        y = numpy.stack(filtered_windows)
        d = add_constant(numpy.arange(1, filtered_windows[0].shape[0] + 1))
        soln = d @ numpy.linalg.inv(d.T @ d) @ d.T
        if not return_windows:
            return numpy.swapaxes(y.T, 1, 2) - numpy.swapaxes(y.T, 1, 2) @ soln
        else:
            return numpy.swapaxes(y.T, 1, 2) - numpy.swapaxes(y.T, 1, 2) @ soln, len(filtered_windows)

    def f_n(error, return_windows=False):
        if not return_windows:
            return numpy.sqrt(numpy.power(error, 2.0).mean(1).mean(1))
        else:
            return numpy.sqrt(numpy.power(error[0], 2.0).mean(1).mean(1)), error[1]

    def n_values(xx, mn, mx):
        return [numpy.asarray(range(mn, mx))[
                    numpy.max(numpy.argwhere(numpy.asarray([xx.shape[0] // i for i in range(mn, mx)]) == n))] for n in
                numpy.unique([xx.shape[0] // i for i in range(mn, mx)])]

    def f(xx, ns, return_windows=False):
        if not return_windows:
            fns = list(map(lambda n: f_n(regression_error(windows(unbounded(xx), n), n)), ns))
            return numpy.vstack(fns)
        else:
            fns = list(map(lambda n: f_n(regression_error(windows(unbounded(xx), n), n, return_windows=return_windows),
                                         return_windows=return_windows), ns))
            return numpy.vstack([f[0] for f in fns]), numpy.hstack([f[1] for f in fns])

    ens = n_values(x, mn=min_window_size, mx=max_window_size)
    features = numpy.log(ens).reshape(-1, 1)

    window_numbers = None
    if not return_windows:
        y = numpy.log(f(xx=x, ns=ens))
    else:
        y = f(xx=x, ns=ens, return_windows=return_windows)
        window_numbers = y[1]
        y = numpy.log(y[0])

    if not return_confidence_interval:
        lr = LinearRegression().fit(features, y)
        r_squared = lr.score(features, y)
        hurst = lr.coef_.flatten()
        return hurst, r_squared
    else:
        estimates = [OLS(endog=y[:, i], exog=add_constant(features)).fit() for i in range(y.shape[1])]
        hurst = [est.params[1] for est in estimates]
        cis = [est.conf_int(alpha=0.05)[1, :] for est in estimates]
        rsquared = [est.rsquared for est in estimates]
        if not return_windows:
            return hurst, cis, rsquared
        else:
            return hurst, cis, rsquared, ens, window_numbers