import numpy as np
from scipy.stats import norm


class acq_functions(object):
    # This class has all the desired acq_functions

    def __init__(self, acq_name, bounds, count):
        # The implemented acq functions
        Acq_names = ["PI", "EI", "gp_ucb"]
        IsTrue = [val for idx, val in enumerate(Acq_names) if val in acq_name]

        if IsTrue == []:
            err = (
                "The acquisition function "
                "{} has not been implemented, "
                "please choose one from the given list".format(acq_name)
            )
            raise NotImplementedError(err)
        else:
            self.acq_name = acq_name

        self.count = count

    # Calls the specified acq with the arguments
    def acq_val(self, model, x, y_max):
        X = x

        if self.acq_name == "PI":
            return self._PI(model, X, y_max)

        if self.acq_name == "EI":
            return self._EI(model, X, y_max)

        if self.acq_name == "gp_ucb":
            return self._gp_ucb(model, X)

    @staticmethod
    def _PI(model, x, y_max):
        kappa = 0
        mean, var = model.predict(x)
        std = np.sqrt(var)
        z = (mean - y_max - kappa) / (std)
        prob = norm.cdf(z)
        return prob

    @staticmethod
    def _EI(model, x, y_max):
        mean, var = model.predict(x)
        std = np.sqrt(var)
        a = mean - y_max
        z = a / std
        improve = a * norm.cdf(z) + std * norm.pdf(z)
        return improve

    def _gp_ucb(self, model, x):
        beta = np.log(self.count)
        mean, var = model.predict(x)
        val = mean + np.sqrt(beta) * np.sqrt(var)
        return val
