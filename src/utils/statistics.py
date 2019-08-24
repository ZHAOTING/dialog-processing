import numpy as np
from scipy import stats

class SignificanceTestMetrics:
    def __init__(self):
        pass

    def calculate_contingency(self, data_A, data_B):
        n = len(data_A)
        assert len(data_B) == n
        ABrr = 0
        ABrw = 0
        ABwr = 0
        ABww = 0
        for i in range(0, n):
            if(data_A[i] == 1 and data_B[i] == 1):
                ABrr = ABrr+1
            if (data_A[i] == 1 and data_B[i] == 0):
                ABrw = ABrw + 1
            if (data_A[i] == 0 and data_B[i] == 1):
                ABwr = ABwr + 1
            else:
                ABww = ABww + 1
        return np.array([[ABrr, ABrw], [ABwr, ABww]])

    def mcNemar(self, table):
        statistic = float(np.abs(table[0][1]-table[1][0]))**2/(table[1][0]+table[0][1])
        pval = 1-stats.chi2.cdf(statistic, 1)
        return pval

class OutlierDetector:
    def __init__(self):
        pass

    def detect_by_std_mean(self, data, n_dev):
        if len(data) == 0:
            return []

        # Set upper and lower limit to n_dev standard deviations
        data_std = np.std(data)
        data_mean = np.mean(data)
        anomaly_cut_off = data_std * n_dev

        lower_limit = data_mean - anomaly_cut_off
        upper_limit = data_mean + anomaly_cut_off

        # Generate outliers
        outliers = []
        for data_point in data:
            if data_point > upper_limit or data_point < lower_limit:
                outliers.append(data_point)
        return outliers

    def detect_by_abd_median(self, data, n_dev):
        ''' from "Detecting outliers: Do not use standard deviation around the mean, use absolute deviation around the median"
        '''
        if len(data) == 0:
            return []

        # Set upper and lower limit to n_dev absolute deviation medians
        data_median = np.median(data)
        abs_deviation = np.abs(data-data_median)
        abs_deviation_median = np.median(abs_deviation)
        b = 1.4826
        mad = abs_deviation_median*b
        anomaly_cut_off = mad * n_dev

        lower_limit = data_median - anomaly_cut_off
        upper_limit = data_median + anomaly_cut_off

        # Generate outliers
        outliers = []
        for data_point in data:
            if data_point > upper_limit or data_point < lower_limit:
                outliers.append(data_point)
        return outliers

class InterAnnotatorAgreementMetrics:
    def __init__(self):
        pass

    def fleiss_kappa(self, M, n_annotators):
        """
        from https://gist.github.com/skylander86/65c442356377367e27e79ef1fed4adee

        See `Fleiss' Kappa <https://en.wikipedia.org/wiki/Fleiss%27_kappa>`_.
        :param M: a matrix of shape (:attr:`N`, :attr:`k`) where `N` is the number of subjects and `k` is the number of categories into which assignments are made. `M[i, j]` represent the number of raters who assigned the `i`th subject to the `j`th category.
        :type M: numpy matrix
        """
        M = np.array(M)
        N, k = M.shape  # N is # of items, k is # of categories

        p = np.sum(M, axis=0) / (N * n_annotators)
        P = (np.sum(M * M, axis=1) - n_annotators) / (n_annotators * (n_annotators - 1))
        Pbar = np.sum(P) / N
        PbarE = np.sum(p * p)

        kappa = (Pbar - PbarE) / (1 - PbarE)

        return kappa

    def krippendorff_alpha(self, data, metric_type, force_vecmath=False, convert_items=float, missing_items=None):
        '''
        from https://github.com/grrrr/krippendorff-alpha/blob/master/krippendorff_alpha.py

        Calculate Krippendorff's alpha (inter-rater reliability):
        
        data is in the format
        [
            {unit1:value, unit2:value, ...},  # coder 1
            {unit1:value, unit3:value, ...},   # coder 2
            ...                            # more coders
        ]
        or 
        it is a sequence of (masked) sequences (list, numpy.array, numpy.ma.array, e.g.) with rows corresponding to coders and columns to items
        
        metric_type: type of function calculating the pairwise distance
        force_vecmath: force vector math for custom metrics (numpy required)
        convert_items: function for the type conversion of items (default: float)
        missing_items: indicator for missing items (default: None)
        '''

        def nominal_metric(a, b):
            return a != b

        def interval_metric(a, b):
            return (a-b)**2

        def ratio_metric(a, b):
            return ((a-b)/(a+b))**2

        if metric_type == "nominal":
            metric = nominal_metric
        elif metric_type == "interval":
            metric = interval_metric
        elif metric_type == "ratio":
            metric = ratio_metric

        # number of coders
        m = len(data)

        # set of constants identifying missing values
        if missing_items is None:
            maskitems = []
        else:
            maskitems = list(missing_items)
        if np is not None:
            maskitems.append(np.ma.masked_singleton)

        # convert input data to a dict of items
        units = {}
        for d in data:
            try:
                # try if d behaves as a dict
                diter = d.items()
            except AttributeError:
                # sequence assumed for d
                diter = enumerate(d)

            for it, g in diter:
                if g not in maskitems:
                    try:
                        its = units[it]
                    except KeyError:
                        its = []
                        units[it] = its
                    its.append(convert_items(g))

        units = dict((it, d) for it, d in units.items() if len(d) > 1)  # units with pairable values
        n = sum(len(pv) for pv in units.values())  # number of pairable values

        if n == 0:
            raise ValueError("No items to compare.")

        np_metric = (np is not None) and ((metric in (interval_metric, nominal_metric, ratio_metric)) or force_vecmath)

        Do = 0.
        for grades in units.values():
            if np_metric:
                gr = np.asarray(grades)
                Du = sum(np.sum(metric(gr, gri)) for gri in gr)
            else:
                Du = sum(metric(gi, gj) for gi in grades for gj in grades)
            Do += Du/float(len(grades)-1)
        Do /= float(n)

        if Do == 0:
            return 1.

        De = 0.
        for g1 in units.values():
            if np_metric:
                d1 = np.asarray(g1)
                for g2 in units.values():
                    De += sum(np.sum(metric(d1, gj)) for gj in g2)
            else:
                for g2 in units.values():
                    De += sum(metric(gi, gj) for gi in g1 for gj in g2)
        De /= float(n*(n-1))

        return 1.-Do/De if (Do and De) else 1.


class CorrelationMetrics:
    def __init__(self):
        pass

    def pearson_cor(self, data1, data2):
        r, p = stats.pearsonr(data1, data2)
        return r, p

    def spearman_cor(self, data1, data2=None):
        rho, p = stats.spearmanr(data1, data2)
        return rho, p
