import numpy as np


def calc_prob_class_given_sensitive(predicted, sensitive, predicted_goal, sensitive_goal):
    """
    utils function for calculating DI score.

    Returns P(predicted = predicted_goal | sensitive = sensitive_goal).  Assumes that predicted
    and sensitive have the same length.  If there are no attributes matching the given
    sensitive_goal, this will error.
    """
    match_count = 0.0
    total = 0.0
    for sens, pred in zip(sensitive, predicted):
        if str(sens) == str(sensitive_goal):
            total += 1
            if str(pred) == str(predicted_goal):
                match_count += 1

    return match_count / total


def calc_generalized_entropy_index_given_b(b, alpha):
    """
    utils function for calculating GEI score.
    :param b: a set of benefit vector
    :param alpha: default 2
    :return GEI score
    """
    if alpha == 1:
        # moving the b inside the log allows for 0 values
        gei = np.mean(np.log((b / np.mean(b))**b) / np.mean(b))
    elif alpha == 0:
        gei = -np.mean(np.log(b / np.mean(b)) / np.mean(b))
    else:
        gei = np.mean((b / np.mean(b))**alpha - 1) / (alpha * (alpha - 1))
    return gei


class metrics():
    def __init__(self, data, prediction, priority_group=dict({'race': 'Caucasian'}), advantaged_outcome=0):
        """
        :param data: data frame of test data
        :param prediction: a dict of keys 
            - two_year_recid
            - RandomForestClassifier
            - LogisticRegressionCV
            - BernoulliNB
            - GaussianNB
            - KNeighborsClassifier
            - DecisionTreeClassifier
            each entry is the prediction (0 / 1) for the target
        :param priority_group: a dict of key (sensitive name, e.g. sex), values (unprotected value, e.g. Male)
                               in this project, we only consider single sensitive feature "race"
        :param advantaged_outcome: the outcome that has advantage. In this project, it will be 0,
                                   which means the individual do not commit crime within 2 years
        """
        assert len(list(priority_group)) > 0, "priorty group must not be empty"
        self.data = data
        self.pred = prediction
        self.algs = list(self.pred.keys())
        self.algs.remove('two_year_recid')
        self.labl = self.pred['two_year_recid']
        assert len(list(priority_group.keys())) == 1, "we only allow one sensitive feature"
        self.pgrp = priority_group
        self.advt = advantaged_outcome
    
    def get_privileged_index(self):
        keys = list(self.pgrp.keys())
        idx = None
        for key in keys:
            val = self.pgrp[key]
            if idx is None:
                idx = self.data[key] == val
            else:
                idx = idx & self.data[key] == val
        assert idx is not None
        return idx
    
    def get_accuracy(self):
        """
        calculate accuracy for whole prediction
        :return: a dict of {alg: acc}
        """
        accs = dict()
        for alg in self.algs:
            pred = self.pred[alg]
            N = pred.shape[0]
            correct = (pred == self.labl).sum()
            acc = correct / N
            accs[alg] = acc
        return accs

    def get_equality_opportunity(self, sensitive_name='race', s1='Caucasian', s2='African-American'):
        """
        :param sensitive_name: the name of the feature that is sensitive, e.g. race
        :param s1, s2: the two group in the sensitive name that you want to compare
        :return a dict of {alg: odds}
        """
        assert s1 in set(self.data[sensitive_name]), "s1 should be a value of feature sensitive_name"
        assert s2 in set(self.data[sensitive_name]), "s2 should be a value of feature sensitive_name"
        assert s1 != s2, "s1 and s2 should be different"
        idx1 = self.data[sensitive_name] == s1
        lab1 = self.labl[idx1]
        idx2 = self.data[sensitive_name] == s2
        lab2 = self.labl[idx2]
        equal_oppo = dict()
        for alg in self.algs:
            pred1 = self.pred[alg][idx1]
            pred2 = self.pred[alg][idx2]
            prob1 = np.logical_and(pred1 == self.advt, lab1 == self.advt).sum() / (lab1 == self.advt).sum()
            prob2 = np.logical_and(pred2 == self.advt, lab2 == self.advt).sum() / (lab2 == self.advt).sum()
            score = prob1 / prob2
            equal_oppo[alg] = score
        return equal_oppo


    def get_DI_score(self):
        """
        Reference:
            Sorelle Friedler, Carlos Scheidegger, Suresh Venkatasubramanian, Sonam Choudhary, 
            Evan Hamilton and Derek Roth. 2018. "A comparative study of fairness-enhancing 
            interventions in machine learning. " arXiv:1802.04422v1 [stat.ML] 13 Feb 2018.
        """

        predicted = self.pred
        single_sensitive_name = list(self.pgrp.keys())[0]  # race
        single_unprotected = self.pgrp[single_sensitive_name]  # Caucasian
        positive_pred = self.advt

        sensitive = self.data[single_sensitive_name]  # only look at sample of feature "race"
        sensitive_values = list(set(sensitive))
        sensitive_values.remove(single_unprotected)
        # the list is ['Other', 'Hispanic', 'African-American', 'Asian', 'Native American']
        
        if len(sensitive_values) <= 1:
             print("ERROR: Attempted to calculate DI without enough sensitive values:" + \
                   str(sensitive_values))
             return 1.0

        DIs = dict()
        for alg in self.algs:
            predicted = self.pred[alg]
            unprotected_prob = calc_prob_class_given_sensitive(predicted, sensitive, positive_pred,
                                                               single_unprotected)
            
            total = 0.0
            for sens in sensitive_values:
                pos_prob = calc_prob_class_given_sensitive(predicted, sensitive, positive_pred, sens)
                DI = 0.0
                if unprotected_prob > 0:
                    DI = pos_prob / unprotected_prob
                if unprotected_prob == 0.0 and pos_prob == 0.0:
                    DI = 1.0
                total += DI

            if total == 0.0:
                DIs[alg] = 1.0
            else:
                DIs[alg] = total / len(sensitive_values)

        return DIs


    def get_generalized_entropy_index(self, alpha=2):
        """Generalized entropy index is proposed as a unified individual and
        group fairness measure in [3].  With :math:`b_i = \hat{y}_i - y_i + 1`:

        :param alpha: Parameter that regulates the weight given to distances
                      between values at different parts of the distribution.
                      Default is 2.
        :return a dict of {alg: gei}

        References:
            [3] T. Speicher, H. Heidari, N. Grgic-Hlaca, K. P. Gummadi, A. Singla, A. Weller, and M. B. Zafar,
            "A Unified Approach to Quantifying Algorithmic Unfairness: Measuring Individual and Group Unfairness via Inequality Indices,"
            ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 2018.
        """
        y_true = self.labl
        geis = dict()
        for alg in self.algs:
            y_pred = self.pred[alg]
            y_pred = (y_pred == self.advt).astype(np.float64)
            y_true = (y_true == self.advt).astype(np.float64)
            # notice here: b represent the benefits of individual
            # If y_pred = 0, y_true = 1, then the benefits is 0. 
            # Meaning it suppose to be great result (1), but instead predict bad result (0)
            # If y_pred = 1, y_true = 0, then the benefits is 2.
            # Meaning it suppose to be bad result (0), but instead predict good result (1)
            # so here since 0 is advert result, we need to recalculate the benefits
            b = 1 + y_pred - y_true
            gei = calc_generalized_entropy_index_given_b(b, alpha)
            geis[alg] = gei
        return geis


    def get_generalized_entropy_index_within_group(self, alpha=2):
        """Calculate GEI within group
        """
        single_sensitive_name = list(self.pgrp.keys())[0]  # race
        sensitive = self.data[single_sensitive_name]  # only look at sample of feature "race"
        sensitive_values = list(set(sensitive))
        # the list is ['Other', 'Hispanic', 'African-American', 'Asian', 'Native American', 'Caucasian']
        n = self.labl.shape[0]
        y_true = self.labl

        geis = dict()
        for alg in self.algs:
            gei = 0.0
            y_pred = self.pred[alg]
            y_pred = y_pred == self.advt
            y_true = y_true == self.advt
            b = 1 + y_pred - y_true
            mu = np.mean(b)

            for sens in sensitive_values:
                idx = self.data[single_sensitive_name] == sens
                n_g = idx.sum()
                y_pred_g = y_pred[idx]
                y_true_g = y_true[idx]
                b_g = 1 + y_pred_g - y_true_g
                mu_g = np.mean(b_g)
                gei_g = calc_generalized_entropy_index_given_b(b_g, alpha)

                tmp = (n_g / n) * (mu_g / mu) ** alpha * gei_g
                gei += tmp

            geis[alg] = gei
        return geis


    def get_generalized_entropy_index_between_group(self, alpha=2):
        """Calculate GEI between group
        """
        single_sensitive_name = list(self.pgrp.keys())[0]  # race
        sensitive = self.data[single_sensitive_name]  # only look at sample of feature "race"
        sensitive_values = list(set(sensitive))
        # the list is ['Other', 'Hispanic', 'African-American', 'Asian', 'Native American', 'Caucasian']
        n = self.labl.shape[0]
        y_true = self.labl

        geis = dict()
        for alg in self.algs:
            gei = 0.0
            y_pred = self.pred[alg]
            y_pred = y_pred == self.advt
            y_true = y_true == self.advt
            b = 1 + y_pred - y_true
            mu = np.mean(b)

            for sens in sensitive_values:
                idx = self.data[single_sensitive_name] == sens
                n_g = idx.sum()
                y_pred_g = y_pred[idx]
                y_true_g = y_true[idx]
                b_g = 1 + y_pred_g - y_true_g
                mu_g = np.mean(b_g)

                tmp = (n_g / n) * ((mu_g / mu) ** alpha - 1)
                gei += tmp / (alpha * (alpha - 1))

            geis[alg] = gei
        return geis