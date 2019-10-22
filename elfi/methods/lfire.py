"""Public module documentation comes here."""

import logging

import numpy as np
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.preprocessing import StandardScaler

from elfi.methods.parameter_inference import ParameterInference
from elfi.methods.results import LfirePosterior
from elfi.methods.utils import ModelPrior
from elfi.model.elfi_model import Summary

logger = logging.getLogger(__name__)


class LFIRE(ParameterInference):
    """Class documentation comes here."""

    def __init__(self, model, params_grid=None, marginal=None, output_names=None, Cs=None, cv=None,
                 **kwargs):
        """LFIRE method.

        Parameters:
        ----------
        model : elfi model
        params_grid : numpy array
            Parameter grid
        marginal : numpy array
        Cs : float or numpy array
            Cs is either a numpy array or positive float containing the inverse of regularisation strength.
        cv : int or string
            cv contains either the number folds if the CV function is k-Fold, otherwise a name of the CV function.
        """

        # super(LFIRE, self).__init__(model, output_names, **kwargs)
        super().__init__(model, output_names, **kwargs)

        self.summary_names = self._get_summary_names()
        if len(self.summary_names) == 0:
            raise NotImplementedError('Your model must have at least one Summary node.')

        self.params_grid = self._resolve_params_grid(params_grid)
        self.marginal = self._resolve_marginal(marginal)
        self.observed = self._get_observed_summary_values()
        self.joint_prior = ModelPrior(self.model)

        if isinstance(Cs, np.ndarray):
            self.CV_boolean = True
            self.logreg_config = {
                'penalty': 'l1',
                'Cs': Cs,
                'solver': 'liblinear',
                'cv': cv,
            }
        else:
            self.CV_boolean = False
            self.logreg_config = {
                'penalty': 'l1',
                'C': Cs,
                'solver': 'liblinear',
            }

        n_batches = self.params_grid.shape[0]
        self.state['posterior'] = np.zeros(n_batches)
        for parameter_name in self.parameter_names:
            self.state[parameter_name] = np.zeros(n_batches)
        # Store values of coefficients, C and intercept
        self.state['coefficients'] = []
        self.state['C'] = []
        self.state['intercept'] = []

    def set_objective(self):
        """Class method documenation comes here."""
        self.objective['n_batches'] = self.params_grid.shape[0]
        self.objective['n_sim'] = self.params_grid.shape[0] * self.batch_size

    def extract_result(self):
        """Class method documentation comes here."""
        return LfirePosterior(
            method_name='LFIRE',
            outputs=self.state,
            parameter_names=self.parameter_names
        )

    def update(self, batch, batch_index):
        """Class method documenation comes here."""
        super().update(batch, batch_index)  # LFIRE, self

        # Parse likelihood values
        # likelihood = self._get_summary_values(batch=batch)
        likelihood = [batch[summary_name] for summary_name in self.summary_names]
        likelihood = np.column_stack(likelihood)

        # Create training data
        X = np.vstack((likelihood, self.marginal))
        y = np.concatenate((np.ones(likelihood.shape[0]), -1*np.ones(self.marginal.shape[0])))

        # Scale
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Logistic regression
        # Checks if the Cs variable contains multiple values in a list
        # or just one
        if self.CV_boolean is True:
            m = LogisticRegressionCV(**self.logreg_config)
        else:
            m = LogisticRegression(**self.logreg_config)

        m.fit(X_scaled, y)

        # Convert back to the original scale
        coefficients = np.true_divide(m.coef_,  scaler.scale_)
        intercept = m.intercept_ - np.dot(coefficients, scaler.mean_)

        # Likelihood value
        log_likelihood_value = intercept + np.sum(np.multiply(coefficients, self.observed))
        likelihood_value = np.exp(log_likelihood_value)

        # Joint prior value
        parameter_values = [batch[parameter_name] for parameter_name in self.parameter_names]
        joint_prior_value = self.joint_prior.pdf(parameter_values)

        # Posterior value
        posterior_value = joint_prior_value * likelihood_value

        # Update state dictionary
        self.state['posterior'][batch_index] = posterior_value[0]
        for parameter_name in self.parameter_names:
            self.state[parameter_name][batch_index] = batch[parameter_name]
        self.state['coefficients'].append(coefficients)
        if self.CV_boolean is True:
            self.state['C'].append(m.C_)
        self.state['intercept'].append(intercept)

    def prepare_new_batch(self, batch_index):
        """Class method documentation comes here."""
        params = self.params_grid[batch_index]
        names = self.parameter_names
        batch = {p: params[i] for i, p in enumerate(names)}
        return batch

    def _generate_params_grid(self):
        """Class method documentation comes here."""
        raise NotImplementedError

    def _resolve_params_grid(self, params_grid):
        """Class method documentation comes here."""
        if params_grid is None:
            # TODO: add logging.
            return self._generate_params_grid()
        elif isinstance(params_grid, np.ndarray) and len(params_grid.shape) == 2:
            return params_grid
        else:
            # TODO: add raised text.
            raise TypeError('Put some text here!')

    # def _get_summary_values(self, observed=False, batch=None):
    #    """Returns the summary statistics values of given data.

    #    Parameters:
    #    ----------
    #    observed : boolean
    #        If true the function returns the summary statistcs of the observed data,
    #        if false the function returns the summary statistics of given batch.
    #    batch : numpy array
    #        The data set containing the values of Summary nodes.

    #    Returns:
    #    ----------
    #    The summary statistics of given data for each variable (#observations x #summary statistcs)
    #    """
    #    # Create the list to store the values
    #    ss_list = []
    #    if observed is True:
    #        for summary_name in self.summary_names:
    #            summary_statistics = np.array([self.model[summary_name].observed])
                # If summary statistics contain more than one value
    #            if summary_statistics.ndim > 2:
    #                for i in np.arange(summary_statistics.shape[2]):
    #                    ss_list.append(summary_statistics[:, :, i][0])
    #            elif summary_statistics.ndim == 2:
    #                ss_list.append(summary_statistics[0])
    #        return np.column_stack(ss_list)  # np.array(ss_list).T

    #    elif observed is False:
    #        if batch is None:
    #            raise NotImplementedError("You need to give a batch.")
    #        for summary_name in self.summary_names:
    #            summary_statistics = np.array([batch[summary_name]])
                # If summary statistics contain more than one value
    #            if summary_statistics.ndim > 2:
    #                for i in np.arange(summary_statistics.shape[2]):
    #                    ss_list.append(summary_statistics[:, :, i][0])
    #            elif summary_statistics.ndim == 2:
    #                ss_list.append(summary_statistics[0])
    #        return np.column_stack(ss_list)  # np.array(ss_list).T

    def _generate_marginal(self):
        """Class method documentation comes here."""
        batch = self.model.generate(self.batch_size)
        # marginal = self._get_summary_values(batch=batch)
        marginal = [batch[summary_name] for summary_name in self.summary_names]
        marginal = np.column_stack(marginal)
        return marginal

    def _resolve_marginal(self, marginal):
        """Class method documentation comes here."""
        if marginal is None:
            marginal = self._generate_marginal()
            logger.info('New marginal data ({} x {}) are generated.'.format(marginal.shape[0],
                                                                            marginal.shape[1]))
            return marginal
        elif isinstance(marginal, np.ndarray) and len(marginal.shape) == 2:
            return marginal
        else:
            # TODO: add raised text.
            raise TypeError('Put some text here!')

    def _get_summary_names(self):
        """Get the names of summary statistics."""
        summary_names = []
        for node in self.model.nodes:
            if isinstance(self.model[node], Summary) and not node.startswith('_'):
                summary_names.append(node)
        return summary_names

    def _get_observed_summary_values(self):
        """Get summary statistic values for observed data."""
        # observed_ss = self._get_summary_values(observed=True)
        observed_ss = [self.model[summary_name].observed for summary_name in self.summary_names]
        observed_ss = np.column_stack(observed_ss)
        return observed_ss
