# -*- coding: utf-8 -*-
"""
Contributors:
    - Louis Rémus
"""
import numpy as np


# Decorator to have metric ignore NaNs
def nan_ignoring(metric):
    """
    Decorator to make metric ignore NaNs

    Args:
        metric (callable): metric to modify

    Returns:
        nan-ignoring metric function
    """

    def nan_wrapper(y_true, y_estimated):
        """
        Wrapping function

        Args:
            y_true (array-like):
            y_estimated (array-like):

        Returns:
            nan-ignoring metric function value for (y_true, y_estimated)
        """
        # Boolean mask
        # True is y_true or y_estimated are NaN
        # False otherwise
        nan_mask = np.logical_or(np.isnan(y_true), np.isnan(y_estimated))
        # New arguments to the metric function: values without NaNs
        kwargs = {
            'y_true': y_true[~nan_mask],
            'y_estimated': y_estimated[~nan_mask]
        }
        return metric(**kwargs)

    return nan_wrapper
