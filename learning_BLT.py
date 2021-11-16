"""
    File name: learning_BLT.py
    Author: Katja Brand
    Created: 28/10/2021
    Description: Contains learning functions used to run BLT models.
"""
import numpy as np
import theano.tensor as T

def rescorla_wagner(o, t, v, alpha):
    """
    A simple Rescorla-Wagner model. NOTE: copied from DMpy.learning for reference

    Args:
        o: outcome
        v: prior value
        alpha: learning rate

    Returns:
        value: Value on current trial
        pe: prediction error
    """
    pe = o - v
    value = v + alpha * pe

    return (value, pe)

def dual_lr_rw(o, t, s, v, alpha_p, alpha_n):
    """
    Rescorla-Wagner model adapted with a dual learning rate based on type of stimulus used in trial.

    Args:
        o: Trial outcome
        t: Trial number
        s: Stimulus type used in trial
        v: Value on previous trial
        alpha_p: Learning rate for positive stimuli
        alpha_n: Learning rate for negative stimuli

    Returns:
        value: Value on current trial
        pe: prediction error
        weighted_pe: prediction error weighted by learning rate
    """

    pe = o - v

    weighted_pe = T.switch(T.eq(s, 0), alpha_p * pe, alpha_n * pe) # if stim == 0, use alpha_p

    value = v + weighted_pe

    return (value, pe, weighted_pe)
