#!/usr/bin/env python

import fathon
from fathon import fathonUtils as fu


def do_mfdfa(data: list, win_sizes: list, q_list: list, rev_seg: bool, pol_order: int):
    # zero-mean cumulative sum of data
    data_cs = fu.toAggregated(data)

    # init mfdfa object
    pymfdfa = fathon.MFDFA(data_cs)

    # compute fluctuation function and generalized Hurst exponents
    n, F = pymfdfa.computeFlucVec(win_sizes, q_list, rev_seg, pol_order)
    list_H, list_H_intercept = pymfdfa.fitFlucVec()

    # compute mass exponents
    tau = pymfdfa.computeMassExponents()

    # compute multifractal spectrum
    alpha, mfSpect = pymfdfa.computeMultifractalSpectrum()

    return n, F, list_H, list_H_intercept, tau, alpha, mfSpect