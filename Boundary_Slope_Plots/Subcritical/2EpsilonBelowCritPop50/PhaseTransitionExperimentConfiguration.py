# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 17:27:56 2020

@author: lnajt
"""


import sys
# insert at 1, 0 is the script path (or '' in REPL)
#sys.path.insert(1, '../')

import PhaseTransitionExperimentCode


mu = 2.63815853
bases = [3.125] # ( results in 1/mu - .6, roughly, 1/mu = .379 and the inverse of this is .32
pops = [.5]
time_between_outputs = 1000000
total_run_length =     200000000
PhaseTransitionExperimentCode.run_experiment(bases,  pops, time_between_outputs, total_run_length)
