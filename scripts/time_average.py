import numpy as np
import pylab as plt
from IPython.parallel import Client
import os
import sys

# add parent directory to python path
# (because we do not "install" the llc module)
sys.path.append('..')
from llc import llc_model

# where to save the time-averaged output
tave_output_dir = '/Users/rpa/tmp'

#base_dir = os.path.join(os.environ['LLC'], 'llc_1080')
#LLC = llc_model.LLCModel1080(
base_dir = os.path.join(os.environ['LLC'], 'llc_4320')
LLC = llc_model.LLCModel4320(
        #data_dir = os.path.join(base_dir, 'run_day732_896'),
        data_dir = os.path.join(base_dir, 'MITgcm/run'),
        grid_dir = os.path.join(base_dir, 'grid'))

iters = arange(1,10,2)
varnames = ['Theta', 'Salt', 'U', 'V', 'W']

averager = LLC.get_time_averager_factory(varnames,iters,
                output_dir=tave_output_dir)

# the parallel interface
c = Client(profile='mpi')
dview = c.direct_view()

if len(dview) < averager.num_procs:
    raise ValueError('Not enough engines available.')

# send out the segments to be processed
dview.scatter('ae', averager.engines)
dview.execute('ae.process()')


