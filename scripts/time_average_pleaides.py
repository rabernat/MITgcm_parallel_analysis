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
tave_output_dir = '/nobackupp8/rpaberna/DATASTORE.RPA/projects/llc/llc_4320/tave'

base_dir = os.path.join(os.environ['LLC'], 'llc_4320')
LLC = llc_model.LLCModel4320(
        data_dir = os.path.join(base_dir, 'MITgcm/run'),
        grid_dir = os.path.join(base_dir, 'grid'))

dt = 144 * 24
iters = np.arange(10368,279360+1,dt)
varnames = ['Theta', 'Salt', 'U', 'V', 'W']

# the parallel interface
c = Client(profile='mpi')
dview = c[:]
dview.block = True
# import necessary modules
dview.execute('import sys')
dview.execute("sys.path.append('..')")
dview.execute("from llc import llc_model")

Nprocs = len(dview)

GB = 1073741824
averager = LLC.get_time_averager_factory(
                varnames=varnames,
                iters=iters,
                Nprocs=Nprocs,
                output_dir=tave_output_dir,
                maxmem=3*GB)

# send out the segments to be processed
dview.scatter('ae', averager.engines)
dview.block = False
ar = dview.execute('ae[0].process()')


