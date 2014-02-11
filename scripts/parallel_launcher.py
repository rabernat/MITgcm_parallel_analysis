import os
import sys

# add parent directory to python path
# (because we do not "install" the llc module)
sys.path.append('..')
from llc import llc_model

# the lower resoltuion
base_dir_1080 = os.path.join(os.environ['LLC'], 'llc_1080')
LLC1080 = llc_model.LLCModel1080(
    data_dir = os.path.join(base_dir_1080, 'run_day732_896'),
    grid_dir = os.path.join(base_dir_1080, 'grid'))

# higher resolution
base_dir_4320 = os.path.join(os.environ['LLC'], 'llc_4320')
LLC4320 = llc_model.LLCModel4320(
        data_dir = os.path.join(base_dir_4320, 'MITgcm/run'),
        grid_dir = os.path.join(base_dir_4320, 'grid'))

# set make this True to use parallel execution
try:
    from IPython.parallel import Client
    # connect to ipcluster server
    c = Client(profile='mpi')
    dview = c.direct_view()
    # there has to be a way around doing this
    lbv = c.load_balanced_view()
    mapfunc = lbv.map_async
except ValueError:
    # just use serial execution
    print "Couldn't connect to ipcluster. Using serial execution."
    mapfunc = map
    
# import whatever module will do the work
# it must define a function called work_on_tile(tile)
import plot_theta

for result in mapfunc(plot_theta.work_on_tile, LLC1080.get_tile_factory()):
    print result

    
    
