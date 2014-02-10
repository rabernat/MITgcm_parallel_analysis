import numpy as np
import pylab as plt
from IPython.parallel import Client
import os
import sys

# add parent directory to python path
# (because we do not "install" the llc module)
sys.path.append('..')
from llc import llc_model

base_dir = os.path.join(os.environ['LLC'], 'llc_1080')
LLC = llc_model.LLCModel1080(
#base_dir = os.path.join(os.environ['LLC'], 'llc_4320')
#LLC = llc_worker.LLCModel4320(
        data_dir = os.path.join(base_dir, 'run_day732_896'),
        grid_dir = os.path.join(base_dir, 'grid'))

# set make this True to use parallel execution
if False:
    # connect to ipcluster server
    c = Client(profile='default')
    dview = c.direct_view()
    lbv = c.load_balanced_view()
    mapfunc = lbv.map_async
else:
    # just use serial execution
    mapfunc = map
    
# this is where the work gets done
def work_on_tile(tile):
    # need to reimport the modules
    # there must be a cleaner way to do this, but I don't know how
    try:
        from llc import llc_model
    except ImportError:
        sys.path.append('..')
        from llc import llc_model
    import numpy as np
    depth = np.ma.masked_equal(tile.load_grid('Depth.data'),0)
    return depth.mean()

tile_depths = []
for meandepth in mapfunc(work_on_tile, LLC.get_tile_factory()):
    tile_depths.append(meandepth)

print 'Mean depth: ', np.ma.masked_array(tile_depths).mean()
    

    
    
