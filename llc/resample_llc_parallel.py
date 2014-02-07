import numpy as np
import pylab as plt
from IPython.parallel import Client
from itertools import imap
import llc_worker
import os

base_dir = os.path.join(os.environ['LLC'], 'llc_1080')
LLC = llc_worker.LLCModel1080(
        data_dir = os.path.join(base_dir, 'run_day732_896'),
        grid_dir = os.path.join(base_dir, 'grid'))

c = Client(profile='default')
dview = c.direct_view()
lbv = c.load_balanced_view()
    
def work_on_tile(tile):
    import llc_worker
    import numpy as np
    # only work on lat-lon tiles
    #if tile.Nface < 4:
    if True:
        depth = np.ma.masked_equal(tile.load_grid('Depth.data', zrange=0),0)
        if ( depth > 0.).any():
            res = tile.pcolormesh(depth, 'tile_images/depth_%04d.png' % tile.id, clim=[0,6000])
            return (tile.id, res)
        return None

for res in lbv.map_async(work_on_tile, LLC.get_tile_factory()):
#for res in map(work_on_tile, LLC.get_tile_factory()):
    if res is not None:
        tid = res[0]
        bounds = np.array(res[1][0])
        figsize = np.array(res[1][1])
        print 'TILE ID: %g' % tid
        print 'bounds: '
        print bounds
        print 'figsize: '
        print figsize
    

    
    
