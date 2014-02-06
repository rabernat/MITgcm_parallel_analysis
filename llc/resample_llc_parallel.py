import numpy as np
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
    if tile.Nface < 4:
        depth = np.ma.masked_equal(tile.load_grid('Depth.data', zrange=0),0)
        if ( depth > 0.).any():
            return tile.export_geotiff(depth, 'tile_images/depth')
        return None

for res in lbv.map_async(work_on_tile, LLC.get_tile_factory()):
    print res
    #print 'Tile mean position: ', res

