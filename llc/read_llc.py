import numpy as np
from IPython.parallel import Client
from itertools import imap
import llc_worker
import os

base_dir = '/Volumes/Bucket1/llc/llc_1080'
LLC = llc_worker.LLCModel1080(
        data_dir = os.path.join(base_dir, 'run_day732_896'),
        grid_dir = os.path.join(base_dir, 'grid'))

c = Client(profile='default')
dview = c.direct_view()
lbv = c.load_balanced_view()
    
def work_on_tile(tile):
    import llc_worker
    pos = tile.get_mean_position()
    return pos

for res in lbv.map_async(work_on_tile, LLC.get_tile_factory()):
    print 'Tile mean position: ', res

