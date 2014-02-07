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

#c = Client(profile='default')
#dview = c.direct_view()
#lbv = c.load_balanced_view()
    
def work_on_tile(tile):
    import llc_worker
    import numpy as np
    # only work on lat-lon tiles
    if tile.Nface < 4:
        depth = np.ma.masked_equal(tile.load_grid('Depth.data', zrange=0),0)
        tid = np.ma.masked_array(tile.id*np.ones(depth.shape), depth.mask)
        if ( depth > 0.).any():
            geom_a,geom_b,data_regrid =  tile.export_geotiff(tid, 'tile_images/depth')
            if tile.id==30:
                plt.figure(2)
                plt.pcolormesh(np.ma.masked_array(data_regrid))
            return ((tile.Nface,tile.id),(geom_a, geom_b))
        return None

L = 20037508.
fig = plt.figure()
ax = plt.axes()
N = 540
ax.set_xlim([-L,L]); ax.set_ylim([-L,L])
#for res in lbv.map_async(work_on_tile, LLC.get_tile_factory()):
for res in map(work_on_tile, LLC.get_tile_factory()):
    print res
    if res is not None:
        id,Nface = res[0]
        geom_a,geom_b = res[1]
        x0,y0,dx,dy = np.array(geom_a)[np.r_[0,3,1,5]]
        ax.plot([x0, x0 + N*dx, x0 + N*dx, x0, x0],[y0, y0, y0 + N*dy, y0 + N*dy, y0])
        ax.text( x0 + N*dx/2, y0 + N*dy/2, '%g (%ga)' % (id, Nface), ha='center')
        if geom_b is not None:
            x0,y0,dx,dy = np.array(geom_b)[np.r_[0,3,1,5]]
            ax.plot([x0, x0 + N*dx, x0 + N*dx, x0, x0],[y0, y0, y0 + N*dy, y0 + N*dy, y0])
            ax.text( x0 + N*dx/2, y0 + N*dy/2, '%g (%gb)' % (id, Nface), ha='center')
            
        plt.draw()
        plt.show()
        plt.pause(0.1)
