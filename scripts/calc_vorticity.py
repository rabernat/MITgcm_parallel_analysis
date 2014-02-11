import numpy as np
import pylab as plt
from IPython.parallel import Client
import os
import sys

# add parent directory to python path
# (because we do not "install" the llc module)
sys.path.append('..')
from llc import llc_model

#base_dir = os.path.join(os.environ['LLC'], 'llc_1080')
#LLC = llc_model.LLCModel1080(
base_dir = os.path.join(os.environ['LLC'], 'llc_4320')
LLC = llc_model.LLCModel4320(
        #data_dir = os.path.join(base_dir, 'run_day732_896'),
        data_dir = os.path.join(base_dir, 'MITgcm/run'),
        grid_dir = os.path.join(base_dir, 'grid'))

# set make this True to use parallel execution
if True:
    # connect to ipcluster server
    c = Client(profile='mpi')
    dview = c.direct_view()
    # there has to be a way around doing this
    lbv = c.load_balanced_view()
    mapfunc = lbv.map_async
else:
    # just use serial execution
    mapfunc = map
    
#iter = 777480
# this is where the work gets done
def work_on_tile(tile):
    # need to reimport the modules
    # there must be a cleaner way to do this, but I don't know how
    from llc import llc_model
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    iter = 221760
    figdir = 'figures/tiles/vorticity/'
    
    # load grid data
    tile.load_geometry()
    
    # load velocity files
    U = tile.load_data('U.%010d.data' % iter )
    V = tile.load_data('V.%010d.data' % iter )

    # get a mask for the surface
    mask = (tile.hfac['C'][0]==0.)[np.newaxis,:,:]
    #mask = (tile.depth[0]==0.)[np.newaxis,:,:]

    if mask.all():
        return None
    else:
        # integrate over top five levels
        Us = np.ma.masked_array(
            tile.average_vertical(U, krange=np.r_[:5]), mask)
        Vs = np.ma.masked_array(
            tile.average_vertical(V, krange=np.r_[:5]), mask)

        # calculate vorticity
        # pad for plotting
        vort = np.ma.masked_array(np.zeros((tile.Ny,tile.Nx)), True)
        vort[1:,1:] = tile.ra['Z'][:, 1:, 1:]**-1 * ( 
            tile.delta_i( tile.dy['C'] * Vs )[:, 1:, :] -
            tile.delta_j( tile.dx['C'] * Us )[:, :, 1:]
        )
       
        # only plot if there are unmasked results
        if (~vort.mask).sum() > 50:
            pc_out = tile.pcolormesh(vort, figdir + 'vort_%04d.png' % tile.id,
             proj=True, clim=[-1e-4, 1e-4], cmap=plt.get_cmap('bwr'))
        else:
            pc_out = None

        return (tile.id, tile.Nface, tile.lonaxis,
                np.sqrt(vort**2).mean(), pc_out)

i = 0
for vort_mean in mapfunc(work_on_tile, LLC.get_tile_factory()):
    print i, vort_mean
    i += 1

    
    
