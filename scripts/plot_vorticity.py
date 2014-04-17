from IPython.parallel import Client
#################
# Cluster Setup #
#################
# connect to ipcluster server
c = Client(profile='mpi')
# direct view for accessing all engines
dview = c.direct_view()
# sync imports
with dview.sync_imports():
    import numpy as np
    import pylab as plt
    import os
    import sys
    from llc import llc_model
# load balaned view    
lbv = c.load_balanced_view()

###############
# Model Setup #
###############
import os
base_dir = os.path.join(os.environ['LLC'], 'llc_4320')
LLC = llc_model.LLCModel4320(
        data_dir = os.path.join(base_dir, 'MITgcm/run'),
        grid_dir = os.path.join(base_dir, 'grid'))

########################
# The Analysis Program #
########################
def work_on_tile(tile):
    
    iter = 221760
    figdir = 'figures/tiles/vorticity/'
    
    # load grid data
    tile.load_geometry()
    
    # load velocity files
    U = tile.load_data('U.%010d.data' % iter )
    V = tile.load_data('V.%010d.data' % iter )

    # get a mask for the surface
    mask = (tile.hfac['C'][0]==0.)[np.newaxis,:,:]

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


###################################
# Map the Function to Each Engine #
###################################
for vort_mean in lbv.map_async(work_on_tile, LLC.get_tile_factory()):
    print vort_mean

    
    
