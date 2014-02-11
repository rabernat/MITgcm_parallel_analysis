# all the imports must be done inside the function
# the working directory is MITgcm_parallel_analysis (the parent of this one)

def work_on_tile(tile):
    from llc import llc_model
    import numpy as np
    import os

    # this is necessary for rendering without a display
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    iter = 777480
    figdir = '../figures/%s' % tile.llc.name
    
    # load grid data
    tile.load_geometry()
    
    # load velocity files
    T = tile.load_data('Theta.%010d.data' % iter )
    S = tile.load_data('Salt.%010d.data' % iter )

    # get a mask for the surface
    mask = (tile.hfac['C'][0]==0.)[np.newaxis,:,:]

    if mask.all():
        return None
    else:
        # integrate over top five levels
        Ts = np.ma.masked_array(
            tile.average_vertical(T, krange=np.r_[:10]), mask)
        Ss = np.ma.masked_array(
            tile.average_vertical(S, krange=np.r_[:10]), mask)
       
        # only plot if there are unmasked results
        if (~Ts.mask).sum() > 50:
            tile.pcolormesh(Ts[0], os.path.join(figdir,'theta_tiles','theta_%04d.png' % tile.id),
             shade=True, clim=[-2,30], cmap=plt.get_cmap('Spectral_r'))
            tile.pcolormesh(Ss[0], os.path.join(figdir,'salt_tiles','salt_%04d.png' % tile.id),
              shade=True, clim=[32.,38.], cmap=plt.get_cmap('gnuplot2'))

        return (tile.id, tile.Nface)

    
    
