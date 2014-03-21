import numpy as np
import os
from MITgcmutils import mds
    
class MITgcmModel(object):
    """The parent object that describes a generic MITgcm setup."""

    def __init__(self, Nx=None, Ny=None, Nz=None,
        data_dir=None, grid_dir=None, default_dtype=np.dtype('>f4')):

        self.Nx = Nx
        self.Ny = Ny
        self.Nz = Nz
        self.dtype = default_dtype
        
        # default to working directory
        if data_dir is None:
            data_dir = '.'
        if grid_dir is None:
            grid_dir = '.'
        self.data_dir = data_dir
        self.grid_dir = grid_dir   
        
    def load_data_file(self, fname, *args):
        return self.memmap_face(
            os.path.join(self.data_dir, fname), *args)

    def load_grid_file(self, fname, *args):
        return self.memmap_face(
            os.path.join(self.grid_dir, fname), *args)

    def memmap_face(self, fname, Nface):
        """Returns a memmap to the requested face"""

        # figure out the size of the file
        varsize = os.path.getsize(fname) / self.dtype.itemsize
        
        is_vertical = True
        # for vertical coordinates
        if varsize == self.Nz:
            mmshape = (self.Nz, 1, 1)
        elif varsize == (self.Nz+1):
            mmshape = (self.Nz+1, 1, 1)
        else:
            is_vertical = False
            Nz = varsize / self.Ntop / self.Nxtot
            if Nz==1 or Nz==self.Nz:
                mmshape = (Nz, self.Nxtot, self.Ntop)
            else:
                raise IOError('File %s is the wrong size' % fname)

        # read the data as a memmap
        mm = np.memmap(fname, mode='r', dtype=self.dtype,
                    order='C', shape=mmshape)
                    
        # just bail if it is a vertical file
        if is_vertical:
            return mm

        # true face index
        N = self.faceorder[Nface]
        
        # the start and stop location of the face on disk
        idx_lims = np.hstack([0,np.cumsum(self.facedims.prod(axis=1)/self.Ntop)])
        mm = mm[:,idx_lims[N]:idx_lims[N+1]]
        dims = self.facedims[N]
        if self.reshapeface[N]:
            # needs to be reshaped
            mm = mm.reshape((Nz,self.Ntop,self.Nside), order='C')
            # but not transposed
            #mm = mm.transpose((0,2,1))
            #mm = mm[:,::-1,:]
        return mm
        
    def describe_faces(self):
        for n in range(self.Nfaces):
            xc = self.load_grid_file('XC.data',n)
            yc = self.load_grid_file('YC.data',n)
            print 'Face %g:' % n
            print ' lower left  (XC=% 6.2f, YC=% 6.2f)' % (xc[0,0,0],yc[0,0,0])
            print ' lower right (XC=% 6.2f, YC=% 6.2f)' % (xc[0,0,-1],yc[0,0,-1])
            print ' upper left  (XC=% 6.2f, YC=% 6.2f)' % (xc[0,-1,0],yc[0,-1,0])
            print ' upper right (XC=% 6.2f, YC=% 6.2f)' % (xc[0,-1,-1],yc[0,-1,-1])
            
    def get_tile_factory(self, **kwargs):
        return LLCTileFactory(self, **kwargs)
    
class MITgcmTile(object):
    """This class describes a locally regular portion of an MITgcm model grid"""
    
    def __init__(self, ylims, xlims):
        self.llc = llc_model_parent
        self.Nface = Nface
        self.ylims = ylims
        self.xlims = xlims
        self.Nx = xlims[1] - xlims[0]
        self.Ny = ylims[1] - ylims[0]
        self.id = tile_id
        self.shape = (self.llc.Nz, self.Ny, self.Nx)
        self.lonaxis = self.llc._lonaxis(Nface)
    
    def load_grid(self, fname, **kwargs):
        return self.load_data(fname, grid=True, **kwargs)
    
    # this class should be overwritten by the llc methods
    def load_data(self, fname, grid=False):
        if grid:
            loadfunc = self.llc.load_grid_file
        else:
            loadfunc = self.llc.load_data_file    
        data = loadfunc(fname, self.Nface)
        if (data.shape[1]==1) and (data.shape[2]==1):
            # its a 1d vertical file
            return data
        else:
            return data[
                :, self.ylims[0]:self.ylims[1], self.xlims[0]:self.xlims[1] ]

    def load_geometry(self):
        """This loads the grid geometry into local variables.
        But they aren't actually read from disk until you try to access them"""

        # horizontal grid
        self.x = dict(
         C = self.load_grid('XC.data'),
         G = self.load_grid('XG.data')  )
        self.y = dict(
         C = self.load_grid('YC.data'),
         G = self.load_grid('YG.data')  )
        self.dx = dict(
         C = self.load_grid('DXC.data'),
         G = self.load_grid('DXG.data') )
        self.dy = dict(
         C = self.load_grid('DYC.data'),
         G = self.load_grid('DYG.data') )
            
        # vertical grid
        self.r = dict(
         C = self.load_grid('RC.data'),
         F = self.load_grid('RF.data') )
        self.dr = dict(
         C = self.load_grid('DRC.data'),
         F = self.load_grid('DRF.data') )
        self.depth = self.load_grid('Depth.data')
        
        self.z = self.r
        self.dz = self.dr
        
        # area information
        self.ra = dict(
         C = self.load_grid('RAC.data'),
         S = self.load_grid('RAS.data'),
         W = self.load_grid('RAW.data'),
         Z = self.load_grid('RAZ.data') )
        self.hfac = dict(
         C = self.load_grid('hFacC.data'),
         S = self.load_grid('hFacS.data'),
         W = self.load_grid('hFacW.data') )
        
    def integrate_vertical(self, data, krange=None, rpt='C', hpt='C'):
        if krange is None:
            krange = np.r_[:self.Nz]
        return np.sum( data[krange] *
            self.dz[rpt][krange] * self.hfac[hpt][krange], axis=0 )[np.newaxis,:,:]
    
    def average_vertical(self, data, **kwargs):
        return (self.integrate_vertical(data, **kwargs) / 
                self.integrate_vertical(np.ones((self.llc.Nz,1,1)), **kwargs) )
        
    # DERIVATIVES in the horizontal
    # Derivatives are problematic because in theory we need to communicate
    # with other tiles. For now, we solve this by assuming that, since the domains
    # are so huge, we can throw away results at the edge of the tile.
    # The real solution is to implement a halo.
    
    # try to define the derivatives using the same notation as MITgcm documentation
    def delta_i(self, data):
        return data[:,:,1:] - data[:,:,:-1]
    
    def delta_j(self, data):
        return data[:,1:,:] - data[:,:-1,:]
            
    def pcolormesh(self, data, fname, proj=True, clim=None, shade=False, **kwargs):
        """Output a tile that can be turned into a map"""
        if data.shape != (self.Ny,self.Nx):
            raise ValueError('Only 2D data of the correct shape can be pcolored')
        
        # import plotting modules
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.colors import LightSource
        
        # load both corner and centers, necessary for pcolor
        lon_c = self.x['C'][0]
        lat_c = self.y['C'][0]
        lon_g = self.x['G'][0]
        lat_g = self.y['G'][0]
        #lon_c = self.load_grid('XC.data', zrange=0)
        #lat_c = self.load_grid('YC.data', zrange=0)
        #lon_g = self.load_grid('XG.data', zrange=0)
        #lat_g = self.load_grid('YG.data', zrange=0)
        
        # wrap if necessary
        wrap_flag =  (np.diff(lon_g,axis=self.lonaxis-1) < -180).any()
        if wrap_flag:
            lon_c = np.copy(lon_c)
            lon_g = np.copy(lon_g)
            lon_g[lon_g < 0.] += 360.          
            lon_c[lon_c < 0.] += 360.          
        
        # create bounds for pcolor
        dx_lon = 2*(lon_c[:,-1] - lon_g[:,-1])
        dy_lat = 2*(lat_c[-1,:] - lat_g[-1,:])
        lon_quad = np.zeros((self.Ny+1, self.Nx+1), self.llc.dtype)
        lat_quad = np.zeros((self.Ny+1, self.Nx+1), self.llc.dtype)
        lon_quad[:self.Ny, :self.Nx] = lon_g
        lat_quad[:self.Ny, :self.Nx] = lat_g
        lon_quad[:self.Ny, -1] = lon_g[:,-1] + dx_lon
        lon_quad[-1, :self.Nx] = lon_g[-1,:]
        lon_quad[-1, -1] = lon_quad[-2, -1]
        lat_quad[-1, :self.Nx] = lat_g[-1,:] + dy_lat
        lat_quad[:self.Ny, -1] = lat_g[:,-1]
        lat_quad[-1, -1] = lat_quad[-1, -2]

        if proj:
            x_quad, y_quad = latlon_to_meters((lat_quad, lon_quad))
            x_c, y_c = latlon_to_meters(
                (np.ma.masked_array(lat_c, data.mask),
                 np.ma.masked_array(lon_c, data.mask)))
            dx = 2*self.llc.L / (self.llc.Ntop*4)
            xlim_true, ylim_true = self.llc.L, self.llc.L
        else:
            x_quad, y_quad = lon_quad, lat_quad    
            x_c = np.ma.masked_array(lon_c, data.mask)
            y_c = np.ma.masked_array(lat_c, data.mask)
            dx = 360. / (self.llc.Ntop*4)
            xlim_true, ylim_true = 180.,90.

        # bounds for the output image
        pad = 5*dx
        x_min, x_max = x_c.min()-pad, x_c.max()+pad
        y_min = max(y_c.min()-pad, -ylim_true)
        y_max = min(y_c.max()+pad, ylim_true)
        
        dpi = 80.
        figsize = (x_max-x_min)/dx/dpi, (y_max-y_min)/dx/dpi
        
        try:
            
            fig = plt.figure(figsize=figsize, dpi=dpi)
            ax = fig.add_axes([0,0,1,1])
            ax.set_xlim((x_min, x_max))
            ax.set_ylim((y_min, y_max))
            ax.set_axis_off()
            pc = ax.pcolormesh(x_quad, y_quad, data, **kwargs)
            if clim is not None:
                pc.set_clim(clim)    
            fig.savefig(fname, dpi=dpi, figsize=figsize, transparent=True)
            
            figfiles = [fname]
            # generate another image that is for shading
            if shade:
                # draw using grayscale
                suff = fname[-4] == '.png'
                new_fname = fname[:-4] + "_bw.png"
                pc.remove()
                pc = ax.pcolormesh(x_quad, y_quad, data, cmap=plt.cm.binary)
                if clim is not None:
                    pc.set_clim(clim)
                fig.savefig(new_fname, dpi=dpi, figsize=figsize, transparent=True)
                figfiles.append(new_fname)
            plt.close(fig)
        
            # write world file
            for filename in figfiles:
                wf = open('%sw' % filename, 'w')
                wf.write('%10.9f \n' % dx) # pixel X size
                wf.write('%10.9f \n' % 0.) # rotation about x axis
                wf.write('%10.9f \n' % 0.) # rotation about y axis
                wf.write('%10.9f \n' % -dx) # pixel Y size
                wf.write('%10.9f \n' % x_min) # X coordinate of upper left pixel center
                wf.write('%10.9f \n' % y_max) # Y coordinate of upper left pixel center
                wf.close()
        except ValueError:
            pass

        
        return ((x_min,y_min,x_max,y_max),figsize)
    
        
        
        
    
        
        
        
        
        
