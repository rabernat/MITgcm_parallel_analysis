import numpy as np
import os

class LLCModel:
    """The parent object that describes a whole MITgcm Lat-Lon Cube setup."""

    def __init__(self, Nfaces=None, Nside=None, Ntop=None, Nz=None,
        data_dir=None, grid_dir=None, default_dtype=np.dtype('>f4'),
        L=20037508.342789244):

        self.Nfaces = Nfaces
        self.Nside = Nside
        self.Ntop = Ntop
        self.Nz = Nz
        self.Nxtot = 4*Nside + Ntop # the total X dimension of the files
        self.Nxlon = 4*Ntop # the number of points around a latitude circle
        # the total number of values in a single variable
        self.Ntot = self.Nz*self.Nxtot*self.Ntop
        self.dtype = default_dtype
        self.L = L
        
        # default to working directory
        if data_dir is None:
            data_dir = '.'
        if grid_dir is None:
            grid_dir = '.'
        self.data_dir = data_dir
        self.grid_dir = grid_dir   
        
        # default grid layout within output files (I hope all LLC setups are like this)
        self.facedims=np.array([
            (Nside, Ntop), # first LL face
            (Nside, Ntop), # second LL face
            (Ntop, Ntop),  # cap face
            (Ntop, Nside), # third LL face (transposed)
            (Ntop, Nside)] # fourth LL face (transposed)
            )
            
        # whether to reshape the face
        self.reshapeface = [False,False,False,True,True]
        # which axis is longitude
        self.lonaxis = [2,2,2,1,1]
        # put the cap face at the end
        self.faceorder = [0,1,3,4,2]
        
        # a name for the run
        self.name = 'llc_%04d' % self.Ntop         

    def _facedim(self,Nface):
        return self.facedims[self.faceorder[Nface]]
        
    def _lonaxis(self,Nface):
        return self.lonaxis[self.faceorder[Nface]]
        
    # for just getting a direct memmap    
    def load_data_raw(self, fname):
        return np.memmap(
                os.path.join(self.data_dir, fname), 
                mode='r', dtype=self.dtype)

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
        
    def get_time_averager_factory(self, **kwargs):
        return LLCTimeAveragerFactory(self, **kwargs)
        
# extend the basic class for the specific grids
class LLCModel4320(LLCModel):
    """LLC grid for 4230 domain size."""
    def __init__(self, *args, **kwargs):
        LLCModel.__init__(self,
            Nfaces=5, Nside=12960, Ntop=4320, Nz=90,
            *args, **kwargs)

class LLCModel2160(LLCModel):
    """LLC grid for 2160 domain size."""
    def __init__(self, *args, **kwargs):
        LLCModel.__init__(self,
            Nfaces=5, Nside=6480, Ntop=2160, Nz=90,
            *args, **kwargs)

class LLCModel1080(LLCModel):
    """LLC grid for 1080 domain size."""
    def __init__(self, *args, **kwargs):
        LLCModel.__init__(self,
            Nfaces=5, Nside=3240, Ntop=1080, Nz=90,
            *args, **kwargs)

class LLCTimeAveragerFactory:
    
    """For the simple/easy job of time averaging."""
    def __init__(self, llc_model_parent, varnames=[], iters=[],
                    Nprocs=1,
                    output_dir=None, maxmem=`1073741824`):
        self.llc = llc_model_parent
        self.varnames = varnames
        self.Nvars = len(varnames)
        self.iters = iters
        self.Niters = len(iters)
        self.output_dir = output_dir
        self.Nprocs = Nprocs

        if np.mod(self.llc.Ntot, self.Nprocs) != 0:
            raise ValueError('The variable size is not divisible by the number of processes')
        # the number of items in each engine's variable
        self.Nitems = self.llc.Ntot / self.Nprocs

        memneeded = self.Nitems * self.Nvars  * self.llc.dtype.itemsize
        if memneeded > maxmem:
            raise MemoryError(
             'The operation will require more memory (%g) that the maximum allowed per process (%g)' % (memneeded,maxmem))
        
        # set up output files
        self.output_files = dict()
        for v in varnames:
            fname = os.path.join(self.output_dir,'%s.TAVE.data' % v)
            # create the file
            mm = np.memmap(fname, dtype=self.llc.dtype, mode='w+', shape=(self.llc.Ntot,))
            del mm # flush the buffer to file
            self.output_files[v] = fname
            
        # initialize engines
        self.engines = []
        for n in range(self.Nprocs):
            self.engines.append(
                LLCTimeAveragerEngine(self, n)
            )

class LLCTimeAveragerEngine:
    
    def __init__(self, parent, n):
        self.parent = parent
        self.dtype = self.parent.llc.dtype
        self.Nitems = self.parent.Nitems
        self.offset = self.Nitems * n * self.dtype.itemsize
                
    def process(self, use_memmap=False):
        # set up accumulators
        count = 0
        avg_vars = dict()
        for v in self.parent.varnames:
            avg_vars[v] = np.zeros((self.Nitems,), self.dtype)
        
        for i in self.parent.iters:
            for v in self.parent.varnames:
                print 'Processing %s %010d offset %12i' % (v,i,self.offset)
                if use_memmap:
                    mm = np.memmap(
                      os.path.join(self.parent.llc.data_dir,
                       '%s.%010d.data' % (v, i)),
                       dtype = self.dtype,
                       mode = 'r',
                       shape = (self.Nitems,),
                       offset = self.offset
                    )
                else:
                    f = open(
                      os.path.join(self.parent.llc.data_dir,
                      '%s.%010d.data' % (v, i)), 'r' )
                    f.seek(self.offset)
                    mm = np.fromfile(f, dtype=self.dtype,
                                count=self.Nitems)
                    f.close()
                print '  variable size %12i' % len(mm)
                avg_vars[v] += mm
                del mm
            count += 1
        
        # write
        for v in self.parent.varnames:
            print 'Outputting %s offset %12i' % (self.parent.output_files[v],self.offset)
            mm = np.memmap(
               self.parent.output_files[v],
               dtype = self.dtype,
               mode = 'r+',
               shape = (self.Nitems,),
               offset = self.offset
            )
            mm[:] = avg_vars[v] / count
            del mm            

class LLCTileFactory:
    """Has generator for splitting domain into tiles."""
    
    def __init__(self, llc_model_parent, tileshape=(540,540)):
        self.llc = llc_model_parent
        self.tileshape = tileshape
        self.tiledim = []
        print 'Using tile shape %g x %g' % tileshape
        for n in range(self.llc.Nfaces):
            dims = self.llc._facedim(n)
            # make sure the shapes are compatible
            if np.mod(dims[0],tileshape[0]) or np.mod(dims[1],tileshape[1]):
                raise ValueError('Tile shape is not compatible with face dimensions')
            tdims = (dims[0]/tileshape[0], dims[1]/tileshape[1])
            self.tiledim.append( tdims )
            print ' face %g: %g x %g tiles' % (n, tdims[0], tdims[1])
        self.tiledim = np.array(self.tiledim)
        self.Ntiles = self.tiledim.prod(axis=1).sum()
        print 'Total tiles: %g' % self.Ntiles
        self._reset_iterator()
        
    def _reset_iterator(self):
        # indices for iterator
        self._idx_face = 0
        self._idx_x = 0
        self._idx_y = 0
        self._ntile = 0
    
    def __iter__(self):
        return self
        
    def next(self):
        if self._idx_x==self.tiledim[self._idx_face][1]:
            self._idx_x = 0
            self._idx_y += 1
        if self._idx_y==self.tiledim[self._idx_face][0]:
            self._idx_y = 0
            self._idx_face += 1
        if self._idx_face==self.llc.Nfaces:
            raise StopIteration
        xlims = self.tileshape[0] * np.r_[self._idx_x,self._idx_x+1]
        ylims = self.tileshape[1] * np.r_[self._idx_y,self._idx_y+1]
        self._idx_x += 1
        self._ntile += 1
        return LLCTile(self.llc, self._idx_face,
                        ylims, xlims, self._ntile-1)
    
    def get_tile(self, Ntile):
        self._reset_iterator()
        for t in self:
            if Ntile == t.id:
                break
        return t
        
# a utility function
def latlon_to_meters((lat,lon)):
    """Converts given lat/lon in WGS84 Datum to XY in Spherical Mercator EPSG:900913"""
    a = 6378137.
    originShift = 2 * np.pi * a / 2.
    mx = lon * originShift / 180.0
    my = np.log( np.tan((90 + lat) * np.pi / 360.0 )) / (np.pi / 180.0)
    my = my * originShift / 180.0
    return mx, my
    
class LLCTile:
    """This class describes a usable subregion of the LLC model"""
    
    def __init__(self, llc_model_parent, Nface, ylims, xlims, tile_id):
        self.llc = llc_model_parent
        self.Nface = Nface
        self.ylims = ylims
        self.xlims = xlims
        self.Nx = xlims[1] - xlims[0]
        self.Ny = ylims[1] - ylims[0]
        self.Nz = self.llc.Nz
        self.id = tile_id
        self.shape = (self.llc.Nz, self.Ny, self.Nx)
        self.lonaxis = self.llc._lonaxis(Nface)
    
    def load_grid(self, fname, **kwargs):
        return self.load_data(fname, grid=True, **kwargs)
    
    def load_data(self, fname, grid=False):
        if grid:
            loadfunc = self.llc.load_grid_file
        else:
            loadfunc = self.llc.load_data_file    
        data = loadfunc(fname, self.Nface)
        if (data.shape[1]==1) and (data.shape[2]==1):
            # its a 1d vertical file
            return data.view(np.ndarray)
        else:
            return data[
                :, self.ylims[0]:self.ylims[1], self.xlims[0]:self.xlims[1] ].view(np.ndarray)

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
    
        
        
        
    
        
        
        
        
        
