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
            (Ntop, Nside), # first LL face
            (Ntop, Nside), # second LL face
            (Ntop, Ntop),  # cap face
            (Ntop, Nside), # third LL face (transposed)
            (Ntop, Nside)] # fourth LL face (transposed)
            )
            
        # whether to reshape the face
        self.reshapeface = [False,False,False,True,True]
        # whether to transpose to the face
        self.transposeface = [True,True,False,False,False]
        # put the cap face at the end
        self.faceorder = [0,1,3,4,2]            

    def _facedim(self,Nface):
        return self.facedims[self.faceorder[Nface]]

    def load_data_file(self, fname, *args):
        return self.memmap_face(
            os.path.join(self.data_dir, fname) *args)

    def load_grid_file(self, fname, *args):
        return self.memmap_face(
            os.path.join(self.grid_dir, fname), *args)

    def memmap_face(self, fname, Nface):
        """Returns a memmap to the requested face"""

        # figure out the size of the file
        fsize = os.path.getsize(fname)
        Nz = fsize / self.dtype.itemsize / self.Ntop / self.Nxtot
        if Nz==1 or Nz==self.Nz:
            mmshape = (Nz,self.Ntop,self.Nxtot)
        else:
            raise IOError('File %s is the wrong size' % fname)

        # read the data as a memmap
        mm = np.memmap(fname, mode='r', dtype=self.dtype,
                    order='F', shape=mmshape)

        # true face index
        N = self.faceorder[Nface]
        
        # the start and stop location of the face on disk
        idx_lims = np.hstack([0,np.cumsum(self.facedims.prod(axis=1)/self.Ntop)])
        mm = mm[:,:,idx_lims[N]:idx_lims[N+1]]
        dims = self.facedims[N]
        if self.reshapeface[N]:
            # needs to be transposed
            mm = mm.reshape((Nz,self.Nside,self.Ntop), order='F')
            mm = mm[:,::-1,:]
        if self.transposeface[N]:
            mm = mm.transpose((0,2,1))
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
        
# extend the basic class for the specific grids
class LLCModel4320(LLCModel):
    """A specific LLC grid."""
    
    def __init__(self, *args, **kwargs):
        LLCModel.__init__(self,
            Nfaces=5, Nside=12960, Ntop=4320, Nz=90,
            *args, **kwargs)
            
class LLCModel1080(LLCModel):
    def __init__(self, *args, **kwargs):
        LLCModel.__init__(self,
            Nfaces=5, Nside=3240, Ntop=1080, Nz=90,
            *args, **kwargs)

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
            tdims = (dims[1]/tileshape[1], dims[0]/tileshape[0])
            self.tiledim.append( tdims )
            print ' face %g: %g x %g tiles' % (n, tdims[0], tdims[1])
        self.tiledim = np.array(self.tiledim)
        self.Ntiles = self.tiledim.prod(axis=1).sum()
        print 'Total tiles: %g' % self.Ntiles
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
        xlims = self.tileshape[1] * np.r_[self._idx_x,self._idx_x+1]
        ylims = self.tileshape[0] * np.r_[self._idx_y,self._idx_y+1]
        self._idx_x += 1
        self._ntile += 1
        return LLCTile(self.llc, self._idx_face,
                        ylims, xlims, self._ntile-1)
    
    def get_tile(self, Ntile):
        Nface = np.argmax(cumsum(self.tiledim.prod(axis=1))>Ntile)
        # this is annoying
        

# a utility function
def latlon_to_meters((lat,lon)):
    """Converts given lat/lon in WGS84 Datum to XY in Spherical Mercator EPSG:900913"""
    a = 6378137.
    originShift = 2 * np.pi * a / 2.
    mx = lon * originShift / 180.0
    my = np.log( np.tan((90 + lat) * np.pi / 360.0 )) / (np.pi / 180.0)
    my = my * originShift / 180.0
    return mx, my

def output_gdal_geotiff(data, fname, proj4_str, geo_transform):
    from osgeo import gdal, osr    
    dst_driver = gdal.GetDriverByName("GTiff")
    srs = osr.SpatialReference()
    srs.ImportFromProj4(proj4_str)
    (Ny, Nx) = data.shape

    dst_ds = dst_driver.Create(fname, Nx, Ny, 1 , gdal.GDT_Float32)
    dst_ds.SetGeoTransform( geo_transform )
    dst_ds.SetProjection(srs.ExportToWkt())
    dst_ds.GetRasterBand(1).WriteArray(data.astype('<f4'))
    dst_ds = None


class LLCTile:
    """This class describes a usable subregion of the LLC model"""
    
    def __init__(self, llc_model_parent, Nface, ylims, xlims, tile_id):
        self.llc = llc_model_parent
        self.Nface = Nface
        self.ylims = ylims
        self.xlims = xlims
        self.Nx = xlims[1] - xlims[0]
        self.Ny = ylims[1] - ylims[0]
        self.id = tile_id
        self.shape = (self.llc.Nz, self.Ny, self.Nx)
    
    def load_grid(self, fname, **kwargs):
        return self.load_data(fname, grid=True, **kwargs)
    
    def load_data(self, fname, zrange=None, grid=False):
        if zrange is None:
            zrange = np.r_[:self.llc.Nz]
        if grid:
            loadfunc = self.llc.load_grid_file
        else:
            loadfunc = self.llc.load_data_file    
        return loadfunc(fname, self.Nface)[
                zrange, self.ylims[0]:self.ylims[1], self.xlims[0]:self.xlims[1] ]
    
    def load_latlon(self):
        if not hasattr(self, 'lon'):
            self.lon = self.load_grid('XC.data', zrange=0)
            self.lat = self.load_grid('YC.data', zrange=0)
        
            
    def pcolormesh(self, data, fname, proj=False, clim=None, **kwargs):
        """Output a tile that can be turned into a map"""
        if data.shape != (self.Ny,self.Nx):
            raise ValueError('Only 2D data of the correct shape can be pcolored')
        
        # load both corner and centers, necessary for pcolor
        lon_c = self.load_grid('XC.data', zrange=0)
        lat_c = self.load_grid('YC.data', zrange=0)
        lon_g = self.load_grid('XG.data', zrange=0)
        lat_g = self.load_grid('YG.data', zrange=0)
        
        # wrap if necessary
        wrap_flag =  (np.diff(lon_g,axis=1) < -180).any()
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

        # get the bounds only on the non-masked data
        # doing this on x screws up the automatic wrapping
        #x_min = max(x_c.min(), -xlim_true)
        #x_max = min(x_c.max(), xlim_true)
        pad = 5*dx
        x_min, x_max = x_c.min()-pad, x_c.max()+pad
        y_min = max(y_c.min()-pad, -ylim_true)
        y_max = min(y_c.max()+pad, ylim_true)
        
        dpi = 80.
        figsize = (x_max-x_min)/dx/dpi, (y_max-y_min)/dx/dpi
       
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = fig.add_axes([0,0,1,1])
        ax.set_xlim((x_min, x_max))
        ax.set_ylim((y_min, y_max))
        pc = ax.pcolormesh(x_quad, y_quad, data)
        ax.set_axis_off()
        if clim is not None:
            pc.set_clim(clim)
        
        fig.savefig(fname, dpi=dpi, figsize=figsize, transparent=True)
        plt.close(fig)
        
        # write world file
        wf = open('%sw' % fname, 'w')
        wf.write('%10.9f \n' % dx) # pixel X size
        wf.write('%10.9f \n' % 0.) # rotation about x axis
        wf.write('%10.9f \n' % 0.) # rotation about y axis
        wf.write('%10.9f \n' % -dx) # pixel Y size
        wf.write('%10.9f \n' % x_min) # X coordinate of upper left pixel center
        wf.write('%10.9f \n' % y_max) # Y coordinate of upper left pixel center
        wf.close()
        
        return ((x_min,y_min,x_max,y_max),figsize)
    
        
    # for resampling purposes
    def export_geotiff(self, data, basename='tile'):
        import pyresample
        
        self.load_latlon()
        # mask with the same mask as the input data
        if hasattr(data, 'mask'):
            lon = np.ma.masked_array(self.lon, data.mask).copy()
            lat = np.ma.masked_array(self.lat, data.mask)
        else:
            lon, lat = self.lon.copy(), self.lat
            
        # detect a jump
        if (np.diff(lon,axis=1)<-350).any():
            lon[lon<0] += 360.
        
        # the "Google" projection
        proj4_str = '+proj=merc +a=6378137 +b=6378137 +lat_ts=0.0 +lon_0=0.0 +x_0=0.0 +y_0=0 +k=1.0 +units=m'
        proj4_str_full = proj4_str + ' +nadgrids=@null +wktext +no_defs'
        
        # determine the extent
        #area_extent = np.hstack( [
        #    latlon_to_meters((self.lat[0,0], self.lon[0,0])),
        #    latlon_to_meters((self.lat[-1,-1], self.lon[-1,-1])) ] )
        area_extent = np.hstack( [
            latlon_to_meters((lat.min(), lon.min())),
            latlon_to_meters((lat.max(), lon.max())) ] )

        # set up pyresample to use the same resolution as the tile itself
        area_def = pyresample.utils.get_area_def('tile%5d' % self.id, 'Google Maps Global Mercator', 'GMGM',
                        proj4_str, self.Nx, self.Ny, area_extent)
        grid_def = pyresample.geometry.GridDefinition(lons=lon, lats=lat)

        # need to define the approximate grid size
        dx = abs(area_extent[2] - area_extent[0]) / (self.Nx)
        
        # the heavy lifting
        data_regrid = pyresample.kd_tree.resample_nearest(
                grid_def, data, area_def, dx, fill_value=None)

        # find out if we have to split the tile
        L = self.LLC.L
        x,y = area_def.get_proj_coords()
        if  x[0,-1] > L:
            i_split = (x[0,:] > L).nonzero()[0][0]
        else:
            i_split = self.Nx
            
        # write using GDAL
        
	output_fname = '%s_%04da.tiff' % (basename, self.id)
        # this is key
        # In a north up image, padfTransform[1] is the pixel width, and padfTransform[5] is the pixel height.
        # The upper left corner of the upper left pixel is at position (padfTransform[0],padfTransform[3]).
        # Xp = padfTransform[0] + P*padfTransform[1] + L*padfTransform[2];
        # Yp = padfTransform[3] + P*padfTransform[4] + L*padfTransform[5];
        geo_transform_a = (x[0,0], area_def.pixel_size_x, 0,
                         y[0,0], 0, -area_def.pixel_size_y)
        geo_transform_b = None

        output_gdal_geotiff(data_regrid[:,:i_split], output_fname, proj4_str_full, geo_transform_a)
        
        if i_split < self.Nx:
            output_fname_b = '%s_%04db.tiff' % (basename, self.id)
            geo_transform_b = (x[0,i_split]-(2*L), area_def.pixel_size_x, 0,
                             y[0,i_split], 0, -area_def.pixel_size_y)
            output_gdal_geotiff(data_regrid[:,i_split:].copy(), output_fname_b, proj4_str_full, geo_transform_b)
        
        return (geo_transform_a, geo_transform_b, data_regrid)
        
        
        
    
        
        
        
        
        
