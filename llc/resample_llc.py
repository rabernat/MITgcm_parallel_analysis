import numpy as np
import llc_worker
import pyresample
import os
import time

base_dir = '/Volumes/Bucket1/llc/llc_1080'
LLC = llc_worker.LLCModel1080(
        data_dir = os.path.join(base_dir, 'run_day732_896'),
        grid_dir = os.path.join(base_dir, 'grid'))

# earth radius
a = 6378169.0

#msg_area = pyresample.geometry.AreaDefinition('msg_full',
#    'Full globe MSG image 0 degrees','msg_full',
#    {'a': '6378169.0', 'b': '6356584.0','h': '35785831.0', 'lon_0': '0','proj': 'geos'},
#    LLC.Ntop*4, LLC.Ntop*4,[-5568742.4, -5568742.4,5568742.4, 5568742.4])

# Proj4 for google map
area_id = 'WGS84'
area_name = 'Google Maps Global Mercator'
proj_id = 'WGS84_gm'
#proj4_args = '+proj=merc +a=6378137 +b=6378137 +lat_ts=0.0 +lon_0=0.0 +x_0=0.0 +y_0=0 +k=1.0 +units=m +nadgrids=@null +wktext  +no_defs'
proj4_args = '+proj=merc +a=6378137 +b=6378137 +lat_ts=0.0 +lon_0=0.0 +x_0=0.0 +y_0=0 +k=1.0 +units=m'
x_size = LLC.Ntop*4
y_size = LLC.Ntop*4
#area_extent = (-5326849.0625,-5326849.0625,5326849.0625,5326849.0625)
area_extent = [-20037508.342789244, -20037508.342789244, 20037508.342789244, 20037508.342789244]
area_def = pyresample.utils.get_area_def(area_id, area_name, proj_id, proj4_args,
                                       x_size, y_size, area_extent)

#msg_area = pyresample.geometry.AreaDefinition('msg_full',
#    'Full globe MSG image 0 degrees','msg_full',
#    {'proj': 'merc', 'a': 6378137.0, 'b': 6378137.0,
#     'lat_ts': 0.0, 'lon_0': 0.0, 'x_0': 0.0, 'y_0': 0.0, 'k':1.0, 'units':'m'},
#    LLC.Ntop*4, LLC.Ntop*4,[-5568742.4, -5568742.4,5568742.4, 5568742.4])

result = np.ma.masked_array(np.zeros(area_def.shape, dtype=LLC.dtype), True)

for n in range(LLC.Nfaces):
    lon = LLC.load_grid_file('XC.data',n)[0]
    lat = LLC.load_grid_file('YC.data',n)[0]
    mask = (lon==0.)
    depth = np.ma.masked_array(
             LLC.load_grid_file('Depth.data',n)[0], mask)
    
    # need to define the approximate grid size
    dx = 2*np.pi*a / (LLC.Ntop*4)
    
    # the pyresample stuff
    # grid definition
    grid_def = pyresample.geometry.GridDefinition(lons=lon, lats=lat)
    
    # two methods of resampling
    # 1 - KDtree
    print n
    
    tprev = time.time()
    res = pyresample.kd_tree.resample_nearest(
            grid_def, depth, area_def, dx, fill_value=None)
    # only fill in where masked
    replace_mask = (result.mask & ~res.mask)
    result[replace_mask] = res[replace_mask]
    print time.time() - tprev

    # 2 - using grid stuff (doesn't seem to work)
    # container with depth data 
    #tprev = time.time()
    #grid_con = pyresample.image.ImageContainerNearest(
    #            depth, grid_def, radius_of_influence=dx)
    #area_con = grid_con.resample(area_def)
    #resultB += area_con.image_data
    #print time.time() - tprev