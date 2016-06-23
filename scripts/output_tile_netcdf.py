import os
import sys
import numpy as np
#from scipy.io import netcdf_file
from netCDF4 import Dataset
sys.path.append('../')
from llc import llc_model
import sys

# defaults
tid = 144
iter0 = 10368
# get tile id and start iteration number from command line arguments
if len(sys.argv)>1:
    tid = int(sys.argv[1])
if len(sys.argv)>2:
    iter0 = int(sys.argv[2])

deltaT = 25.
diter = 144
iterN = 487152
#iterstep = diter*24 # one day steps
iterstep = diter

# I get a memory error if I true to include the grid files
do_grid = False

output_dir = '/nobackup/rpaberna/LLC/tile_data'

base_dir_4320 = os.path.join(os.environ['LLC'], 'llc_4320')
LLC4320 = llc_model.LLCModel4320(
        data_dir = os.path.join(base_dir_4320, 'MITgcm/run'),
        grid_dir = os.path.join(base_dir_4320, 'grid'),
        use_memmap=False)

#tid = 144

for tile in LLC4320.get_tile_factory():
    if tile.id==tid:
        break

# set up netcdf output
fname = '%s/LLC4320_%04d_%010d.nc' % (output_dir, tile.id, iter0) 
#f = netcdf_file(fname, 'w')
f = Dataset(fname, 'w')
f.history = 'Generated for Xiao'

f.createDimension('Nt')
f.createDimension('Nx', tile.Nx)
f.createDimension('Ny', tile.Ny)
f.createDimension('Nzc', tile.Nz)
f.createDimension('Nzf', tile.Nz+1)

# coordinates
niter = f.createVariable('niter', 'i', ('Nt',))
niter.units = 'iteration number'
ntime = f.createVariable('time', 'f', ('Nt',))
ntime.units = 'time (s)'

# grid data
if do_grid:
    tile.load_geometry()

    xc = f.createVariable('xc', 'f', ('Ny','Nx'))
    xc[:] = tile.x['C'].squeeze()
    xc.units = 'longitude at cell center'
    xg = f.createVariable('xg', 'f', ('Ny','Nx'))
    xg[:] = tile.x['G'].squeeze()
    xg.units = 'longitude at cell western boundary'
    yc = f.createVariable('yc', 'f', ('Ny','Nx'))
    yc[:] = tile.y['C'].squeeze()
    yc.units = 'latitude at cell center'
    yg = f.createVariable('yg', 'f', ('Ny','Nx'))
    yg[:] = tile.y['G'].squeeze()
    yg.units = 'latitude at cell southern boundary'
    rc = f.createVariable('rc', 'f', ('Nzc',))
    rc[:] = tile.r['C'].squeeze()
    rc.units = 'depth at cell center'
    rf = f.createVariable('rf', 'f', ('Nzf',))
    rf[:] = tile.r['F'].squeeze()
    rf.units = 'depth at cell edges'

    # areas
    rac = f.createVariable('rac', 'f', ('Ny','Nx'))
    rac[:] = tile.ra['C'].squeeze()
    rac.units = 'area of c-point (m^2)'
    ras = f.createVariable('ras', 'f', ('Ny','Nx'))
    ras[:] = tile.ra['S'].squeeze()
    ras.units = 'area of s-point (m^2)'
    raw = f.createVariable('raw', 'f', ('Ny','Nx'))
    raw[:] = tile.ra['W'].squeeze()
    raw.units = 'area of w-point (m^2)'
    raz = f.createVariable('raz', 'f', ('Ny','Nx'))
    raz[:] = tile.ra['Z'].squeeze()
    raz.units = 'area of z-point (m^2)'

    # derivatives
    dxc = f.createVariable('dxc', 'f', ('Ny','Nx'))
    dxc[:] = tile.dx['C'].squeeze()
    dxc.units = 'm'
    dxg = f.createVariable('dxg', 'f', ('Ny','Nx'))
    dxg[:] = tile.dx['G'].squeeze()
    dxg.units = 'm'
    dyc = f.createVariable('dyc', 'f', ('Ny','Nx'))
    dyc[:] = tile.dy['C'].squeeze()
    dyc.units = 'm'
    dyg = f.createVariable('dyg', 'f', ('Ny','Nx'))
    dyg[:] = tile.dy['G'].squeeze()
    dyg.units = 'm'
    drc = f.createVariable('drc', 'f', ('Nzf',))
    drc[:] = tile.dr['C'].squeeze()
    drc.units = 'm'
    drf = f.createVariable('drf', 'f', ('Nzc',))
    drf[:] = tile.dr['F'].squeeze()
    drf.units = 'm' 
    # masks
    hfacc = f.createVariable('hfacc', 'f', ('Nzc','Ny','Nx'))
    hfacc[:] = tile.hfac['C']
    hfacc.units = 'dimensionless (0 to 1)'
    hfacw = f.createVariable('hfacw', 'f', ('Nzc','Ny','Nx'))
    hfacw[:] = tile.hfac['W']
    hfacw.units = 'dimensionless (0 to 1)'
    hfacs = f.createVariable('hfacs', 'f', ('Nzc','Ny','Nx'))
    hfacs[:] = tile.hfac['S']
    hfacs.units = 'dimensionless (0 to 1)'
    
    f.sync()

# actual dATA
uvar = f.createVariable('U', 'f', ('Nt','Nzc','Ny','Nx'))
uvar.units = 'm/s'
vvar = f.createVariable('V', 'f', ('Nt','Nzc','Ny','Nx'))
vvar.units = 'm/s'
wvar = f.createVariable('W', 'f', ('Nt','Nzc','Ny','Nx'))
wvar.units = 'm/s'
tvar = f.createVariable('T', 'f', ('Nt','Nzc','Ny','Nx'))
tvar.units = 'degrees C'
svar = f.createVariable('S', 'f', ('Nt','Nzc','Ny','Nx'))
svar.units = 'PSU'
# surface and bottom data
pvar = f.createVariable('Phibot', 'f', ('Nt','Ny','Nx'))
pvar.units = 'm^2 / s^2'
evar = f.createVariable('Eta', 'f', ('Nt','Ny','Nx'))
evar.units = 'm'


# load velocity files
for nt, iter in enumerate(xrange(iter0, iterN, iterstep)):
    niter[nt] = iter
    ntime[nt] = deltaT*iter
    T = tile.load_data('Theta.%010d.data' % iter )
    S = tile.load_data('Salt.%010d.data' % iter )
    U = tile.load_data('U.%010d.data' % iter)
    V = tile.load_data('V.%010d.data' % iter)
    W = tile.load_data('W.%010d.data' % iter)
    Phibot = tile.load_data('PhiBot.%010d.data' % iter)
    Eta = tile.load_data('Eta.%010d.data' % iter)

    uvar[nt] = U
    vvar[nt] = V
    wvar[nt] = W
    tvar[nt] = T
    svar[nt] = S
    pvar[nt] = Phibot.squeeze()
    evar[nt] = Eta.squeeze()
    f.sync()

f.close()




