import os
import sys
import numpy as np
from scipy.io import netcdf_file
sys.path.append('../')
from llc import llc_model


base_dir_4320 = os.path.join(os.environ['LLC'], 'llc_4320')
LLC4320 = llc_model.LLCModel4320(
        data_dir = os.path.join(base_dir_4320, 'MITgcm/run'),
        grid_dir = os.path.join(base_dir_4320, 'grid'))

tid = 144

for tile in LLC4320.get_tile_factory():
    if tile.id==tid:
        break

iter = 221760

# load grid data
tile.load_geometry()

# load velocity files
T = tile.load_data('Theta.%010d.data' % iter )
S = tile.load_data('Salt.%010d.data' % iter )
U = tile.load_data('U.%010d.data' % iter)
V = tile.load_data('V.%010d.data' % iter)
W = tile.load_data('W.%010d.data' % iter)
Phibot = tile.load_data('PhiBot.%010d.data' % iter)
Eta = tile.load_data('Eta.%010d.data' % iter)

fname = '../tile_data/LLC4320_%04d_%010d.nc' % (tile.id, iter) 
f = netcdf_file(fname, 'w')
f.history = 'Generated for Xiao'

f.createDimension('Nx', tile.Nx)
f.createDimension('Ny', tile.Ny)
f.createDimension('Nzc', tile.Nz)
f.createDimension('Nzf', tile.Nz+1)

# coordinates
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
dyg = f.createVariable('yg', 'f', ('Ny','Nx'))
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

# actual dATA
uvar = f.createVariable('U', 'f', ('Nzc','Ny','Nx'))
uvar[:] = U
uvar.units = 'm/s'
vvar = f.createVariable('V', 'f', ('Nzc','Ny','Nx'))
vvar[:] = V
vvar.units = 'm/s'
wvar = f.createVariable('W', 'f', ('Nzc','Ny','Nx'))
wvar[:] = W
wvar.units = 'm/s'
tvar = f.createVariable('T', 'f', ('Nzc','Ny','Nx'))
tvar[:] = T
tvar.units = 'degrees C'
svar = f.createVariable('S', 'f', ('Nzc','Ny','Nx'))
svar[:] = S
svar.units = 'PSU'
# surface and bottom data
pvar = f.createVariable('Phibot', 'f', ('Ny','Nx'))
pvar[:] = Phibot.squeeze()
pvar.units = 'm^2 / s^2'
evar = f.createVariable('Eta', 'f', ('Ny','Nx'))
evar[:] = Eta.squeeze()
evar.units = 'm'


f.close()




