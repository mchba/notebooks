import numpy as np
import xarray
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.pyplot import cm
from scipy.interpolate import griddata


# PLOT STYLE ##################################
mpl.style.use('classic')
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams["legend.scatterpoints"] = 1
plt.rcParams["legend.numpoints"] = 1
plt.rcParams['grid.linestyle'] = ':'
mpl.rcParams['lines.linewidth'] = 2
plt.rcParams['axes.grid']=True
yd = dict(rotation=0, ha='right')
plt.close('all')



# HELPER FUNCTIONS ##########################
def calc_stresses(data, rho=1.225):
    # Store budget terms in this dictionary
    data_vars = {}

    # 1st derivatives
    diffx = data.differentiate(coord='x')
    diffy = data.differentiate(coord='y')
    diffz = data.differentiate(coord='z')

    # Calculate stresses
    data_vars['uv'] = -data['muT'] / rho * (diffy['U'] + diffx['V'])
    data_vars['uw'] = -data['muT'] / rho * (diffz['U'] + diffx['W'])
    data_vars['vw'] = -data['muT'] / rho * (diffz['V'] + diffy['W'])
    data_vars['uu'] = 2. / 3. * data['tke'] - data['muT'] / rho * 2 * diffx['U']
    data_vars['vv'] = 2. / 3. * data['tke'] - data['muT'] / rho * 2 * diffy['V']
    data_vars['ww'] = 2. / 3. * data['tke'] - data['muT'] / rho * 2 * diffz['W']

    # Assemble DataArrays to one DataSet
    stress_terms = xarray.Dataset(data_vars=data_vars)
    return stress_terms


# PARAMETERS ########################################
D = 126.0
zh = 90.0
UH = 8
x_ex = [-5*D, 1*D, 5*D]


## DATA
# Neutral k-eps-fP, U=8, TI=12, NREL5MW
filename = 'flowdata_invariants.nc'
data = xarray.open_dataset(filename)
x_off = 2.5*D  # Trasnlate everything 2.5D to the right to have AD at x=0
data = data.assign_coords(x=(data.x + x_off)) # Translate coordinates
stresses = calc_stresses(data)



# PLOT XY-plane #############################
#fig, ax = plt.subplots(1, 2, sharex='col', figsize=(12, 6),
#                       gridspec_kw={'width_ratios': (30, 1)})
if False:
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(8, 8))
    plt.subplots_adjust(hspace=-0.2)
    xyplane = data.interp(z=zh)
    X, Y = np.meshgrid(xyplane.x, xyplane.y)
    p = ax[0].contourf(X / D, Y / D, xyplane['U'].T / UH, np.linspace(0.4, 1.05, 10), cmap=cm.jet)
    # Plot where budgets are extracted:
    ax[0].plot([np.min(data['x']), np.max(data['x'])], [0, 0], 'k--', label='visualize stresses on this line')
    #for i in range(len(x_ex)):
    #    ax[0].plot([x_ex[i] / D], [0], 'ko', markersize=8, label='extract stress tensor for Jupyter notebook' if i == 1 else '')
    ax[0].legend(fontsize=10)
    ax[0].axis('scaled')
    ax[0].set_xlim(left=-6, right=10)
    ax[0].set_ylim(bottom=-2, top=2)
    ax[1].set_xlabel('$x/D$')
    ax[0].set_ylabel('$y/D$', yd)
    ax[0].set_title('At $z = z_{hub}$')
    ax[0].grid()
    axpos = np.array(ax[0].get_position())  # [x0, y0; x1, y1], 0=lower left, 1=upper right
    cbar = fig.add_axes([axpos[1,0]+0.02, axpos[0,1], 0.03, axpos[1,1]-axpos[0,1]]) # [minx, miny, dx, dy]
    comcolRANS = plt.colorbar(p, cax=cbar, orientation='vertical', aspect=100)
    comcolRANS.set_label('$U/U_H$', rotation=0, ha='left')
    comcolRANS.set_ticks(np.arange(0.3, 1.05, 0.1))
    
    
    # Plot stress lines
    reynolds = ['uu', 'vv', 'ww', 'uv', 'uw', 'vw']
    reynolds_str = ["$\overline{u'u'}$","$\overline{v'v'}$","$\overline{w'w'}$","$\overline{u'v'}$","$\overline{u'w'}$","$\overline{v'w'}$"]
    stress_line = stresses.interp(y=0, z=zh)
    for i in range(len(reynolds)):
        ax[1].plot(stress_line['x']/D,stress_line[reynolds[i]],label=reynolds_str[i])
    
    
    # ylabel and legend
    ax[1].set_ylabel(r"$\overline{u_i' u_j'}$ [-]",yd);
    ax[1].legend(loc='center left',fontsize=12,bbox_to_anchor=(1.0, 0.5),
              ncol=1, fancybox=True, shadow=True,scatterpoints=1, handlelength=2,title='Stresses:')
    
    
    
    
    

## CALC EIGENVALUES ##########################
def calc_eigenvalues(stress):
    '''
    Input: 1D, 2D or 3D netCDF file of stresses
    Output: Corresponding netCDF file with sorted eigenvalues
    '''
    
    # Store lam1, lam2 and lam3 in this dictionary
    data_vars = {}

    # Try: If 3D data is supplied
    # Except: If 1D or 2D data is supplied, we need to reshape to 3D netcdf data
    try:
        Nx = len(stress.x)
    except:
        Nx = 1
        stress = stress.expand_dims('x',axis=0)
    try:
        Ny = len(stress.y)
    except:
        Ny = 1
        stress = stress.expand_dims('y',axis=1)
    try:
        Nz = len(stress.z)
    except:
        Nz = 1
        stress = stress.expand_dims('z',axis=2)
        
    
    uiuj = np.zeros((Nx, Ny, Nz, 3, 3))
    uiuj[:,:,:,0,0] = stress['uu'][:,:,:]
    uiuj[:,:,:,1,1] = stress['vv'][:,:,:]
    uiuj[:,:,:,2,2] = stress['ww'][:,:,:]
    uiuj[:,:,:,1,0] = stress['uv'][:,:,:]
    uiuj[:,:,:,0,1] = stress['uv'][:,:,:]
    uiuj[:,:,:,2,0] = stress['uw'][:,:,:]
    uiuj[:,:,:,0,2] = stress['uw'][:,:,:]
    uiuj[:,:,:,1,2] = stress['vw'][:,:,:]
    uiuj[:,:,:,2,1] = stress['vw'][:,:,:]
    
    lam = np.zeros((Nx, Ny, Nz, 3))
    
    lam1 = 0*stress['uu']  # 3D DataArray full of 0's
    lam2 = 0*stress['uu']  # 3D DataArray full of 0's
    lam3 = 0*stress['uu']  # 3D DataArray full of 0's
    
    I = np.diag(np.ones(3))
    
    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                uiuj_test = uiuj[i,j,k]
                tke = 0.5*np.trace(uiuj_test)
                bij = uiuj_test/(2*tke) - 1/3*I
                w, V = np.linalg.eig(bij)
                #lam = np.sort(w)[::-1]   # [::-1] needed to have increasing order
                #lam1[i,j,k], lam2[i,j,k], lam3[i,j,k] = np.sort(w)[::-1]   # [::-1] needed to have increasing order
                lam[i,j,k] = np.sort(w)[::-1]   # [::-1] needed to have increasing order
                #if(np.sum(lam) > 1e-6):
                #    print('Something wrong... trace should be 0!')
    

    # Calculate stresses
    lam1.values = lam[:,:,:,0]
    lam2.values = lam[:,:,:,1]
    lam3.values = lam[:,:,:,2]
    data_vars['lam1'] = lam1
    data_vars['lam2'] = lam2
    data_vars['lam3'] = lam3
    
    # Calculate barycentric coefficients
    data_vars['C1'] = data_vars['lam1'] - data_vars['lam2']
    data_vars['C2'] = 2*data_vars['lam2'] - 2*data_vars['lam3']
    data_vars['C3'] = 3*data_vars['lam3'] + data_vars['lam2']/data_vars['lam2']

    # Assemble DataArrays to one DataSet
    eigenvalues = xarray.Dataset(data_vars=data_vars)
    return eigenvalues

#stresses_test = stresses.where((stresses.z>50) & (stresses.z<54), drop=True) 
stresses_test = stresses.interp(z=zh)

ev = calc_eigenvalues(stresses_test)


### Scatter plot #########
fig = plt.figure()
X, Y = np.meshgrid(ev.x, ev.y)
Xf = X.flatten()
Yf = Y.flatten()
C1f = ev['C1'].T.values.flatten()
C2f = ev['C2'].T.values.flatten()
C3f = ev['C3'].T.values.flatten()
Cf = np.column_stack((C1f, C2f, C3f))
def plot_bar2():
    plt.scatter(Xf, Yf, color=Cf, marker='s', s=5)
plot_bar2()
plt.axis('scaled')
plt.savefig('bary_map.pdf',bbox_inches='tight')


### Imshopw plot ###############
# Interpolate data to structured grid
dx = 8
xs = np.arange(-800,801,dx)
ys = np.arange(-200,201,dx)
extent = [xs[0]-dx, xs[-1]+dx, ys[0]-dx, ys[-1]+dx] 
ev_g = ev.squeeze(drop='True').interp(x=xs,y=ys)
rgb_g = np.dstack((ev_g['C1'].values.T, ev_g['C2'].values.T, ev_g['C3'].values.T))

fig = plt.figure()
plt.plot([0,0],[-40,40],'k',linewidth=1)
plt.imshow(rgb_g, origin='lower', extent=extent, interpolation='None')  # "origin=lower" = origo in lower left corner
plt.savefig('bary_map_im.pdf',bbox_inches='tight')

