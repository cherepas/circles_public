import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
from scipy.special import sph_harm
import re
import pathlib

def rewalk(mainpath, lastfilename):
    cip = []
    for root, directories, filenames in os.walk(mainpath): 
        for filename in filenames:
            if filename[-len(lastfilename):] == lastfilename:
                cip.append(os.path.join(root,filename))
    return(cip)
def gety(coef):
    plt.rc('text', usetex=True)

    # Grids of polar and azimuthal angles
    theta = np.linspace(0, np.pi, 100)
    phi = np.linspace(0, 2*np.pi, 100)
    # Create a 2-D meshgrid of (theta, phi) angles.
    theta, phi = np.meshgrid(theta, phi)
    # Calculate the Cartesian coordinates of each point in the mesh.
    xyz = np.array([np.sin(theta) * np.sin(phi),
                    np.sin(theta) * np.cos(phi),
                    np.cos(theta)])
    f = np.zeros([100,100]).astype('complex128')
    for l in range(int(np.sqrt(len(coef)))):
        for m in range(-l,l+1):
            fb = coef[l*(l+1)+m] * sph_harm(abs(m), l, phi, theta)
            f += fb

    Yx, Yy, Yz = np.abs(f) * xyz
    return(Yx, Yy, Yz, f)
def ploty(coef,ax, ax_lim = 40):
    Yx, Yy, Yz, f = gety(coef)
    cmap = plt.cm.ScalarMappable(cmap=plt.get_cmap('PRGn'))
    cmap.set_clim(-0.5, 0.5)


    #ax = fig.add_subplot(projection='3d')

    ax.plot_surface(Yx, Yy, Yz,
#                     cmap=cm.coolwarm,
                   facecolors=cmap.to_rgba(f.real),
                    rstride=2, cstride=2, axes = ax)
    # Draw a set of x, y, z axes for reference.
    ax.plot([-ax_lim, ax_lim], [0,0], [0,0], c='0.5', lw=1, zorder=10)
    ax.plot([0,0], [-ax_lim, ax_lim], [0,0], c='0.5', lw=1, zorder=10)
    ax.plot([0,0], [0,0], [-ax_lim, ax_lim], c='0.5', lw=1, zorder=10)
    ax.set_xlim(-ax_lim, ax_lim)
    ax.set_ylim(-ax_lim, ax_lim)
    ax.set_zlim(-ax_lim, ax_lim)
    ax.axis('on')
    # axes.set_xticks([])
    # axes.set_yticks([])
    #plt.axis('off')
    plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
def shplot(cip, nrows, ncols, labtext):
#    cip = rewalk(mainpath, 'o')
    plt.rcParams["figure.figsize"] = (25,10)
#    nrows=1
#    ncols=4
    #figure, axes = plt.subplots(nrows, ncols)
    #axes = axes.ravel()
    #for i, ax in enumerate(axes):
    #print(np.unravel_index(i, (nrows,ncols)))
    #     ax = axes[i]
    #file = open(cip[i],'r') 
    fig = plt.figure(figsize=plt.figaspect(1.))
    gs1 = gridspec.GridSpec(nrows, ncols)
    gs1.update(wspace=0.15, hspace=0.5) # set the spacing between axes. 
    #coef = np.genfromtxt(cip[0], delimiter = ',')
    #print(coef.shape)
    for i in range(nrows*ncols):
    #     ax = plt.subplot(gs1[i]) 
    #     ax = fig.add_subplot(2,5,i+1,projection='3d', frameon = False)
    #    ax = plt.subplot(2,5,i+1,projection='3d',gs1[i])
        if i >= len(cip):
            break
        coef = np.genfromtxt(cip[i], delimiter = ',')
        ax = plt.subplot(gs1[i], projection = '3d')
        plt.subplots_adjust(top = 0.99, bottom=0.01, hspace=0, wspace=0)
    #fig = plt.figure(figsize=plt.figaspect(1.))
        ploty(coef[0],ax,ax_lim = 40)
        ax.title.set_text(labtext[i])
#         fig.suptitle('3d shape based on spherical harmonics', fontsize = 18)
    #fig.title('wfewef')
#shplot(mainpath)
#     figure.tight_layout()
    #plt.savefig((r'C:\Users\v.cherepashkin\Documents\GitHub\circles\plot output\input_data'+str(time.time())+'.png').replace('\\', '/'))