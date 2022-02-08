import os
import torch as t
import numpy as np
import matplotlib
matplotlib.use('PS')
import matplotlib.pyplot as plt
from os.path import join as jn
import imageio
from pathlib import Path
from numpy import linalg as LA
import csv
import shutil

def newfold(dir1, machine):
    i = 0
    while True:
        n = str(i)
        dirname = jn(dir1,n.zfill(3) + machine)
        if not os.path.exists(dirname):
            Path(dirname).mkdir(parents=True, exist_ok=True)
            # os.mkdir(dirname)
            break
        else:
            i += 1
            continue
    return (dirname)

def simple_time_tracker(log_fun):
    def _simple_time_tracker(fn):
        @wraps(fn)
        def wrapped_fn(*args, **kwargs):
            start_time = time.time()

            try:
                result = fn(*args, **kwargs)
            finally:
                elapsed_time = time.time() - start_time

                # log the result
                log_fun({
                    'function_name': fn.__name__,
                    'total_time': elapsed_time,
                })

            return result

        return wrapped_fn
    return _simple_time_tracker
def _log(message):
    print('[SimpleTimeTracker] {function_name} {total_time:.3f}'.format(**message))

def showpoints(ax, el, az, pcd, vox2mm, lims=None):
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
              'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
    # fig = plt.figure(figsize=(10,10))
    # ax = fig.add_subplot(projection='3d')
    for i, p in enumerate(pcd):
        if p.shape[0] < p.shape[1]:
            p = p.T
        ax.scatter(p[:,0], p[:,1], p[:,2], c=colors[i], s=5)
    ax_lim = lims if lims is not None else [-60 * vox2mm, 60 * vox2mm]
    # print('ax_lim=',ax_lim)
    # ax.scatter(p0[0, :], p0[1, :], p0[2, :], marker='o', s=10, c="r", alpha=0.6)
    # if p1 is not None:
    #     ax.scatter(p1[0, :], p1[1, :], p1[2, :], marker='o', s=10, c="g", alpha=0.6)
    ax.view_init(elev=el, azim=az)
    # ax_lim = 60 * vox2mm
    ax.set_xlim(ax_lim[0], ax_lim[1])
    ax.set_ylim(ax_lim[0], ax_lim[1])
    ax.set_zlim(ax_lim[0], ax_lim[1])

def showmanypoints(cbs, nim, p, q, pathes, angles_list, phase, i_batch, cnt,
                   dirname, mo, vox2mm):
    fig = plt.figure(figsize=(15, 15))
    el = np.repeat(35, 3)
    # print('p shape=',p.shape)
    # az = np.arange(25,150,30)
    az = np.arange(25, 115, 30)
    if mo != 'color_channel':
        ui = 3 if cbs / nim >= 3 else cbs // nim
        coef = nim
    else:
        ui = min(p.shape[0], 3)
        coef = 1

    for j in range(ui):
        for i in range(el.shape[0]):
            ax = fig.add_subplot(3, el.shape[0], i + 1 + 3 * j,
                                 projection='3d')
            showpoints(ax, el[i], az[i],
                       [p[coef * j], q[coef * j]], vox2mm)
            if j == 0:
                ax.set_title(
                    'azimut = {} degrees'.format(az[i]))
            # if i == 0:
            #     # print(int(angles_list[nim*j]))
            #     ax.set_zlabel(pathes[nim*j]+'ang = '+\
            #      str(int(angles_list[nim*j])))
    plt.subplots_adjust(wspace=0, hspace=0.05)
    Path(dirname).mkdir(parents=True, exist_ok=True)
    plt.savefig(jn(dirname, \
                   'pc_' + phase + '_' + str(cnt).zfill(3) + \
                   '_' + str(i_batch).zfill(3) + '.png'))
    plt.close(fig)

    # @simple_time_tracker(_log)

def fn2p(y_n, f_n, dirs, nsp, vox2mm, iscuda):
    p = t.zeros( \
        y_n.shape[0], 3, nsp)
    p = p.cuda() if iscuda else p
    for i in range(y_n.shape[0]):
        far = t.matmul(y_n[i], f_n[i])
        p[i, 0, :] = far * t.cos(dirs[i, :, 0]) * \
                     t.sin(dirs[i, :, 1])
        p[i, 1, :] = far * t.sin(dirs[i, :, 0]) * \
                     t.sin(dirs[i, :, 1])
        p[i, 2, :] = far * t.cos(dirs[i, :, 1])
    # p *= vox2mm
    return p

def f2p(far, dirs, nsp, vox2mm):
    if isinstance(far, t.Tensor):
        p = t.zeros( \
            far.shape[0], 3, nsp)
        # print(far.shape, dirs.shape)
        p = p.cuda() if iscuda else p
        for i in range(far.shape[0]):
            # far = t.matmul(y_n[i], f_n[i])
            p[i, 0, :] = far[i] * t.cos(dirs[i, :, 0]) * \
                         t.sin(dirs[i, :, 1])
            p[i, 1, :] = far[i] * t.sin(dirs[i, :, 0]) * \
                         t.sin(dirs[i, :, 1])
            p[i, 2, :] = far[i] * t.cos(dirs[i, :, 1])
    elif isinstance(far, np.ndarray):
        p = np.zeros( \
            [far.shape[0], 3, nsp])
        dirs = dirs.detach().cpu().numpy() if \
            isinstance(dirs, t.Tensor) else dirs
        # dirs = dirs.transpose() if max(dirs.shape[0]>
        dirs = np.expand_dims(dirs, axis=0) if len(dirs.shape) == 2 else dirs
        # print(far.shape, dirs.shape)
        # p = p.cuda() if iscuda else p
        # print(gf())
        for i in range(far.shape[0]):
            # far = t.matmul(y_n[i], f_n[i])
            p[i, 0, :] = far[i] * np.cos(dirs[i, :, 0]) * \
                         np.sin(dirs[i, :, 1])
            p[i, 1, :] = far[i] * np.sin(dirs[i, :, 0]) * \
                         np.sin(dirs[i, :, 1])
            p[i, 2, :] = far[i] * np.cos(dirs[i, :, 1])
    # print(p.shape, getframeinfo(currentframe()).lineno)
    p *= vox2mm
    return p

    # @simple_time_tracker(_log)
def lossfig(dirname, lossar, ylabel, title, xlim, ylim, lb, xlabel):
    plt.rcParams["figure.figsize"] = (9, 5)
    fig = plt.figure()
    # axes = plt.gca()
    if lb == 'pc+f':
        labels_text = [['train 0', 'val 0'], ['train 1', 'val 1']]
        colist = ['C0', 'C1']
        stylist = ['-', '--']
        lossar = lossar.reshape((2, 2, lossar.shape[-1]))
    else:
        labels_text = ['train', 'val']
    if lb == 'pc+f':
        for i in range(2):
            for j in range(2):
                plt.plot(
                    np.arange(lossar.shape[-1]), lossar[i, j, :],
                    label=labels_text[i][j], linewidth=3,
                    color=colist[i], linestyle=stylist[j])
    else:
        for i in range(2):
            plt.plot(
                np.arange(lossar.shape[-1]), lossar[i, :],
                label=labels_text[i], linewidth=3)
    #     plt.show()
    plt.grid()
    plt.legend(
        bbox_to_anchor=(1.05, 1),
        loc='best', borderaxespad=0., fontsize=24)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    if any(ylim):
        plt.ylim(ylim)
    elif not any(ylim):
        plt.autoscale(enable=True, axis='y', tight=None)
    if any(xlim):
        plt.xlim(xlim)
    elif not any(xlim):
        plt.autoscale(enable=True, axis='x', tight=None)
    axes = plt.gca()
    plt.xlabel(xlabel, fontsize=24)
    plt.ylabel(ylabel, fontsize=24)
    plt.locator_params(axis="x", nbins=4)
    plt.locator_params(axis="y", nbins=4)
    plt.title(title, fontsize=24)
    # print( 'wefwef')
    for tp in ['.png', '.pdf']:
        plt.savefig(dirname + ylabel + tp, bbox_inches='tight')
    # try:
    #     print(get_ipython().__class__.__name__)
    #     plt.show(fig)
    # except:
    #     pass
    plt.close(fig)

def lossout(fnm, xlabel, lossar, epp, dirname, lb):
    lossfig(jn(dirname, fnm), lossar,
        'Loss', fnm.replace('_',' '), (0, epp), (0, 1), lb, xlabel)
    logloss = np.ma.log10(lossar)
    lossfig(jn(dirname, fnm), logloss.filled(0),
        'log10(Loss)', fnm.replace('_',' '), (0, epp), (0, 0), lb, xlabel)

def backproject(prmat, img, pcd, dirname, rotate_output):
    # img0 = np.asarray(io.imread('D:/cherepashkin1/phenoseed/598/1484717/1491988/rotation_000.tif'))
    # print('hello')
    pcd = pcd.transpose()
    # print(pcd.shape)
    # pcd0 = np.concatenate((pcd, np.expand_dims(np.repeat(1,
    #                                                      pcd.shape[0]), axis=1)), axis=1)
    # # print('pcd0 shape = ', pcd0.shape)
    # # print(pcd0.shape)
    # if rotate_output:
    #     prmat00 = prmat
    #     pcd1 = np.matmul(pcd0, prmat00)
    # else:
    #     pcd1 = pcd0
    # pcd2 = pcd1 / np.repeat(np.expand_dims(pcd1[:, 3], axis=1), 4, axis=1)
    # print(prmat)
    # print(pcd.shape)
    pcd2 = forwardproj(pcd,prmat.T)
    # vox2mm = 1
    # pcd3 = pcd2 - np.mean(pcd2, axis=0)
    pcd3 = pcd2.transpose()
    # print(pcd3.shape)
    # pcd3 = pcd3[:,::4]
    # print(pcd3[:2,:].shape)
    # fig = plt.figure(figsize=(15, 6))
    fig, ax = plt.subplots(1,1, figsize=(8,8))
    # ax = fig.add_subplot(1, 1, 1)
    # pcd3 *= 2.5
    # print(getstat(pcd3))
    # print(pcd3.shape)
    ax.scatter(pcd3[0, :], pcd3[1, :], s=10)
    sz = 1
    ax.set_xlim([-sz, sz])
    ax.set_ylim([-sz, sz])
    # ax.set_aspect('equal', adjustable='box')
    # _ = ax.axis('off')
    # ax = fig.add_subplot(1, 2, 2)
    # print('shape img = ', img.shape)
    print('img stat=', getstat(img))
    # ax.imshow(255-img, cmap='gray')
    # ax.imshow(np.max(img)-img, cmap='Greys')
    _ = ax.axis('off')
    plt.savefig(dirname + '_back_projection.png')
    try:
        print(get_ipython().__class__.__name__)
        plt.show(fig)
    except:
        pass
    plt.close(fig)

def getstat(img):
    if isinstance(img, t.Tensor):
        tp = (float(t.min(img)), float(t.max(img)), float(t.mean(img)), float(t.std(img)))
    elif isinstance(img, np.ndarray):
        tp = (np.min(img), np.max(img), np.mean(img), np.std(img))
    return tp

def shplot(coef, dirname, cnt):
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
    fig = plt.figure(figsize=plt.figaspect(1.))
    ax = fig.add_subplot(projection='3d')
    Yx, Yy, Yz = plot_Y(ax, coef, xyz, phi, theta)
    plt.savefig(jn(dirname,'surface' + str(cnt).zfill(3) + '.png'))
    plt.close(fig)
    cnt += 1
    return(cnt)

def plot_Y(ax, coef, xyz, phi, theta):
    """Plot the spherical harmonic of degree el and order m on Axes ax."""
    f = np.zeros([100, 100]).astype('complex128')
    for od in range(int(np.sqrt(len(coef)))):
        for m in range(-od, od+1):
            fb = coef[od*(od+1)+m] * sph_harm(abs(m), od, phi, theta)
            f += fb
    Yx, Yy, Yz = np.abs(f) * xyz
    cmap = plt.cm.ScalarMappable(cmap=plt.get_cmap('PRGn'))
    cmap.set_clim(-0.5, 0.5)

    ax.plot_surface(Yx, Yy, Yz,
                    facecolors=cmap.to_rgba(f.real),
                    rstride=2, cstride=2)
    # Draw a set of x, y, z axes for reference.
    ax_lim = 50
    ax.plot([-ax_lim, ax_lim], [0, 0], [0, 0], c='0.5', lw=1, zorder=10)
    ax.plot([0, 0], [-ax_lim, ax_lim], [0, 0], c='0.5', lw=1, zorder=10)
    ax.plot([0, 0], [0, 0], [-ax_lim, ax_lim], c='0.5', lw=1, zorder=10)
    # Set the Axes limits and title, turn off the Axes frame.
    ax_lim = 40
    ax.set_xlim(-ax_lim, ax_lim)
    ax.set_ylim(-ax_lim, ax_lim)
    ax.set_zlim(-ax_lim, ax_lim)
    ax.axis('on')
    return(Yx, Yy, Yz)

def getcen(el):
    centre = [linalg.det(el[:,[1,2,3]]), -linalg.det(el[:,[0,2,3]]), linalg.det(el[:,[0,1,3]])]/(-linalg.det(el[:,[0,1,2]]))
    return centre

def forwardproj(pcd, pr):
    pcd2 = np.concatenate((pcd, np.expand_dims(np.repeat(1, pcd.shape[0]), axis = 1)), axis=1)
    pcd2 = np.matmul(pcd2,pr.T)
    pcd2 = pcd2/np.repeat(np.expand_dims(pcd2[:,3], axis=1), 4, axis = 1)
    return(pcd2[:,:3])

def getcenf(el):
    centre = [linalg.det(el[:,[1,2,3]]), -linalg.det(el[:,[0,2,3]]), linalg.det(el[:,[0,1,3]]), -linalg.det(el[:,[0,1,2]])]
    return centre

def sf(fig, dirname):
    for i in ['.png', '.pdf']:
        fig.savefig('./'+dirname+'/'+str(int(os.listdir('./'+dirname+\
                                        '/')[-1].split('.')[0])+1).zfill(3)+i)

def prmatread(path):
    with open(path, 'r') as f:
        prmatext = f.readlines()
    pr2 = [prmatext[i].replace('[','').replace(']','').replace(';','')\
           for i in range(len(prmatext))]
    pr3 = ''
    for i in pr2:
        pr3+=i
    pr4 = np.genfromtxt(StringIO(pr3), delimiter=',')
    return pr4

def outliers(ar):
    return(list(np.where(np.abs(ar-np.mean(ar)) > 6*np.std(ar))[0]))

def getstat(img):
    if isinstance(img, t.Tensor):
        tp = (float(t.min(img)), float(t.max(img)), float(t.mean(img)), float(t.std(img)))
    elif isinstance(img, np.ndarray):
        tp = (np.min(img), np.max(img), np.mean(img), np.std(img))
    return tp

def viewpoints(pcd, lims = None):
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
              'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(projection='3d')
    for i, p in enumerate(pcd):
        if p.shape[0] < p.shape[1]:
            p = p.T
        ax.scatter(p[:,0], p[:,1], p[:,2], c=colors[i], s=5)
    if lims is not None:
        ax.set_xlim(lims[0], lims[1])
        ax.set_ylim(lims[0], lims[1])
        ax.set_zlim(lims[0], lims[1])
    return fig, ax

def makegif(dirname,phase):
    cip = []
    for root, directories, filenames in os.walk(dirname):
        for filename in filenames:
            if 'pc_' + phase in filename and not 'checkpoint' in filename:
                cip.append(jn(root, filename))
    cip.sort()
    images = []
    for filename in cip:
        images.append(imageio.imread(filename))
    imageio.mimsave(jn(dirname, phase+'_movie.gif'), images, duration=0.5)

def namelist(path, fltr):
    cip = []
    for root, structure, files in os.walk(path):
        for file in files:
            if fltr in file:
                cip.append(os.path.join(root, file))
    return cip

def pose6tomat(a):
    ar = np.array([[a[0], a[1], a[2]],
            [a[1], a[3], a[4]],
            [a[2], a[4], a[5]]])
    return(ar)

def moments(pcd):
    if pcd.shape[0] < pcd.shape[1]:
        pcd = pcd.T
    X = pcd[:, 0]
    Y = pcd[:, 1]
    Z = pcd[:, 2]
    car = np.array([np.matmul(X, X.T), np.matmul(X, Y.T), np.matmul(X, Z.T),
            np.matmul(Y, Y.T), np.matmul(Y, Z.T), np.matmul(Z, Z.T)])
    return(car)

def rot2eul(R):
    if np.abs(R[2,0]) > 1:
        R[2,0] = round(R[2,0])
    beta = -np.arcsin(R[2,0])
    alpha = np.arctan2(R[2,1]/np.cos(beta),R[2,2]/np.cos(beta))
    gamma = np.arctan2(R[1,0]/np.cos(beta),R[0,0]/np.cos(beta))
    return np.array((alpha, beta, gamma))

def eul2rot(theta):
    R = np.array([[np.cos(theta[1])*np.cos(theta[2]),       np.sin(theta[0])*np.sin(theta[1])*np.cos(theta[2]) - np.sin(theta[2])*np.cos(theta[0]),      np.sin(theta[1])*np.cos(theta[0])*np.cos(theta[2]) + np.sin(theta[0])*np.sin(theta[2])],
                  [np.sin(theta[2])*np.cos(theta[1]),       np.sin(theta[0])*np.sin(theta[1])*np.sin(theta[2]) + np.cos(theta[0])*np.cos(theta[2]),      np.sin(theta[1])*np.sin(theta[2])*np.cos(theta[0]) - np.sin(theta[0])*np.cos(theta[2])],
                  [-np.sin(theta[1]),                        np.sin(theta[0])*np.cos(theta[1]),                                                           np.cos(theta[0])*np.cos(theta[1])]])

    return R

def pcd2eul(pcd):
    ans = rot2eul(LA.eig(pose6tomat(moments(pcd)))[1])
    return ans

def namelist(path, fltr):
    cip = []
    for root, structure, files in os.walk(path):
        for file in files:
            if fltr in file:
                cip.append(os.path.join(root, file))
    return cip

def rotplot(C):
    print('print me')
    try:
        plt.close(fig)
    except:
        pass
    fig1, ax = plt.subplots(3,3, figsize=(8,8))
    for i in range(3):
        for j in range(3):
            ax[i][j].plot(C[:,i,j])
    ax[0,0].set_xlabel('angle, degrees/10')
    ax[0,0].set_ylabel('element of the matrix')
    fig1.suptitle('Elements of the rotation matrix from projection matricies')
    return fig1

def getOri(pcd0, C, eigsort=True):
    print('439')
    mator1 = np.zeros([36,3,3])
    rotangles = np.zeros([36,3])
    for j in range(36):
        pcd = np.matmul(pcd, C[j,:,:].T)
        pcd = pcd-np.mean(pcd,axis=0)
        w, v = LA.eig(pose6tomat(moments(pcd)))
        if eigsort:
            mator0 = v[:,np.argsort(w)]
            print(448, LA.det(mator0))
            mator1[j,:,:] = LA.det(mator0)*mator0
        else:
            mator1[j,:,:] = v
        rotangles[j,:] = rot2eul(mator1[j,:,:])
    return mator1, rotangles

def testfun():
    print('test')
    # sf(fig, 'figs')
# def h5read(path, dname):
#     return np.array(h5py.File(path,'r').get('dataset'))

def plotori(mator, tit, xtit, ylim = (-1, 1), figtype='scatter'):
    fig, ax = plt.subplots(3,3, figsize=(7,7))
    # print(mator[0].shape)
    for i in range(3):
        for j in range(3):
            for m in mator:
                if figtype == 'scatter':
                    ax[i,j].scatter(np.arange(m.shape[0]), m[:,i,j], s=3)
                elif figtype == 'line':
                    ax[i,j].plot(m[:,i,j])
            ax[i,j].set_ylim(ylim)
            if i in (0,1):
                ax[i,j].set_xticks([])
            if j in (1,2):
                ax[i,j].set_yticks([])
    ax[2,0].set_xlabel(xtit)
    ax[0,0].set_ylabel('element of the matrix')
    fig.suptitle(tit)
#     plt.show(fig)
    return fig

def savelist(listf, filename):
    with open(filename, "w") as f:
        for element in listf:
            f.write(element + "\n")

class Bunch(object):
    def __init__(self, adict):
        self.__dict__.update(adict)

def dic2csv(path, dct):
    with open(path, 'w', newline='\n') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        for key, value in dct.items():
            writer.writerow([key, value])

def csv2dic(path):
    with open(path, 'r', newline='\n') as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        return Bunch(dict(reader))

def saferm(dirname):
    # recursively delte dirname content
    for filename in os.listdir(dirname):
            file_path = jn(dirname, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))