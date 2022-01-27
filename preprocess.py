import numpy as np
#import os
#from os.path import join as jn
from helpers import *
from inspect import currentframe, getframeinfo
import pandas as pd
def preprodf(homepath, opt, mt1):
    excpath = jn(homepath, opt.specie + '_exceptions.txt')
    csvPathSep = None
    #TODO it is obsolete to create a file from separate F_N
    if not os.path.isfile(excpath) and not opt.single_folder:
        cip = []
        for root, directories, filenames in os.walk(csvPathSep):
            for filename in filenames:
                if 'F_N.csv' in filename:
                    cip.append(jn(root, filename))
        F_Nm = np.zeros([len(cip), opt.ampl])
        for i in range(len(cip)):
            F_Nm[i, :] = np.genfromtxt(cip[i], delimiter=',')
        for i in range(opt.ampl):
            ar = F_Nm[:, i]
            a = np.concatenate((a, np.where(np.abs(ar - np.mean(ar)) > 6 * np.std(ar))[0]))
        exception_list = [cip[index][-23:-8] for index in np.unique(a.astype(int))]
        savelist(exception_list, excpath)
    elif os.path.isfile(excpath) and not opt.single_folder:
        with open(excpath, "r") as f:
            exception_list = f.readlines()
    else:
        exception_list = []
    print(len(exception_list))
    with open(excpath.replace('exceptions.txt', 'exceptions_good.txt'), "r") as f:
        good = f.readlines()
    for g in good:
        exception_list = [x for x in exception_list if g not in x]
    pts = jn(homepath, 'csv', opt.specie + 'frame.csv')
    print('file to frame csv', pts)
    if mt1:
        frameinfo = getframeinfo(currentframe())
        print(frameinfo.filename, frameinfo.lineno)
    if opt.noise_output:
        if mt1:
            frameinfo = getframeinfo(currentframe())
            print(frameinfo.filename, frameinfo.lineno)
        lframe = pd.DataFrame()
        lframe.insert(
            0, 'file_name', np.zeros(5270).astype(str))
    elif not opt.noise_output and os.path.isfile(pts) and \
            opt.use_existing_csv:
        if mt1:
            frameinfo = getframeinfo(currentframe())
            print(frameinfo.filename, frameinfo.lineno)
        # TODO use only one file for all F_Nw, prmat and zero angle
        lframe = pd.read_csv(pts)
        lframewh = lframe.copy()
        print("lframe's length after laoding = ", len(lframe))
    elif not opt.noise_output and not (os.path.isfile(pts) and \
                                       opt.use_existing_csv):
        if mt1:
            frameinfo = getframeinfo(currentframe())
            print(frameinfo.filename, frameinfo.lineno)
        cip = []
        for word in opt.specie.split(','):
            csvpath = jn(csvPathSep, word + 'csv')
            for root, directories, filenames in os.walk(csvpath):
                for filename in filenames:
                    search_str = 'F_N_' + str(nsp) + '.csv'
                    for i in range(len(exception_list)):
                        if not any(exception in cip[i] for exception in exception_list):
                            #                 if search_str in filename and flag:
                            tfolds = jn(root, filename).split('/')[-3:]
                            cip.append(jn(tfolds[0], tfolds[1], tfolds[2]))
        lframe = pd.DataFrame()
        lframe.insert(
            0, 'file_name', [cip[i].split('_F_N')[0] for i in range(len(cip))])

        list_zero = []
        if mt1:
            frameinfo = getframeinfo(currentframe())
            print(frameinfo.filename, frameinfo.lineno)

        for idx in range(len(lframe)):
            s = np.zeros(36)
            rotpath = jn(
                dataPath,
                lframe.iloc[idx, 0].replace('csv', ''))
            for j in range(36):
                img = io.imread(
                    jn(
                        rotpath, 'rotation_' + str(10 * j).zfill(3) + '.tif'))
                s[j] = np.sum(255 - img)
            list_zero.append(np.argmax(s))
        lframe.insert(1, 'zero_angle', list_zero)
        lframe = lframe.sample(frac=1)
        lframe.to_csv(pts, index=False)
    inn = opt.inputt == 'img' and not opt.noise_input
    nim = opt.num_input_images
    if inn and not opt.rand_angle:
        alf = np.array([10 * int(36 * i / nim) for i in range(nim)])
    elif inn and opt.rand_angle:
        alf = np.random.choice(
            np.arange(0, 360, 10), size=nim, replace=False)
    elif inn and nim == 1 and opt.zero_angle:
        alf = [0]
    pts = jn(homepath, 'csv', opt.specie + '_view_sep_' + str(nim) + '.csv')
    if opt.view_sep and os.path.isfile(pts) and \
            opt.use_sep_csv:
        lframe_sep = pd.read_csv(pts)
    else:
        lframe_sep = lframe
    isusable = os.path.isfile(pts) and \
               opt.use_sep_csv and int(len(lframe_sep) / 5283) == nim
    if opt.view_sep and not (isusable) and not opt.zero_angle:
        lframe_sep = pd.DataFrame(columns=('file_name', 'angle'))
        for idx in range(len(lframe)):
            for j, angle in enumerate(alf):
                if lframe.iloc[idx, 1] + int(angle / 10) >= 36:
                    angle = lframe.iloc[idx, 1] + int(angle / 10) - 36
                else:
                    angle = lframe.iloc[idx, 1] + int(angle / 10)
                lframe_sep.loc[idx * len(alf) + j] = [lframe.iloc[idx, 0], angle]
        lframe = lframe_sep
        lframe.to_csv(pts, index=False)
    elif opt.view_sep and isusable:
        if mt1:
            frameinfo = getframeinfo(currentframe())
            print(frameinfo.filename, frameinfo.lineno)
        lframe = pd.read_csv(pts)

    for st in exception_list:
        lframe = lframe[~lframe.file_name.str.contains(st.replace('\n', ''))]
    print('lframe len after excluding all exceptions=', len(lframe))
    if opt.single_folder:
        lframe = lframe[:100]
    lframe = lframe.sort_values(by=['file_name'])
    lframe = lframe.sample(frac=1, random_state=0)
    lframe = lframe[:opt.framelim]
    lframe = lframe.replace({opt.specie + 'csv': opt.csvname}, regex=True)

    return lframe, lframewh, alf, inn