import numpy as np
import torch as t
import random as rnd
#import circle_generator as g
from circle_generator import *
def genbatch(k, imsize, N, n):
    # np.random.seed(0)
    # t.random.manual_seed(0)
    data = t.zeros([N,1,imsize,imsize], dtype=t.float32)
    # if figure_type == 'circle':
    #     k = 3
    # elif figure_type == 'ellipse':
    #     k = 4
    #labels = t.rand(N, 3)
    labels = np.random.random((N,k*n))
    for i in range(N):
        data[i,:,:,:] = ellipses(labels[i,:], k, imsize)
        #image from circle generator is saved. first argument is batch iterable,
        #second argument is color iterable, last two are image dimensions
    labels = t.from_numpy(labels)
    #dividing data into train and test
    testsize = int(N*0.2)
    valsize = int((N-testsize)*0.2)
    #testset = [data[:testsize,:,:,:].float(), labels[:testsize,:].float()]
    testset = [data[:testsize,:,:,:], labels[:testsize,:]]
    datar = data[testsize:,:,:,:]
    labelsr = labels[testsize:,:]
    # dataset = [datar[valsize:,:,:,:].float(), datar[:valsize,:,:,:].float(),\
    #             labelsr[valsize,:].float(), labelsr[:valsize,:].float()]
    dataset = [datar[valsize:,:,:,:].float(), datar[:valsize,:,:,:].float(),\
                labelsr[valsize:,:].float(), labelsr[:valsize,:].float()]
    #cross-validation splitting
    # dataset_i = []
    # dataset = []
    #batcha = range(0, N, valsize)
    #for i, n in enumerate(batcha[:-1]):
    #    dataset_i = [datar[n:batcha[i+1],:,:,:].float(), \
    #                (t.cat((datar[:n,:,:,:], datar[batcha[i+1]:,:,:]), 0)).float(), \
    #                labelsr[n:batcha[i+1],:].float(), \
    #                (t.cat((labelsr[:n,:], labelsr[batcha[i+1]:]), 0)).float(),]
    #    dataset.append(dataset_i)
    return(dataset, testset)
