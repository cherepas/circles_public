from os.path import join as jn
import numpy as np
from torchvision import transforms
from transform_image import *
def transformdef(opt, homepath):
    # exec(open(jn(homepath, "transform"+opt.transappendix+".py")).read())

    # if not opt.noise_output and opt.minmax:
    #     tmean = np.genfromtxt(
    #         jn(homepath,'csv','tmean.csv'),
    #         delimiter=',')
    # else:
    #     tmean = np.zeros([5270, opt.ampl])
    # minmax, minmax3dimage, normalize, center, cmscrop,\
    # cencrop, downsample = ['']*7
    # if opt.minmax:
    #     minmax = 'Minmax(tmean[:,:opt.ampl]), '
    # else:
    #     minmax = ''
    # if opt.inputt == 'pc' and opt.minmax3dimage:
    #     minmax3dimage = \
    #         'Minmax3Dimage(\
    #             np.array([29,  1,  4]), np.array([240, 138, 243])), '
    # else:
    #     minmax3dimage = ''
    # if opt.inputt == 'pc' and opt.normalize:
    #     normalize = 'Normalize(), '
    # else:
    #     normalize = ''
    # if opt.inputt == 'pc' and opt.center:
    #     center = 'Center(), '
    # else:
    #     center = ''
    # if not opt.inputt == 'pc' and opt.cmscrop:
    #     cmscrop = 'CmsCrop(opt.cmscrop),'
    # else:
    #     cmscrop = ''
    # if not opt.inputt == 'pc' and opt.cencrop:
    #     cencrop = 'CentralCrop(opt.cencrop),'
    # else:
    #     cencrop = ''
    # if opt.inputt == 'pc' and opt.downsample:
    #     downsample = 'Downsample(opt.downsample), '
    # else:
    #     downsample = ''
    # if opt.inputt == 'f':
    #     minmax3dimage, normalize, center, cmscrop, downsample, rescale = ['']*6
    # if opt.inputt == 'f' and opt.minmax_f:
    #     minmax_f = 'Minmax_f((2.6897299652941, 102.121738007844)), '
    # else:
    #     minmax_f = ''
    # if opt.inputt == 'img' and opt.rescale:
    #     rescale = 'Rescale(opt.rescale), '
    # else:
    #     rescale = ''
    # if opt.inputt == 'img':
    #     standardize = 'Standardize(opt.standardize), '
    # else:
    #     standardize = ''
    # if opt.noise_input or opt.single_folder:
    #     (minmax, minmax3dimage, normalize, center,
    #      cmscrop, cencrop, downsample, minmax_f) = ['']*8
    #     rescale = 'Rescale(opt.rescale), '
    # es0 = "transforms.Compose([" + \
    #       cencrop+cmscrop+rescale+standardize+minmax3dimage+normalize+center+ \
    #       downsample+"])"
    # TODO rewrite without exec
    # print(68)
    # exec("a = 1")
    # print(a)
    # exec("data_transforms = {\'train\': "+es0+",'val': "+es0+"}")
    data_transforms = {'train': transforms.Compose([CentralCrop(opt.cencrop),
                                                    Rescale(opt.rescale),
                                                    Standardize(opt.standardize)]),
                       'val': transforms.Compose([CentralCrop(opt.cencrop),
                                                    Rescale(opt.rescale),
                                                    Standardize(opt.standardize)]),
                       }
    # print(data_transforms)
    return data_transforms