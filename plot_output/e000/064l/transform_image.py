import numpy as np
from skimage import transform

isprint = False

class CmsCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image = sample['image']
        # print('img stat before cmscrop =', getstat(image[0]))

        h, w = image[0].shape[0:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)
        # h, w = image[0].shape[0:2]
        new_image = np.zeros([enim, new_h, new_w]).astype(np.single)
        # print(image.shape)
        for i in range(enim):
            # img = np.squeeze(image[i])
            # img *= 255
            img = np.squeeze(255-image[i])
            properties = regionprops(
                (img > filters.threshold_otsu(img)).astype(int), img)
            cms = tuple(map(lambda x: int(x), properties[0].centroid))
            tempa = (img[cms[0] - new_h//2: cms[0] + new_h//2,
                     cms[1] - new_w//2: cms[1] + new_w//2]).astype(np.uint8)
            padh = (new_h-tempa.shape[0])//2
            padw = (new_w-tempa.shape[1])//2
            tempb = np.pad(
                tempa, ((padh, new_h-tempa.shape[0]-padh),
                        (padw, new_w-tempa.shape[1]-padw)),
                mode='constant', constant_values=0)
            new_image[i] = tempb
        # image = new_image/255
        image = new_image
        # print('img stat after cmscrop =', getstat(image[0]))
        if isprint:
            print('CmsCrop passed')
        return {'image': image}
class CentralCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image = sample['image']
        # print('img stat before cmscrop =', getstat(image[0]))
        if self.output_size:
            h, w = image[0].shape[0:2]
            if isinstance(self.output_size, int):
                if h > w:
                    new_h, new_w = self.output_size * h / w, self.output_size
                else:
                    new_h, new_w = self.output_size, self.output_size * w / h
            else:
                new_h, new_w = self.output_size

            new_h, new_w = int(new_h), int(new_w)
            # TODO modify on the case, if it is not evenly divisible on 2
            # print(w, h, new_w, new_h, image.shape)
            image = 255-image[:, (h - new_h) // 2:(h + new_h) // 2,
                        (w - new_w) // 2:(w + new_w) // 2]
        # print('image shape', image.shape)
        if isprint:
            print('CmsCrop passed')
        return {'image': image}
class Rescale(object):
    """Rescale the image in a sample to a given size.
    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        # print('output_size',output_size)
        self.output_size = output_size
        # self.opt = opt

    def __call__(self, sample):
        image = sample['image']
        # print('img stat before rescale=', getstat(image[0]))
        if self.output_size:
            # if opt.inputt == 'pc':
            #     img = image
            # else:
            h, w = image[0].shape[0:2]
            print(111, image.shape)
            # print('output_size=',self.output_size)
            # print('h,w',h,w)
            if h != self.output_size:
                if isinstance(self.output_size, int):
                    if h > w:
                        new_h, new_w = \
                            self.output_size * h / w, self.output_size
                    else:
                        new_h, new_w = \
                            self.output_size, self.output_size * w / h
                else:
                    new_h, new_w = self.output_size

                new_h, new_w = int(new_h), int(new_w)
                # print('new_h,new_w',new_h, new_w)
                img = np.zeros([image.shape[0], new_h, new_w]).astype(np.single)
                # print(image.shape,getstat(image[0]))
                for i in range(image.shape[0]):
                    img[i] = np.squeeze(
                        transform.resize(image[i, :, :],(new_h, new_w),
                            preserve_range=True))
                # transforms.ToTensor()
            elif h == self.output_size:
                img = image
            image = img
        # print('img stat after rescale=', getstat(image[0]))
        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        if isprint:
            print('Rescale passed')
        return {'image': image}
class Standardize(object):
    """standardize images on zero mean and unit variance"""
    def __init__(self, stand):
        self.stand = stand
        # self.meanv = meanv
        # self.stdv = stdv
        pass
    def __call__(self, sample):
        image = sample['image']
        # image = 255-image
        # print('img stat before stndardize =', getstat(image[0]))
        if isinstance(self.stand, int) and self.stand == 255:
            image = image/255
        elif isinstance(self.stand, tuple):
            image = (image-self.stand[0]) / self.stand[1]
        # print('img stat after stndardize =', getstat(image[0]))
        if isprint:
            print('Standardize passed')
        return {'image': image}
# class AmpCrop(object):
#     """Crop the label, spherical harmonics amplitude."""
#
#     def __init__(self, ampl):
#         self.ampl = ampl
#
#     def __call__(self, sample):
#         image = sample['image']
#
#         if self.ampl == 441:
#             f_n = f_n
#         else:
#             f_n = f_n[:self.ampl]
#         if isprint:
#             print('ampcrop passed')
#         return {'image': image, 'angles': angles,
#
#
#
#                 'path': path}
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self, iscuda):
        # assert isinstance(device, str)
        # self.device = device
        self.iscuda = iscuda
    def __call__(self, sample):
        image = sample['image']

        # print(self.iscuda)
        # f_n = np.squeeze(f_n)
        image = torch.Tensor(image).cuda() if self.iscuda else\
            torch.Tensor(image)
        # f_n = torch.Tensor(f_n).cuda() if self.iscuda else torch.Tensor(f_n)

        if isprint:
            print('ToTensor passed')
        return {'image': image}

class Minmax3Dimage(object):
    """Normalize 3D input data to be laying in [0,1]"""
    def __init__(self, minv, maxv):
        self.minv = minv
        self.maxv = maxv

    def __call__(self, sample):
        image = sample['image']

        image = (image-self.minv) / (self.maxv-self.minv)
        if isprint:
            print('Minmax3dimage passed')
        return {'image': image}


# class Minmax_f(object):
#     """Normalize 3D input data to be laying in [0,1]"""
#
#     def __init__(self, minmax):
#         minf = minmax[0]
#         maxf = minmax[1]
#         self.minf = minf
#         self.maxf = maxf
#
#     def __call__(self, sample):
#         image = sample['image']
#
#         far = (far-self.minf)/(self.maxf-self.minf)
#         if isprint:
#             print('Minmax_f passed')
#         return {'image': image, 'angles': angles,
#
#
#
#                 'path': path}


class Downsample(object):
    """Downsample the input ply file."""

    def __init__(self, ds):
        self.ds = ds

    def __call__(self, sample):
        image = sample['image']

        image = image[::self.ds, :]
        if isprint:
            print('Downsample passed')
        return {'image': image}


# class Shuffleinput(object):
#     """Shuffle the rows of input ply file."""
#
#     def __init__(self, shuffle_seed):
#         self.shuffle_seed = shuffle_seed
#
#     def __call__(self, sample):
#         np.random.seed(self.shuffle_seed)
#         image = sample['image']
#
#         np.random.shuffle(image)
#         if isprint:
#             print('Shuffleinpute passed')
#         return {'image': image, 'angles': angles,
#
#
#
#                 'path': path}


# class Minmax(object):
#     """Normalize the input data to lay in [0,1]."""
#
#     def __init__(self, tmean):
#         self.tmean = tmean
#
#     def __call__(self, sample):
#         image = sample['image']
#
#         # f_n = ((f_n - np.min(self.tmean[2])) /
#         #        (np.max(self.tmean[3])-np.min(self.tmean[2])))
#         if isprint:
#             print('Minmax passed')
#         return {'image': image, 'angles': angles,
#
#
#
#                 'path': path}


class Reshape(object):
    """Normalize the input data to lay in [0,1]."""

    def __init__(self, input_layer):
        self.input_layer = input_layer

    def __call__(self, sample):
        image = sample['image']

        padval = self.input_layer**2 - image.shape[0]
        if padval >= 0:
            image = np.pad(image, ((0, padval), (0, 0)), mode='constant')
        else:
            image = image[:self.input_layer**2]
        image = np.reshape(image, [3, self.input_layer, self.input_layer])
        if isprint:
            print('Reshape passed')
        return {'image': image}


class Normalize(object):
    """Normalize the input data to lay in [0,1]."""

    def __init__(self):
        pass

    def __call__(self, sample):
        image = sample['image']

        X = image[:, 0]
        Y = image[:, 1]
        Z = image[:, 2]
        C = np.zeros([3, 3])
        C[0, 0] = np.matmul(X, X.transpose())
        C[0, 1] = np.matmul(X, Y.transpose())
        C[0, 2] = np.matmul(X, Z.transpose())
        C[1, 0] = C[0, 1]
        C[1, 1] = np.matmul(Y, Y.transpose())
        C[1, 2] = np.matmul(Y, Z.transpose())
        C[2, 0] = C[0, 2]
        C[2, 1] = C[1, 2]
        C[2, 2] = np.matmul(Z, Z.transpose())
        w, v = LA.eig(C)
        image = np.matmul(v.transpose(), image.transpose()).transpose()
        if isprint:
            print('Normalize passed')
        return {'image': image}


class Center(object):
    """Make the center of masses of point cloud to be in the origin."""

    def __init__(self):
        pass

    def __call__(self, sample):
        image = sample['image']

        image = image - np.mean(image, axis=0)
        if isprint:
            print('Center passed')
        return {'image': image}








# class Divide255(object):
#     """Normalize the input data to lay in [0,1]."""
#
#     def __init__(self):
#         pass
#
#     def __call__(self, sample):
#         image = sample['image']
#
#         if isprint:
#             print('Divide255 passed')
#         image = image/255
#         return {'image': image, 'angles': angles,
#
#
#
#                 'path': path}
