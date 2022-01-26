isprint = False
# class AmpCrop(object):
#     """Crop the label, spherical harmonics amplitude."""
#
#     def __init__(self, ampl):
#         self.ampl = ampl
#
#     def __call__(self, sample):
#         image, path = sample['image'], sample['path']
#
#         if self.ampl == 441:
#             f_n = f_n
#         else:
#             f_n = f_n[:self.ampl]
#         if isprint:
#             print('ampcrop passed')
#         return {'image': image,
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
        image, path = sample['image'], sample['path']

        # print(self.iscuda)
        # f_n = np.squeeze(f_n)
        image = torch.Tensor(image).cuda() if self.iscuda else\
            torch.Tensor(image)
        # f_n = torch.Tensor(f_n).cuda() if self.iscuda else torch.Tensor(f_n)

        if isprint:
            print('ToTensor passed')
        return {'image': image,


                'path': path}


class Minmax3Dimage(object):
    """Normalize 3D input data to be laying in [0,1]"""
    def __init__(self, minv, maxv):
        self.minv = minv
        self.maxv = maxv

    def __call__(self, sample):
        image, path = sample['image'], sample['path']

        image = (image-self.minv) / (self.maxv-self.minv)
        if isprint:
            print('Minmax3dimage passed')
        return {'image': image,



                'path': path}


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
#         image, path = sample['image'], sample['path']
#
#         far = (far-self.minf)/(self.maxf-self.minf)
#         if isprint:
#             print('Minmax_f passed')
#         return {'image': image,
#
#
#
#                 'path': path}


class Downsample(object):
    """Downsample the input ply file."""

    def __init__(self, ds):
        self.ds = ds

    def __call__(self, sample):
        image, path = sample['image'], sample['path']

        ds_image = image[::self.ds, :]
        if isprint:
            print('Downsample passed')
        return {'image': ds_image,



                'path': path}


# class Shuffleinput(object):
#     """Shuffle the rows of input ply file."""
#
#     def __init__(self, shuffle_seed):
#         self.shuffle_seed = shuffle_seed
#
#     def __call__(self, sample):
#         np.random.seed(self.shuffle_seed)
#         image, path = sample['image'], sample['path']
#
#         np.random.shuffle(image)
#         if isprint:
#             print('Shuffleinpute passed')
#         return {'image': image,
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
#         image, path = sample['image'], sample['path']
#
#         # f_n = ((f_n - np.min(self.tmean[2])) /
#         #        (np.max(self.tmean[3])-np.min(self.tmean[2])))
#         if isprint:
#             print('Minmax passed')
#         return {'image': image,
#
#
#
#                 'path': path}


class Reshape(object):
    """Normalize the input data to lay in [0,1]."""

    def __init__(self, input_layer):
        self.input_layer = input_layer

    def __call__(self, sample):
        image, path = sample['image'], sample['path']

        padval = self.input_layer**2 - image.shape[0]
        if padval >= 0:
            image = np.pad(image, ((0, padval), (0, 0)), mode='constant')
        else:
            image = image[:self.input_layer**2]
        image = np.reshape(image, [3, self.input_layer, self.input_layer])
        if isprint:
            print('Reshape passed')
        return {'image': image,



                'path': path}


class Normalize(object):
    """Normalize the input data to lay in [0,1]."""

    def __init__(self):
        pass

    def __call__(self, sample):
        image, path = sample['image'], sample['path']

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
        return {'image': image,



                'path': path}


class Center(object):
    """Make the center of masses of point cloud to be in the origin."""

    def __init__(self):
        pass

    def __call__(self, sample):
        image, path = sample['image'], sample['path']

        image = image - np.mean(image, axis=0)
        if isprint:
            print('Center passed')
        return {'image': image,



                'path': path}


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
        image, path = sample['image'], sample['path']

        h, w = image[0].shape[0:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)
        h, w = image[0].shape[0:2]
        new_image = np.zeros([enim, new_h, new_w])
        # print(image.shape)
        for i in range(enim):
            img = np.squeeze(image[i])
            img *= 255
#            img = np.squeeze(255-image[i])
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
        if isprint:
            print('CmsCrop passed')
        return {'image': new_image/255,



                'path': path}


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

    def __call__(self, sample):
        image, path = sample['image'], sample['path']

        if opt.inputt == 'pc':
            img = image
        else:
            h, w = image[0].shape[0:2]
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
                img = np.zeros([enim, new_h, new_w])
                for i in range(enim):
                    img[i] = np.squeeze(transform.resize(image[i, :, :],
                                                         (new_h, new_w)))
                transforms.ToTensor()
            elif h == self.output_size:
                img = image
        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        if isprint:
            print('Rescale passed')
        return {'image': img,


                'path': path}


class Divide255(object):
    """Normalize the input data to lay in [0,1]."""

    def __init__(self):
        pass

    def __call__(self, sample):
        image, path = sample['image'], sample['path']

        if isprint:
            print('Divide255 passed')
        return {'image': image/255,



                'path': path}
