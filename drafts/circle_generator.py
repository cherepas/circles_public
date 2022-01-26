#import matplotlib.pyplot as plt
#from math import sqrt
import torch as t
import numpy as np
def figure_init(imsize, device):
    #I = t.zeros([imsize,imsize], dtype=t.float32, requires_grad=True)
    #r = t.zeros([imsize,imsize], dtype=t.float32, requires_grad=True)
    X = t.arange(0, imsize, 1, dtype=t.float32, requires_grad=True)
    Y = t.arange(0, imsize, 1, dtype=t.float32, requires_grad=True)
    X, Y = t.meshgrid(X, Y)
    X = X.to(device)
    Y = Y.to(device)
    return(X, Y)
def smoothborder(r, x):
    if len(x) > 4:
        r0 = np.sqrt(x[2]**2+x[3]**2)
    else:
        r0 = x[2]
    I = (r <= r0)
    a = 1
    c = 4
    I2 = (1+a*np.exp(c))/(1+a*t.exp(c*r/r0))
    I = I + (r > r0) * I2
    return(I)
def ellipses(x, k, imsize, device):
    X, Y = figure_init(imsize)
    I = t.zeros([imsize,imsize], dtype=t.float32, requires_grad=True)
    R = t.zeros([imsize,imsize], dtype=t.float32, requires_grad=True)
    I = I.to(device)
    R = R.to(device)
    if k == 3:
        for i in range(int(len(x)/k)):
            r = t.sqrt(((X-x[k*i]*imsize)**2 + (Y-x[k*i+1]*imsize)**2))/imsize
            r = r.to(device)
            I = I + smoothborder(r, x[k*i:k*(i+1)])
    if k == 4:
        for i in range(int(len(x)/k)):
            r = t.sqrt((((X-x[k*i]*imsize)/x[k*i+2])**2 + \
            ((Y-x[k*i+1]*imsize)/x[k*i+3])**2))/imsize
            I = I + smoothborder(r, x[k*i:k*(i+1)])
    #R = (I<1)
    R = (I>=1) + (I<1)*I
    return(R)
# def ellipses0(x, k, imsize):
#     X, Y = figure_init(imsize)
#     I = t.zeros([imsize,imsize], dtype=t.float32, requires_grad=True)
#     if k == 3:
#         for i in range(int(len(x)/k)):
#             r = t.sqrt(((X-x[k*i]*imsize)**2 + (Y-x[k*i+1]*imsize)**2))/imsize
#             for xx in range(imsize):
#                 for yy in range(imsize):
#                     if I[xx, yy] == 0:
#                         I[xx, yy] = smoothborder(r, x[k*i:k*(i+1)])
#
#             I = I + smoothborder(r, x[k*i:k*(i+1)])
#     if k == 4:
#         for i in range(int(len(x)/k)):
#             r = t.sqrt((((X-x[k*i]*imsize)/x[k*i+2])**2 + \
#             ((Y-x[k*i+1]*imsize)/x[k*i+3])**2))/imsize
#             I = I + smoothborder(r, x[k*i:k*(i+1)])
#     return(I)

# def smoothbordere(r, a, b):
#     r0 = np.sqrt(a**2+b**2)
#     I = (r <= r0)
#     a = 1
#     c = 5
#     I2 = (1+a*np.exp(c))/(1+a*t.exp(c*r/r0))
#     I = I + (r > r0) * I2
#     return(I)
# def circle(x, imsize):
#     X, Y = figure_init(imsize)
#     r = t.sqrt(((X-x[0]*imsize)**2 + (Y-x[1]*imsize)**2))/imsize
#     I = smoothborder(r, x)
#     return(I)
# def ellipse(x, figure_type, imsize):
#     X, Y = figure_init(imsize)
#     if figure_type == 'circle':
#         r = t.sqrt(((X-x[0]*imsize)**2 + (Y-x[1]*imsize)**2))/imsize
#     if figure_type == 'ellipse':
#     I = smoothborder(r, x)
#     return(I)
# def one_ellipse_gen(x, imsize):
#     X, Y = figure_init(imsize)
#     r = t.sqrt((((X-x[0]*imsize)/x[2])**2 + ((Y-x[1]*imsize)/x[3])**2))/imsize
#     I = smoothborder(r, x)
#     #I = smoothborder(r, x[2])
#     return(I)
# def circles_gen(x, imsize):
#     X, Y = figure_init(imsize)
#     I = t.zeros([imsize,imsize], dtype=t.float32, requires_grad=True)
#     k = 3
#     for i in range(int(len(x)/k)):
#         r = t.sqrt(((X-x[k*i]*imsize)**2 + (Y-x[k*i+1]*imsize)**2))/imsize
#         I = I + smoothborder(r, x[k*i+2])
#     return(I)
# def ellipses_gen(x, imsize):
#     X, Y = figure_init(imsize)
#     I = t.zeros([imsize,imsize], dtype=t.float32, requires_grad=True)
#     k = 4
#     for i in range(int(len(x)/k)):
#         r = t.sqrt((((X-x[k*i]*imsize)/x[k*i+2])**2 + \
#         ((Y-x[k*i+1]*imsize)/x[k*i+3])**2))/imsize
#         I = I + smoothborder(r, x[k*i+2])
#     return(I)


def center_circle_gen(x, imsize):
    X, Y = figure_init(imsize)
    r = t.sqrt(((X-0.5*imsize)**2 + (Y-0.5*imsize)**2))
    return(smoothborder(r, ))

def ellipses_gen(xv, yv, av, bv, imsize):
    X, Y = figure_init(imsize)
    a = 1
    I = t.zeros([imsize,imsize], dtype=t.float32, requires_grad=True)
    for i in range(len(xv)):
        r = t.sqrt((((X-xv[i]*imsize)/av[i]/imsize)**2 + \
        ((Y-yv[i]*imsize)/bv[i]/imsize)**2))
        I = I + 1/(1+a*t.exp(r))
    return(I)
def one_gauss_circle_gen(x, imsize):
    X, Y = figure_init(imsize)
    rkv = (X-x[0]*imsize)**2 + (Y-x[1]*imsize)**2
    r = t.sqrt(rkv)
    a = 0.1
    #I = t.exp(-rkv*(a/x[2]/imsize)**2)
    I = 1 / (1 + a*t.exp(r/x[2]/imsize))
    return(I)
def coli(I,col):
    im = t.stack([I*col[i] for i in np.arange(3)])
    #dtype=t.float32, requires_grad=True
    return(im)
def one_col_circle_gen(x0, y0, r0, col):
    I = coli(one_circle_gen(x0, y0, r0),col)
    return I
def one_col_ellipse_gen(x0, y0, a0, b0, col):
    I = coli(one_ellipse_gen(x0, y0, a0, b0),col)
    return I
def col_circles_gen(xv, yv, rv, colv, imsize):
    X, Y = figure_init(imsize)
    a = 1
    im = t.zeros([3, imsize,imsize], dtype=t.float32, requires_grad=True)
    for i in range(len(xv)):
        r = t.sqrt(((X-xv[i]*imsize)**2 + (Y-yv[i]*imsize)**2))
        I = 1/(1+a*t.exp(r/(float(rv[i])*imsize)))
        im = im + coli(I,colv[3*i:3*i+3])
    return im
def col_ellipses_gen(xv, yv, av, bv, colv, imsize):
    X, Y = figure_init(imsize)
    a = 1
    im = t.zeros([3, imsize,imsize], dtype=t.float32, requires_grad=True)
    for i in range(len(xv)):
        r = t.sqrt((((X-xv[i]*imsize)/av[i]/imsize)**2 + \
        ((Y-yv[i]*imsize)/bv[i]/imsize)**2))
        I = 1/(1+a*t.exp(r))
        im = im + coli(I,colv[3*i:3*i+3])
    return im
