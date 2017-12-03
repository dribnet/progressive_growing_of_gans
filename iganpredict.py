from __future__ import print_function

import theano
import theano.tensor as T
from time import time
from lib import HOGNet
from lib.rng import np_rng
from lib.theano_utils import floatX, sharedX
import numpy as np

from lib import AlexNet
import lasagne
from scipy import optimize
import argparse
from PIL import Image
from pydoc import locate
from lib import activations
from scipy.special import ndtri, ndtr

# MERGECODE
from theano import tensor as T
import config
import misc
import numpy as np
import scipy.ndimage
# _, _, G = misc.load_pkl("../progressive_growing_of_gans/network-snapshot-009041.pkl")

def def_feature(layer='conv4', up_scale=4):
    print('COMPILING...')
    t = time()
    x = T.tensor4()
    x_t = AlexNet.transform_im(x)
    x_net = AlexNet.build_model(x_t, layer=layer, shape=(None, 3, 64, 64), up_scale=up_scale)
    AlexNet.load_model(x_net, layer=layer)
    x_f = lasagne.layers.get_output(x_net[layer], deterministic=True)
    _ftr = theano.function(inputs=[x], outputs=x_f)
    print('%.2f seconds to compile _feature function' % (time() - t))
    return _ftr

old_way = False
printing_op = theano.printing.Print('vector', attrs = [ 'shape' ])
z_scale = 1.2

def def_bfgs(net, layer='conv4', npx=64, alpha=0.002):
    print('COMPILING...')
    t = time()

    x_f = T.tensor4()
    x = T.tensor4()
    z = T.matrix()

    z = theano.printing.Print('this is z')(z)
    tanh = activations.Tanh()
    tz = tanh(z)
    # tz = printing_op(tz)

    # tz = z_scale * tz
    net.labels_var  = T.TensorType('float32', [False] * 512) ('labels_var')
    gx = net.G.eval(z, net.labels_var, ignore_unused_inputs=True)
    # gx = printing_op(gx)
    # gx = misc.adjust_dynamic_range(gx, [-1,1], [0,1])
    scale_factor = 16
    gx = theano.tensor.signal.pool.pool_2d(gx, ds=(scale_factor, scale_factor), mode='average_exc_pad', ignore_border=True)
    # gx = printing_op(gx)

    if layer is 'hog':
        gx_f = HOGNet.get_hog(gx, use_bin=True, BS=4)
    else:
        gx_t = AlexNet.transform_im(gx)
        gx_net = AlexNet.build_model(gx_t, layer=layer, shape=(None, 3, npx, npx))
        AlexNet.load_model(gx_net, layer=layer)
        gx_f = lasagne.layers.get_output(gx_net[layer], deterministic=True)

    f_rec = T.mean(T.sqr(x_f - gx_f), axis=(1, 2, 3)) * sharedX(alpha)
    x_rec = T.mean(T.sqr(x - gx), axis=(1, 2, 3))
    cost = T.sum(f_rec) + T.sum(x_rec)
    grad = T.grad(cost, z)
    output = [cost, grad, gx]
    _invert = theano.function(inputs=[z, x, x_f], outputs=output)

    print('%.2f seconds to compile _bfgs function' % (time() - t))
    return _invert,z

def def_predict(model_P):
    print('COMPILING...')
    t = time()
    x = T.tensor4()
    z = model_P(x)
    _predict = theano.function([x], [z])
    print('%.2f seconds to compile _predict function' % (time() - t))
    return _predict


def def_invert_models(net, layer='conv4', alpha=0.00001):
    bfgs_model = def_bfgs(net, layer=layer, npx=64, alpha=alpha)
    ftr_model = def_feature(layer=layer)
    return bfgs_model, ftr_model

def invert_bfgs_batch(invert_model, ftr_model, ims, z_predict=None, npx=64):
    zs = []
    recs = []
    fs = []
    t = time()
    n_imgs = ims.shape[0]
    # print('reconstruct %d images using bfgs' % n_imgs)

    for n in range(n_imgs):
        im_n = ims[[n], :, :,:]
        if z_predict is not None:
            z0_n = z_predict[[n],...]
        else:
            z0_n = None
        gx, z_value, f_value = invert_bfgs(invert_model, ftr_model,im=im_n, z_predict=z0_n, npx=npx)
        rec_im = (gx * 255).astype(np.uint8)
        fs.append(f_value[np.newaxis,...])
        zs.append(z_value[np.newaxis,...])
        recs.append(rec_im)
    recs = np.concatenate(recs, axis=0)
    zs = np.concatenate(zs, axis=0)
    fs = np.concatenate(fs, axis=0)
    return recs, zs, fs

# def do_transform(x):
#     return floatX(x).transpose(0, 3, 1, 2) / 127.5 - 1.    

# def do_inverse_transform(x, npx):
#     return (x.reshape(-1, 3, npx, npx).transpose(0, 2, 3, 1) + 1.) / 2.

def do_transform(x):
    return x * 2.0 - 1.    

def do_inverse_transform(x, npx):
    return (x + 1.0) / 2.


def invert_bfgs(invert_model, ftr_model, im, z_predict=None, npx=64):
    _f, z = invert_model
    nz = 512
    if z_predict is None:
        z_predict = np_rng.uniform(-1., 1., size=(1, nz))
    else:
        z_predict = floatX(z_predict)
    # z_predict = np.arctanh(z_predict)
    im_t = do_transform(im)
    ftr = ftr_model(im_t)

    prob = optimize.minimize(f_bfgs, z_predict, args=(_f, im_t, ftr),
                             tol=1e-6, jac=True, method='L-BFGS-B', options={'maxiter':50})
    # print('n_iters = %3d, f = %.3f' % (prob.nit, prob.fun))
    z_opt = prob.x
    z_opt_n = floatX(z_opt[np.newaxis, :])
    [f_opt, g, gx] = _f(z_opt_n, im_t, ftr)
    gx = do_inverse_transform(gx, npx=npx)
    # z_opt = np.tanh(z_opt)
    # z_opt = z_scale * z_opt
    return gx, z_opt,f_opt


def f_bfgs(z0, _f, x, x_f):
    z0_n = floatX(z0[np.newaxis, :])
    [f, g, gx] = _f(z0_n, x, x_f)
    f = f.astype(np.float64)
    g = g[0].astype(np.float64)
    return f, g


def invert_images_opt(invert_models, ims):
    invert_model, ftr_model = invert_models
    n_imgs = len(ims)
    # print('process %d images' % n_imgs)
    z_predict = None
    npx = 64
    recs, zs, loss = invert_bfgs_batch(invert_model, ftr_model, ims, z_predict=z_predict, npx=npx)

    return recs, zs, z_predict


def parse_args():
    parser = argparse.ArgumentParser(description='iGAN: Interactive Visual Synthesis Powered by GAN')
    parser.add_argument('--model_name', dest='model_name', help='the model name', default='shoes_64', type=str)
    parser.add_argument('--model_type', dest='model_type', help='the generative models and its deep learning framework', default='dcgan_theano', type=str)
    parser.add_argument('--input_image', dest='input_image', help='input image', default='./pics/shoes_test.png', type=str)
    parser.add_argument('--output_image', dest='output_image', help='output reconstruction image', default=None, type=str)
    parser.add_argument('--model_file', dest='model_file', help='the file that stores the generative model', type=str, default=None)
    # cnn: feed-forward network; opt: optimization based; cnn_opt: hybrid of the two methods
    parser.add_argument('--solver', dest='solver', help='solver (cnn, opt, or cnn_opt)', type=str, default='cnn_opt')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    if not args.model_file:  # if the model file is not specified
        args.model_file = './models/%s.%s' % (args.model_name, args.model_type)
    if not args.output_image:# if the output image path is not specified
        args.output_image = args.input_image.replace('.png', '_%s.png' % args.solver)

    for arg in vars(args):
        print('[%s] =' % arg, getattr(args, arg))

    # read a single image
    im = Image.open(args.input_image)
    [h, w] = im.size
    print('read image: %s (%dx%d)' % (args.input_image, h, w))
    # define the theano models
    model_class = locate('model_def.%s' % args.model_type)
    gen_model = model_class.Model(model_name=args.model_name, model_file=args.model_file, use_predict=True)
    invert_models = def_invert_models(gen_model, layer='conv4', alpha=0.002)
    # pre-processing steps
    npx = gen_model.npx
    im = im.resize((npx, npx))
    im = np.array(im)
    im_pre = im[np.newaxis, :, :, :]
    # run the model
    # rec, _, _  = invert_images_CNN_opt(invert_models, im_pre, solver=args.solver)
    rec, zs, _  = invert_images_opt(invert_models, im_pre, solver=args.solver)
    np.save("z_01", zs)
    rec = np.squeeze(rec)
    rec_im = Image.fromarray(rec)
    # resize the image (input aspect ratio)
    rec_im = rec_im.resize((h, w))
    print('write result to %s' % args.output_image)
    rec_im.save(args.output_image)