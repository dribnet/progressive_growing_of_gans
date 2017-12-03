import numpy as np
import misc
from theano import tensor as T
import train

from iganpredict import def_invert_models, invert_images_opt

class Net: pass

class ProGroModel:
    def __init__(self, filename=None, model=None):
        if model is not None:
            # error
            return

        _, _, G = misc.load_pkl(filename)
        self.net = Net()
        self.net.G = G
        self.have_compiled = False
        self.net.labels_var  = T.TensorType('float32', [False] * 512) ('labels_var')

        # experiment
        num_example_latents = 10
        self.net.example_latents = train.random_latents(num_example_latents, self.net.G.input_shape)
        self.net.example_labels = self.net.example_latents
        self.net.latents_var = T.TensorType('float32', [False] * len(self.net.example_latents.shape))('latents_var')
        self.net.labels_var  = T.TensorType('float32', [False] * len(self.net.example_latents.shape)) ('labels_var')

        self.net.images_expr = self.net.G.eval(self.net.latents_var, self.net.labels_var, ignore_unused_inputs=True)
        self.net.images_expr = misc.adjust_dynamic_range(self.net.images_expr, [-1,1], [0,1])
        train.imgapi_compile_gen_fn(self.net)

        self.invert_models = def_invert_models(self.net, layer='conv4', alpha=0.002)

    def encode_images(self, images, cond=None):
        # print("images: ", images.shape, images[0][0])
        rec, zs, _  = invert_images_opt(self.invert_models, images)

        # pixels = (255 * images).astype(np.uint8)
        # pixels = np.swapaxes(pixels,1,2)
        # pixels = np.swapaxes(pixels,2,3)
        # # print("SHAPE: {} {}".format(pixels.shape, pixels.dtype))
        # _, zs, _  = invert_images_CNN_opt(self.invert_models, pixels, solver='cnn_opt', npx=self.model_G.npx)
        # print(zs)
        # print("Zs SHAPE: {}".format(zs.shape))
        return zs

    def get_zdim(self):
        # ?
        return 512

    def sample_at(self, z):
        mapped_latents = z.astype(np.float32)
        samples = self.net.gen_fn(mapped_latents, mapped_latents)
        samples = np.clip(samples, 0, 1)

        # self.net.example_latents = z.astype(np.float32)
        # self.net.example_labels = self.net.example_latents
        # self.net.latents_var = T.TensorType('float32', [False] * len(self.net.example_latents.shape))('latents_var')
        # self.net.labels_var  = T.TensorType('float32', [False] * len(self.net.example_latents.shape)) ('labels_var')

        # self.net.images_expr = self.net.G.eval(self.net.example_latents, self.net.labels_var, ignore_unused_inputs=True)
        # self.net.images_expr = misc.adjust_dynamic_range(self.net.images_expr, [-1,1], [0,1])

        # if not self.have_compiled:
        #     train.imgapi_compile_gen_fn(self.net)
        #     self.have_compiled = True

        # samples = self.net.gen_fn(self.net.example_latents, self.net.example_labels)
        # samples = np.clip(samples, 0, 1)

        # print("Samples: ", samples.shape)
        # samples = (samples + 1.0) / 2.0
        return samples
