import numpy as np
import keras
from keras.layers import *
from keras.activations import relu
from keras.models import Model, Sequential
from keras.models import load_model
import keras.backend as K
import time
import sys
sys.path.append('./utils')
sys.path.append('./networks')
import logging
import logging_utils
from parameters import load_params
import plots
import tboard
from DeconvLayer_output_shape import MyConv2DTranspose




class DCGAN:
    
    def __init__(self, configtag, expDir):

        # Load hyperparmeters
        self.configtag = configtag
        logging.info('Parameters:')
        self.init_params(configtag)
        self.expDir = expDir
        self.inits = {'dense':keras.initializers.RandomNormal(mean=0.0, stddev=0.02),
                      'conv':keras.initializers.TruncatedNormal(mean=0.0, stddev=0.02),
                      'tconv':keras.initializers.RandomNormal(mean=0.0, stddev=0.02)}

        # Import slices
        self.real_imgs = np.load('./data/'+self.dataname+'_train.npy')
        self.val_imgs = np.load('./data/'+self.dataname+'_val.npy')
        self.n_imgs = self.real_imgs.shape[0]
        if self.datafmt == 'channels_first':
            self.real_imgs = np.moveaxis(self.real_imgs, -1, 1)
            self.val_imgs = np.moveaxis(self.val_imgs, -1, 1)
        self.set_transf_funcs()
        self.set_transf_funcs_tensor()
        self.real_imgs = self.transform(self.real_imgs)
        


        # Build networks
        self.discrim = self.build_discriminator()
        self.genrtor = self.build_generator()

        # Custom loss/metrics
        def mean_prob(y_true, y_pred):
            '''metric to measure mean probability of D predictions (0=fake, 1=real)'''
            return K.mean(K.sigmoid(y_pred))
        def crossentropy_from_logits(y_true, y_pred):
            '''crossentropy loss from logits (circumvents default Keras crossentropy loss, which is unstable)'''
            return K.mean(K.binary_crossentropy(y_true, y_pred, from_logits=True), axis=-1)
        
        # Compile discriminator so it can be trained separately
        self.discrim.compile(loss=crossentropy_from_logits, 
                             optimizer=keras.optimizers.Adam(lr=self.D_lr, beta_1=0.5),
                             metrics=[mean_prob])

        # Stack generator and discriminator networks together and compile
        z = Input(shape=(1,self.noise_vect_len))
        genimg = self.genrtor(z)
        self.discrim.trainable = False
        decision = self.discrim(genimg)
        self.stacked = Model(z, decision)
        self.stacked.compile(loss=crossentropy_from_logits, optimizer=keras.optimizers.Adam(lr=self.G_lr, beta_1=0.5))

        # Setup tensorboard stuff
        self.TB_genimg = tboard.TboardImg('genimg')
        self.TB_pixhist = tboard.TboardImg('pixhist')
        self.TB_pspect = tboard.TboardImg('pspect')
        self.TB_scalars = tboard.TboardScalars()


    def build_discriminator(self):
        
        dmodel = Sequential()
        for lyrIdx in range(self.nlayers):
            if lyrIdx==0:
                convlayer = Conv2D(filters=self.nconvfilters[lyrIdx], kernel_size=self.convkern, 
                                   strides=self.convstride, padding='same', input_shape=self.imshape, 
                                   data_format=self.datafmt, kernel_initializer=self.inits['conv'])
            else:
                convlayer = Conv2D(filters=self.nconvfilters[lyrIdx], kernel_size=self.convkern, 
                                   strides=self.convstride, padding='same', data_format=self.datafmt, 
                                   kernel_initializer=self.inits['conv'])
            dmodel.add(convlayer)
            dmodel.add(BatchNormalization(epsilon=1e-5, momentum=0.9, axis=self.C_axis))
            dmodel.add(LeakyReLU(alpha=self.alpha))
        dmodel.add(Flatten(data_format=self.datafmt))
        dmodel.add(Dense(1, kernel_initializer=self.inits['dense']))
        dmodel.summary(print_fn=logging_utils.print_func)
        img = Input(shape=self.imshape)
        return Model(img, dmodel(img))


    def build_generator(self):
        fmapsize = self.img_dim//int(2**self.nlayers)
        gmodel = Sequential()
        gmodel.add(Dense(fmapsize*fmapsize*2*self.ndeconvfilters[0], input_shape=(1,self.noise_vect_len),
                         kernel_initializer=self.inits['dense']))
        gmodel.add(BatchNormalization(epsilon=1e-5, momentum=0.9, axis=self.C_axis))
        gmodel.add(ReLU())
        if self.datafmt == 'channels_last':
            fmapshape = (fmapsize, fmapsize, 2*self.ndeconvfilters[0])
        else:
            fmapshape = (2*self.ndeconvfilters[0], fmapsize, fmapsize)
        gmodel.add(Reshape(fmapshape))
        for lyrIdx in range(self.nlayers):
            gmodel.add(MyConv2DTranspose(self.ndeconvfilters[lyrIdx], self.convkern, strides=self.convstride,
                                    padding='same', data_format=self.datafmt, kernel_initializer=self.inits['tconv'])) 
            if lyrIdx == self.nlayers - 1:
                gmodel.add(Activation('tanh')) # last layer is deconv+tanh
            else:
                gmodel.add(BatchNormalization(epsilon=1e-5, momentum=0.9, axis=self.C_axis)) # hidden layers are deconv+batchnorm+relu
                gmodel.add(ReLU())
        gmodel.summary(print_fn=logging_utils.print_func)
        noise = Input(shape=(1,self.noise_vect_len))        
        if self.multichannel:
            # multi-channel re-scaling (currently just one extra channel)
            CH1 = gmodel(noise)
            raw_rescale = Lambda(lambda inp: self.invtransform_tensor(inp)/self.linear_scaler)(CH1)
            CH2 = Activation('tanh')(raw_rescale)
            result = Concatenate(axis=self.C_axis)([CH1, CH2])
            return Model(noise, result)
        else:
            return Model(noise, gmodel(noise))


    def init_params(self,configtag):
        
        params = load_params('./config.yaml', configtag)
        logging.info(str(params))
        self.dataname = params['dataname']
        self.img_dim = params['img_dim']
        self.noise_vect_len = params['noise_vect_len']
        self.nlayers = params['nlayers']
        self.convkern = params['convkern']
        self.convstride = params['convstride']
        self.nconvfilters = params['nconvfilters']
        self.ndeconvfilters = params['ndeconvfilters']
        self.label_flip = params['label_flip']
        self.batchsize = params['batchsize'] 
        self.print_batch = params['print_batch']
        self.checkpt_batch = params['checkpt_batch']
        self.cscale = params['cscale']
        self.datascale = params['datascale']
        self.G_lr, self.D_lr = params['learn_rate']
        self.DG_update_ratio = params['DG_update_ratio']
        self.Nepochs = params['Nepochs']
        self.datafmt = params['datafmt']
        nchannels = 1
        self.multichannel = params['multichannel']
        if self.multichannel:
            self.linear_scaler = 1000.
            nchannels = 2
        self.alpha = 0.2
        self.start = 0
        self.bestchi = np.inf
        self.bestchi_pspect = np.inf
        if self.datafmt == 'channels_last':
            self.C_axis = -1
            self.imshape = (self.img_dim, self.img_dim, nchannels)
        else:
            self.C_axis = 1
            self.imshape = (nchannels, self.img_dim, self.img_dim)

    def train_epoch(self, shuffler, num_batches, epochIdx):
        
        d_losses = []
        d_real_losses = []
        d_fake_losses = []
        g_losses = []

        for batch in range(num_batches):
            iternum = (epochIdx*num_batches + batch)
            t1 = time.time()

            real_img_batch = self.real_imgs[shuffler[batch*self.batchsize:(batch+1)*self.batchsize]]
            if self.multichannel:
                CH2 = np.tanh(self.invtransform(real_img_batch)/self.linear_scaler)
                real_img_batch = np.concatenate((real_img_batch, CH2), axis=self.C_axis)
            noise_vects = np.random.normal(loc=0.0, size=(self.batchsize, 1, self.noise_vect_len))
            fake_img_batch = self.genrtor.predict(noise_vects)
            reals = np.ones((self.batchsize,1))
            fakes = np.zeros((self.batchsize,1))
            for i in range(reals.shape[0]):
                if np.random.uniform(low=0., high=1.0) < self.label_flip:
                    reals[i,0] = 0
                    fakes[i,0] = 1

            # train discriminator
            for iters in range(self.DG_update_ratio//2):
                discr_real_loss = self.discrim.train_on_batch(real_img_batch, reals)
                discr_fake_loss = self.discrim.train_on_batch(fake_img_batch, fakes)
                discr_loss = 0.5*(discr_real_loss[0]+discr_fake_loss[0])
                d_losses.append(discr_loss)
                d_real_losses.append(discr_real_loss[0])
                d_fake_losses.append(discr_fake_loss[0]) 

            # train generator via stacked model
            genrtr_loss = self.stacked.train_on_batch(noise_vects, np.ones((self.batchsize,1)))
            t2 = time.time()

            g_losses.append(genrtr_loss)

            
            t2 = time.time()

            if batch%self.print_batch == 0:
                logging.info("| --- batch %d of %d --- |"%(batch + 1, num_batches))
                logging.info("|Discr real pred=%f, fake pred=%f"%(discr_real_loss[1], discr_fake_loss[1]))
                logging.info("|Discriminator: loss=%f"%(discr_loss))
                logging.info("|Generator: loss=%f"%(genrtr_loss))
                logging.info("|Time: %f"%(t2-t1))
            if iternum%self.checkpt_batch == 0:
                # Tensorboard monitoring
                iternum = iternum/self.checkpt_batch
                self.TB_genimg.on_epoch_end(iternum, self)
                chisq = self.TB_pixhist.on_epoch_end(iternum, self)
                chisq_pspect = self.TB_pspect.on_epoch_end(iternum, self)
                scalars = {'d_loss':np.mean(d_losses), 'd_real_loss':np.mean(d_real_losses),
                           'd_fake_loss':np.mean(d_fake_losses), 'g_loss':np.mean(g_losses), 
                           'chisq':chisq, 'chisq_pspect':chisq_pspect}
                self.TB_scalars.on_epoch_end(self, iternum, scalars)

                d_losses = []
                d_real_losses = []
                d_fake_losses = []
                g_losses = []
                
                if chisq<self.bestchi and iternum>30:
                    # update best chi-square score and save
                    self.bestchi = chisq
                    self.genrtor.save_weights(self.expDir+'models/g_cosmo_best.h5')
                    self.discrim.save_weights(self.expDir+'models/d_cosmo_best.h5')
                    logging.info("BEST saved at %d, chi=%f"%(iternum, chisq))
                if chisq_pspect < self.bestchi_pspect and iternum>30:
                    self.bestchi_pspect = chisq_pspect
                    self.genrtor.save_weights(self.expDir+'models/g_cosmo_best_pspect.h5')
                    self.discrim.save_weights(self.expDir+'models/d_cosmo_best_pspect.h5')
                    logging.info("BEST_PSPECT saved at %d, chi=%f"%(iternum, chisq_pspect))


    def set_transf_funcs(self):
        def transform(x):
            return np.divide(2.*x, x + self.datascale) - 1.
        def invtransform(s):
            return self.datascale*np.divide(1. + s, 1. - s)
        self.transform = transform
        self.invtransform = invtransform

    def set_transf_funcs_tensor(self):
        def transform(x):
            return 2.*x/(x + self.datascale) - 1.
        def invtransform(s):
            return self.datascale*(1. + s)/(1. - s)
        self.transform_tensor = transform
        self.invtransform_tensor = invtransform





