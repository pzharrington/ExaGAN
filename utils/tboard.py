import tensorflow as tf
import keras
import io
import plots
import numpy as np


def _filebuf_to_tf_summary_img(filebuf, name):
    return tf.Summary.Image(encoded_image_string=filebuf.getvalue())



class TboardImg(keras.callbacks.Callback):

    def __init__(self, title):
        super().__init__()
        self.title = title

    def on_epoch_end(self, iternum, GAN):
        out = io.BytesIO()
        val = None
        if self.title == 'genimg':
            plots.save_img_grid(GAN.genrtor, GAN.noise_vect_len, GAN.invtransform, GAN.C_axis, 
                                multichannel=GAN.multichannel, fname=out, Xterm=False, scale=GAN.cscale)
        elif self.title == 'pixhist':
            val = plots.pix_intensity_hist(GAN.val_imgs, GAN.genrtor, GAN.noise_vect_len, 
                                           GAN.invtransform, GAN.C_axis, multichannel=GAN.multichannel, fname=out, Xterm=False)
        elif self.title == 'pspect':
            val = plots.pspect(GAN.val_imgs, GAN.genrtor, GAN.invtransform, 
                               GAN.noise_vect_len, GAN.C_axis, fname=out, Xterm=False, multichannel=GAN.multichannel)
        out.seek(0)
        image = _filebuf_to_tf_summary_img(out, self.title)
        summary = tf.Summary(value=[tf.Summary.Value(tag=self.title, image=image)])
        writer = tf.summary.FileWriter(GAN.expDir+'logs/imgs')
        writer.add_summary(summary, iternum)
        writer.close()
        out.close()
        return val

class TboardScalars(keras.callbacks.Callback):
    
    def __init__(self):
        super().__init__()
    
    def on_epoch_end(self, GAN, iternum, logs):
        writer = tf.summary.FileWriter(GAN.expDir+'logs/scalars')
        for scalar_name in logs:
            summary = tf.Summary(value=[tf.Summary.Value(tag=scalar_name, simple_value=logs[scalar_name])])
            writer.add_summary(summary, iternum)
        writer.close()

