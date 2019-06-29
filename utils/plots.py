import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, PowerNorm, Normalize
from scipy import fftpack

def pix_intensity_hist(vals, generator, noise_vector_length, inv_transf, channel_axis, fname=None, Xterm=True, window=None, multichannel=False):
    """Plots a histogram of pixel intensities for validation set and generated samples"""
    num = len(vals)
    samples = generator.predict(np.random.normal(size=(num,1,noise_vector_length)))
    if multichannel:
        samples = np.take(samples,0,axis=channel_axis) # take the scaled channel
    samples = inv_transf(samples) # transform back to original data scale
    valhist, bin_edges = np.histogram(vals.flatten(), bins=25)
    samphist, _ = np.histogram(samples.flatten(), bins=bin_edges)
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    plt.figure()
    plt.errorbar(centers, valhist, yerr=np.sqrt(valhist), fmt='o-', label='validation')
    plt.errorbar(centers, samphist, yerr=np.sqrt(samphist), fmt='o-', label='generated')
    plt.yscale('log')
    plt.legend(loc='upper right')
    plt.xlabel('Pixel value')
    plt.ylabel('Counts')
    plt.title('Pixel Intensity Histogram')
    if window:
        plt.axis(window)
    if Xterm:
        plt.draw()
    else:
        plt.savefig(fname, format='png')
        plt.close()
    valhist = valhist[:-5]
    samphist = samphist[:-5]
    return np.sum(np.divide(np.power(valhist - samphist, 2.0), valhist))



def save_img_grid(generator, noise_vector_length, inv_transf, channel_axis, fname=None, Xterm=True, scale='lin', multichannel=False):
    """Plots a grid of generated images"""
    imgs_per_side = 2
    samples = generator.predict(np.random.normal(size=(imgs_per_side**2,1,noise_vector_length)))
    if multichannel:
        samples = np.take(samples,0,axis=channel_axis) # take the scaled channel 
    else:
        samples = np.squeeze(samples)
    genimg_sidelen = samples.shape[2]
    tot_len=imgs_per_side*genimg_sidelen
    gridimg = np.zeros((tot_len, tot_len))
    cnt = 0
    for i in range(imgs_per_side):
        for j in range(imgs_per_side):
            gridimg[i*genimg_sidelen:(i+1)*genimg_sidelen, j*genimg_sidelen:(j+1)*genimg_sidelen] \
                = samples[cnt,:,:]
            cnt += 1
    plt.figure(figsize=(5,4))
    if scale == 'pwr':
        imgmap = plt.pcolormesh(gridimg, norm=PowerNorm(gamma=0.2, vmin=0., vmax=2000),
                                cmap='Blues') # Power normalized color scale
    else:
        imgmap = plt.imshow(gridimg, cmap='Blues', norm=Normalize(vmin=-1., vmax=1.)) # Linear color scale
    plt.colorbar(imgmap)
    plt.plot([tot_len//2, tot_len //2], [0, tot_len], 'k-', linewidth='0.6')
    plt.plot([0, tot_len], [tot_len//2, tot_len //2], 'k-', linewidth='0.6')
    plt.axis([0, tot_len, 0 , tot_len])
    plt.tick_params(axis='both', which='both',bottom=False,top=False,left=False,right=False,
                    labelbottom=False, labelleft=False)
    plt.title('Generated Images')
    if Xterm:
        plt.draw()
    else:
        plt.savefig(fname, format='png')
        plt.close()


def azimuthalAverage(image, center=None):
    """
    Calculate the azimuthally averaged radial profile.

    image - The 2D image
    center - The [x,y] pixel coordinates used as the center. The default is 
             None, which then uses the center of the image (including 
             fracitonal pixels).
    
    """
    # Calculate the indices from the image
    y, x = np.indices(image.shape)

    if not center:
        center = np.array([(x.max()-x.min())/2.0, (x.max()-x.min())/2.0])

    r = np.hypot(x - center[0], y - center[1])

    # Get sorted radii
    ind = np.argsort(r.flat)
    r_sorted = r.flat[ind]
    i_sorted = image.flat[ind]

    # Get the integer part of the radii (bin size = 1)
    r_int = r_sorted.astype(int)

    # Find all pixels that fall within each radial bin.
    deltar = r_int[1:] - r_int[:-1]  # Assumes all radii represented
    rind = np.where(deltar)[0]       # location of changed radius
    nr = rind[1:] - rind[:-1]        # number of radius bin

    # Cumulative sum to figure out sums for each radius bin
    csim = np.cumsum(i_sorted, dtype=float)
    tbin = csim[rind[1:]] - csim[rind[:-1]]

    radial_prof = tbin / nr

    return radial_prof



def power_spectrum(image):
    """Computes azimuthal average of 2D power spectrum of a np array image"""
    GLOBAL_MEAN = 0.9998563 # this should be the mean pixel value of the training+validation datasets
    F1 = fftpack.fft2((image - GLOBAL_MEAN)/GLOBAL_MEAN)
    F2 = fftpack.fftshift(F1)
    pspec2d = np.abs(F2)**2
    P_k = azimuthalAverage(pspec2d)
    k = np.arange(len(P_k))
    return k, P_k


def batch_Pk(arr):
    """Computes power spectrum for a batch of images"""
    Pk_arr = []
    for idx in range(arr.shape[0]):
        k, P_k = power_spectrum(np.squeeze(arr[idx]))
        Pk_arr.append(P_k)
    return k, np.array(Pk_arr)


def pspect(val_imgs, generator, invtransform, noise_vect_len, channel_axis, fname=None, Xterm=True, multichannel=False):
    """plots mean and std deviation of power spectrum over validation set + generated samples"""
    num = val_imgs.shape[0]
    gen_imgs = generator.predict(np.random.normal(size=(num,1,noise_vect_len)))
    if multichannel:
        gen_imgs = np.take(gen_imgs,0,axis=channel_axis) # take the scaled channel 
    else:
        gen_imgs = np.squeeze(gen_imgs)
    gen_imgs = invtransform(gen_imgs)
    k, Pk_val = batch_Pk(val_imgs)
    k, Pk_gen = batch_Pk(gen_imgs)

    val_mean = np.mean(Pk_val, axis=0)
    gen_mean = np.mean(Pk_gen, axis=0)
    val_std = np.std(Pk_val, axis=0)
    gen_std = np.std(Pk_gen, axis=0)

    plt.figure()
    plt.fill_between(k, gen_mean - gen_std, gen_mean + gen_std, color='red', alpha=0.4)
    plt.plot(k, gen_mean, 'r--')
    plt.plot(k, val_mean, 'k:')
    plt.plot(k, val_mean + val_std, 'k-')
    plt.plot(k, val_mean - val_std, 'k-')
    plt.xscale('log')
    plt.yscale('log')
    plt.ylabel(r'$P(k)$')
    plt.xlabel(r'$k$')
    plt.title('Power Spectrum')
    if Xterm:
        plt.draw()
    else:
        plt.savefig(fname, format='png')
        plt.close()
    return np.sum(np.divide(np.power(gen_mean - val_mean, 2.0), val_mean))

