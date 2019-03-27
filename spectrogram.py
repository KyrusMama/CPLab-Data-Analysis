import numpy as np
import scipy.io
from scipy import signal
import matplotlib.pylab as plt
import matplotlib
from quantities import Hz, uV, ms, s
from numpy import meshgrid
import scipy.ndimage as ndi
from filters import butter_lowpass_filter, iir_notch
import spike2_data_puller as dp
#Written by Jesse Werth



def plot_spectrogram(data, fs, levels=100, sigma=1, perc_low=1, perc_high=99, nfft=1024, noverlap=512):
    """
    Data
    ------------
    data: Quantity array or Numpy ndarray
        Your time series of voltage values
    fs: Quantity
        Sampling rate in Hz


    Spectrogram parameters
    ------------
    levels: int
        The number of color levels displayed in the contour plot (spectrogram)
    sigma: int
        The standard deviation argument for the gaussian blur
    perc_low, perc_high: int
        Out of the powers displayed in your spectrogram, these are the low and high percentiles
        which mark the low and high ends of your colorbar.

        E.g., there might be a period in the start of the experiment where the voltage time series shifts
        abruptly, which wouldappear as a vertical 'bar' of high power in the spectrogram; setting perc_high
        to a value lower than 100 will make the color bar ignore these higher values (>perc_high), and
        display the 'hottest' colors as the highest power values other than these (<perc_high), allowing
        for better visualization of actual data. Similar effects can be accomplished with vmin/vmax args
        to countourf.
    nfft: int
        The number of data points used in each window of the FFT.
        Argument is directly passed on to matplotlib.specgram(). See the documentation
        `matplotlib.specgram()` for options.
    noverlap: int
        The number of data points that overlap between FFT windows.
        Argument is directly passed on to matplotlib.specgram(). See the documentation
        `matplotlib.specgram()` for options.
    """

    plt.rcParams['image.cmap'] = 'jet'

    spec, freqs, bins, __ = plt.specgram(data, NFFT=nfft, Fs=int(fs), noverlap=noverlap) #gives us time and frequency bins with their power
    Z = np.flipud(np.log10(spec)*10)
    Z = ndi.gaussian_filter(Z, 1)
    extent = 0, np.amax(bins), freqs[0], freqs[-1]
    levels = np.linspace(np.percentile(Z, perc_low), np.percentile(Z, perc_high), 100)
    x1, y1 = np.meshgrid(bins, freqs)
    plt.ylabel('Frequency (Hz)', fontsize=12)
    plt.xlabel('Time (seconds)', fontsize=12)
    plt.suptitle("Spectrogram title", fontsize=15, y=.94)
    plt.contourf(x1, list(reversed(y1)), Z, vmin=None, vmax=None, extent=extent, levels=levels)
    plt.colorbar()
    plt.axis('auto')
    plt.show()

fname = r'E:\sniffer_data_spring_2019\ParameterTest_OE2_022219_odors.smr'
data, fs = dp.get_data(fname, 'Sniff')
plot_spectrogram(data,fs)