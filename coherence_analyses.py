import numpy as np
from scipy import signal
import matplotlib.pylab as plt
import matplotlib
from quantities import Hz, uV, ms, s
import scipy.ndimage as ndi
import scipy.io
from filters import butter_lowpass_filter
import spike2_data_puller as dp
import math
#Written by Jesse Werth



def plot_coherence(a, b, fs, mag_squared=True, overlap=0.5, window='hamming',
                   len_seg=None, axis=0):
    """
    Returns and plots the coherence between two signals.

    Parameters
    ----------
    a, b: Quantity array or Numpy ndarray
        A pair of time series data, between which coherence is computed. The
        shapes and the sampling frequencies of `x` and `y` must be identical.
    fs: Quantity
        Sampling rate in Hz
    mag_squared:
        If true, coherence reported is the magnitude-squared coherence. If false,
        coherence is reported as magnitude.
    overlap: float
        Overlap between segments represented as a float number between 0 (no
        overlap) and 1 (complete overlap). Default is 0.5 (half-overlapped).
    window: string
        These arguments are directly passed on to a helper function
        `elephant.spectral._welch()`. See the respective descriptions in the
        docstring of `elephant.spectral._welch()` for usage.
    len_seg: int
        Length of windows that data is broken up into.
    axis: int
        Axis along which data is analyzed.

    Returns
    ----------
    freqs: ndarray
        Array of sample frequencies
    coherency: ndarray
        Coherence values of a and b
    """
    a = a.reshape(a.shape[0],)
    b = b.reshape(b.shape[0],)
    if len_seg == None:
        noverlap = overlap * int(fs)
        freqs, coherency = signal.coherence(a, b, fs, window=window, nperseg=int(fs),
                                            noverlap=noverlap, axis=axis)
    else:
        noverlap = overlap * len_seg
        freqs, coherency = signal.coherence(a, b, fs, window=window, nperseg=len_seg,
                                            noverlap=noverlap, axis=axis)
    if not mag_squared:
        coherency = map(lambda x: np.sqrt(x), coherency)
    # print(freqs, coherency)
    plt.plot(freqs, coherency)
    plt.show()
    return freqs, coherency


def coherence_over_band(a, b, fs, low=20*Hz, high=55*Hz, plot=True, mag_squared=True,
                        overlap=0.5, window='hamming', len_seg=None, axis=0):
    """
    Calculates the maximum and integrated coherence (using scipy.signal.Welch) in a given frequency band.

    Parameters
    ----------
    a, b: Quantity array or Numpy ndarray
        A pair of time series data, between which coherence is computed. The
        shapes and the sampling frequencies of `x` and `y` must be identical.
    fs: Quantity
        Sampling rate in Hz
    low: quantity
        The low end of the frequency band over which power is analyzed, given in Hz.
        Default is 20*Hz, corresponding to in vitro bulbar gamma.
    high: quantity
        The high end of the frequency band over which power is analyzed, given in Hz.
        Default is 55*Hz, corresponding to in vitro bulbar gamma.
    plot: boolean
        If true, plots the coherence between the signals as a function of frequency.
    mag_squared:
        If true, coherence reported is the magnitude-squared coherence. If false,
        coherence is reported as magnitude.
    overlap: float
        Overlap between segments represented as a float number between 0 (no
        overlap) and 1 (complete overlap). Default is 0.5 (half-overlapped).
    window: string
        These arguments are directly passed on to a helper function
        `elephant.spectral._welch()`. See the respective descriptions in the
        docstring of `elephant.spectral._welch()` for usage.
    len_seg: int
        Length of windows that data is broken up into.
    axis: int
        Axis along which data is analyzed.

    Returns
    ----------
    coh_integral: int
        Value representing the integration of Welch's coherence values over the entire
        frequency band. Because scipy.signal.welch returns two separate arrays for frequencies
        and coherence, the integral is evaluated using the trapezoidal method with Numpy.trapz
    coh_max: int
        Value representing the maximum coherence value across the whole frequency band
    """
    a = a.reshape(a.shape[0],)
    b = b.reshape(b.shape[0],)
    if len_seg == None:
        noverlap = overlap * int(fs)
        freqs, coherency = signal.coherence(a, b, fs, window=window, nperseg=int(fs),
                                            noverlap=noverlap, axis=axis)
    else:
        noverlap = overlap * len_seg
        freqs, coherency = signal.coherence(a, b, fs, window=window, nperseg=len_seg,
                                            noverlap=noverlap, axis=axis)
    df = (freqs[1] - freqs[0])/Hz
    low, high = int(low), int(high)
    if low%df == 0 and high%df == 0:
        low_int_bound = int(low/df)
        high_int_bound = int(high/df) + 1
        if not mag_squared:
            coherency = map(lambda x: np.sqrt(x), coherency)
        coh_integral = np.trapz(coherency[low_int_bound:high_int_bound], dx=df, axis=0)
        coh_max = max(coherency[low_int_bound:high_int_bound])
        coh_integral, coh_max = coh_integral.item(), coh_max
    else:
        raise ValueError("The low and high bounds of freqband, " + str(low) + "-" + str(high) + "must be divisible by the frequency increment, " + str(df) + "for numpy.trapz to work.")
        coh_integral, coh_max = 'undefined'
    print("----- Welch's method used to calculate coherence in the frequency band " + str(low) + " Hz to " + str(high) + " Hz -----")
    print("    Maximum coherence value = " + str(coh_max))
    print("    Integral over band = " + str(coh_integral))
    if plot:
        plt.plot(freqs, np.sqrt(coherency))
        plt.show()
    return coh_integral, coh_max


def downsample_to_a_frequency(l1, fs1, l2, fs2, fs):
    """
        down samples two lists l1 and l2, with frequencies fs1 and fs2, to a new frequency fs
        Parameters
        ----------
        l1: A numpy 1D array,
            the data wou want to down sample
        fs1: float,
            the frequency of the data in l1. fs1 = approximately how many elements in l1 represent a second
        l2: A numpy 1D array,
            the data wou want to down sample
        fs2: float,
            the frequency of the data in l2. fs2 = approximately how many elements in l2 represent a second
        fs: float,
            the frequency that l1 and l2 should be down sampled to. Should be less than fs1 and fs2.

        Returns
        ----------
        l1p, l2p: where l1 and l2 are the down sampled data. Their frequency will be as close to fs as possible.
            for both l1 and l2, len(li)/fsi  should be approximately equal to len(lip)/fs
        """
    l1p = l1[::int(fs1/fs)]
    l2p = l2[::int(fs2/fs)]
    return l1p, l2p


fname = r'E:\sniffer_data_spring_2019\ParameterTest_OE2_022219_odors.smr'
lfp, fs1 = dp.get_data(fname, 'ignore', id_no=13)
sniff, fs2 = dp.get_data(fname, 'ignore', id_no=12)
fs = 52
a, b = downsample_to_a_frequency(lfp, fs1, sniff, fs2, fs)

plot_coherence(a, b, fs)
