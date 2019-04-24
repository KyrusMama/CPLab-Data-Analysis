from scipy import signal
import scipy.ndimage as ndi
from quantities import ms, Hz, uV, s
from filters import butter_lowpass_filter, iir_notch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import spike2_data_puller as dp

#Written by Jesse Werth

def welch_psd(data, fs, plot=True, window='hanning', overlap=0.5, len_seg=None, axis=-1):
    """
    Estimates and plots power spectrum density (PSD) of a given time series using Welch's method

    Parameters
    ----------
    data: Quantity array or Numpy ndarray
        Time series data, of which PSD is estimated.
    fs: Quantity
        Sampling rate in Hz
    plot: Boolean
        If true, plots
    window: str
        The type of window used in power calculation. Argument is directly
        passed on to scipy.signal.welch(). See the documentation
        `scipy.signal.welch()` for options.
    overlap: float
        Overlap between segments represented as a float number between 0 (no
        overlap) and 1 (complete overlap). Default is 0.5 (half-overlapped).
    len_seg: int
        Length of windows that data is broken up into.
    axis: int
        Axis along which data is analyzed.

    Returns
    -------
    freqs: Quantity array or Numpy ndarray
        Frequencies associated with the power estimates in `psd`. `freqs` is
        a 1-dimensional array irrespective of the shape of the input data.
    psd: Quantity array or Numpy ndarray
        PSD estimates of the time series in `signal`.
    """
#     data = data.reshape(data.shape[0],)
    if len_seg is None:
        overlap = overlap * 256
    else:
        overlap = overlap * len_seg
    freqs, psd = signal.welch(data, fs=fs, noverlap=overlap, nperseg=len_seg,
              window=window, nfft=None, detrend='constant',
              return_onesided=True, scaling='density', axis=axis)
    if plot:
        plt.plot(freqs, psd)
        plt.show()
    return freqs, psd


def welch_change_in_power(data, fs, pre, post, duration=60*s, low=20*Hz, high=55*Hz, axis=-1):
    """
    Estimates the change in power (of a given frequency band) between two time
    periods (pre- and post-) using Welch's method.

    Parameters
    ----------
    data: Quantity array or Numpy ndarray
        Time series data, of which PSD is estimated.
    fs: quantity
        Sampling rate in Hz.
    pre: quantity
        The start time of the first ("pre-") time period, given in a time quantity
    post: quantity
        The start time of the second ("post-") time period, given in a time quantity.
    duration: quantity
        The trial duration, given in a time quantity. Default is one minute.
    low: quantity
        The low end of the frequency band over which power is analyzed, given in Hz.
        Default is 20*Hz, corresponding to in vitro bulbar gamma.
    high: quantity
        The high end of the frequency band over which power is analyzed, given in Hz.
        Default is 55*Hz, corresponding to in vitro bulbar gamma.
    axis: int
        Axis along which data is analyzed.

    Returns
    -------
    freqs: Quantity array or Numpy ndarray
        Frequencies associated with the power estimates in `psd`. `freqs` is
        a 1-dimensional array irrespective of the shape of the input data.
    psd: Quantity array or Numpy ndarray
        PSD estimates of the time series in `signal`.
    """
    pre_start, pre_stop = pre.rescale(ms), pre.rescale(ms) + duration.rescale(ms)
    post_start, post_stop = post.rescale(ms), post.rescale(ms) + duration.rescale(ms)
    fs = float(data.sampling_rate)

    # Calculate for pre period
    # NOTE: Time_slice is a function from the elephant package used to slice an array
    # containing data from the whole experiment to just part of it (i.e. pre- period).
    # If you don't use the Elephant package, then just rewrite line 96 and 105 so
    # that it slices your array based on pre- and post- times.
    pre_signal = data.time_slice(pre_start, pre_stop)
    freqs, pre_spec = welch_psd(pre_signal, fs, plot=False, axis=axis)
    pre_spec = pre_spec * Hz/uV**2
    fl = np.sum(freqs<low)
    fh = np.sum(freqs<high)
    pre_gamma = pre_spec[fl:fh]
    pre_power = np.sum(pre_gamma)

    # Calculate for post period
    post_signal = data.time_slice(post_start, post_stop)
    freqs, post_spec = welch_psd(post_signal, fs, plot=False, axis=axis)
    post_spec = post_spec * Hz/uV**2
    fl = np.sum(freqs<low)
    fh = np.sum(freqs<high)
    post_gamma = post_spec[fl:fh]
    post_power = np.sum(post_gamma)

    # Find change in power
    perc = np.float(100 * (post_power - pre_power) / pre_power)
    db = np.float(10 * np.log(post_power/pre_power))
    print("Pre power: " + str(np.round(pre_power,5)))
    print("Post power: " + str(np.round(post_power,5)))
    print("Change in power: " + str(np.round(db,3)) + " decibels (" + str(np.round(perc, 3)) + " %)")


def clip_data(data, fs, start, end):
    sf = int(np.floor(fs * start))
    ff = int(np.floor(fs * end))
    return data[sf:ff]

# filename = r'E:\sniffer_data_spring_2019\ParameterTest_OE1_041919_odors.smr'
#
# data, fs = dp.get_data(filename, 'Sniff')
# data = np.array(data)
# d1 = clip_data(data, fs, 616, 631)
# d2 = clip_data(data, fs, 741, 756)
# welch_psd(d1, fs)
# welch_psd(d2, fs)
