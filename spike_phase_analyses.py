#import elephant
#import elephant_main
import neo
import math
import scipy.signal as signal
from scipy.signal import hilbert
from quantities import ms, Hz, uV, s
from filters import cheby2_bp_filter, butter_lowpass_filter
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from neo.io.nixio import NixIO
from neo.core import AnalogSignal, SpikeTrain
from pip._vendor.progress import spinner
import spike2_data_puller as dp
import scipy as scipy
from quantities import s, dimensionless, ms
#Written by Jesse Werth

degree_sign = u"\u00b0"

rfs = 1000*Hz

def phase_histogram(signal, spikes, before=200*ms, after=200*ms, bins=72, filter_range=[25,35], del_zero = True):
    """
    Creates and displays a circular histogram showing the distribution of oscillation
    phases when the spikes occur over the duration of the signal.

    Parameters
    ----------
    signal: Neo Analog Signal object
        Time series of voltage values; should be a down-sampled signal.
    spikes: Array of Neo SpikeTrain objects
        Spike trains, each of which will have its own phase histogram plotted
    before: time quantity
        window of time before spike event to consider in calculating LFP phase
    after: time quantity
        window of time after spike event to consider in calculating LFP phase
    bins: int
        number of bins to use in the circular histogram
    filter_range: List of two integers or floats
        Designates the low and high bounds of the bandpass filter for the LFP
    del_zero: boolean
        If true, deletes cluster #0

    Returns
    -------
    Displays circular histogram using plt.show(); doesn't return any actual values
    """

    cluster_num = 0
    if del_zero:
        spikes = np.delete(spikes, 0)
        cluster_num = 1
    rfs = signal.sampling_rate
    low = filter_range[0]
    high = filter_range[1]

    sig_start = signal.t_start
    #signal = cheby2_bp_filter(signal, low, high, rfs, order=5, axis=0)
    #signal = neo.core.AnalogSignal(signal, units=uV, sampling_rate=rfs, t_start=sig_start)

    pdone = 0
    spikes=spikes[1::]
    for spike_train in spikes:
        spike_train = spike_train[:len(spike_train)-10:1]  # to make the code run faster

        all_phases = [None] * len(spike_train)
        count = 0
        for n1 in spike_train:
            n = round(n1, 1000. / rfs)
            print(count/len(spike_train))
            spike_start = n - before
            if spike_start < spike_train.t_start:
                spike_start = spike_train.t_start
            spike_finish = n + after
            if spike_finish > spike_train.t_stop:
                spike_finish = spike_train.t_stop

            spike_start, spike_finish = round(spike_start, 1), round(spike_finish, 1)


            lfp = signal.time_slice(spike_start, spike_finish)
            analytic_signal = hilbert(lfp, None, 0)
            instantaneous_phase = np.rad2deg(np.unwrap(np.angle(analytic_signal, deg=False)))  # changed, unwrap uses rads

            n = n - n % (round(lfp.times.rescale(ms), 1)[1] - round(lfp.times.rescale(ms), 1)[0])
            n = round(n.rescale(ms), 1)

            phase_dictionary = dict(zip(list(map(str,(round(lfp.times.rescale(ms), 1)))), instantaneous_phase))
            ip_of_spike = phase_dictionary[str(n)]
            all_phases[count] = ip_of_spike
            count = count + 1
        """
        create the histogram, and caculate vector sum
        """

        if len(all_phases) == 0:
            cluster_num = cluster_num + 1
            continue

        number_bins = bins
        bin_size = 360/number_bins
        print(all_phases)
        counts, theta = np.histogram(np.squeeze(all_phases), np.arange(-180, 180 + bin_size, bin_size))
        theta = theta[:-1]+ bin_size/2.
        theta = theta * np.pi / 180
        a_deg = map(lambda x: np.ndarray.item(x), all_phases)
        a_rad = map(lambda x: math.radians(x), a_deg)
        a_rad = np.fromiter(a_rad, dtype=np.float)

        a_cos = map(lambda x: math.cos(x), a_rad)
        a_sin = map(lambda x: math.sin(x), a_rad)
        a_cos, a_sin = np.fromiter(a_cos, dtype=np.float), np.fromiter(a_sin, dtype=np.float)

        uv_x = sum(a_cos)/len(a_cos)
        uv_y = sum(a_sin)/len(a_sin)
        uv_radius = np.sqrt((uv_x*uv_x) + (uv_y*uv_y)) * max(counts)
        uv_phase = np.angle(complex(uv_x, uv_y))
        uv_phase_deg = round(np.rad2deg(uv_phase),1)
        if uv_phase_deg < 0:
            uv_phase_deg = uv_phase_deg + 360
        """
        plot histogram and vector sum
        """
        fig = plt.figure()
        ax1 = fig.add_axes([0.1, 0.16, 0.05, 0.56])
        ax2 = fig.add_axes([0.75, 0.16, 0.05, 0.56], frameon=False, xticks=(), yticks=())

        histo = fig.add_subplot(111, polar=True)
        histo.yaxis.set_ticks(())

        plt.suptitle("Phase distribution for Neuron #" + str(cluster_num), fontsize=15, y=.94)
        plt.subplots_adjust(bottom=0.12, right=0.90, top=0.78, wspace=0.4)
        width = (2*np.pi) / number_bins
        bars = histo.bar(theta, counts, width = width, bottom=0.000)

        for r, bar in zip(counts, bars):
            bar.set_facecolor(plt.cm.jet(r / max(counts)))
            bar.set_alpha(0.7)

        norm = matplotlib.colors.Normalize(vmin=(counts.min())*len(all_phases)*bin_size, vmax=(counts.max())*len(all_phases)*bin_size)
        cb1 = matplotlib.colorbar.ColorbarBase(ax1, cmap=plt.cm.jet,
                           orientation='vertical', norm=norm, alpha=0.4,)
                         # ticks=np.arange(0,(counts.max())*len(all_phases)*bin_size), )
        cb1.ax.tick_params(labelsize=11)
        cb1.solids.set_rasterized(True)
        cb1.set_label("# spikes")
        cb1.ax.yaxis.set_label_position('left')

        histo.annotate('',xy=(uv_phase, uv_radius), xytext=(uv_phase,0), xycoords='data', arrowprops=dict(width=3, color='black', alpha=0.6, headwidth = 7, headlength=7))
        ax2.annotate("Number of spikes: " + str(len(all_phases)) + "\nVector sum length: " + str(round((uv_radius/max(counts)), 3)) + "\nVector sum phase: " +
            str(uv_phase_deg) + degree_sign + "\np = " + str(np.round(np.exp(-1*len(all_phases)*(uv_radius/max(counts))**2), 6)), xy=(0,1), fontsize=8)

        plt.show()
        cluster_num = cluster_num + 1
        return 1



if __name__ == "__main__":
    base_filename = r'E:\sniffer_data_spring_2019\ParameterTest_OE2_022219_odors.smr'
    electrode_name = "84"
    #h5_filename = base_filename.format(wave_clus="", electrode="", ext="h5")
    #mat_filename = base_filename.format(wave_clus="times_", electrode="_" + electrode_name, ext="mat")

    fs = 1000 * Hz
    #raw_signal = elephant_main.get_electrode_data(h5_filename, electrode_name)
    raw_signal = dp.get_data(base_filename, 'ignore', id_no=13, simplified=False)
    duration = raw_signal.t_stop - raw_signal.t_start
    num_samples = int(duration.rescale(s) * fs / Hz)
    #resampled_signal = butter_lowpass_filter(raw_signal, fs / 2, raw_signal.sampling_rate)
    resampled_signal = signal.resample(raw_signal, num_samples)
    resampled_signal = neo.core.AnalogSignal(resampled_signal, units=uV, sampling_rate=fs)
    spike_trains = dp.get_spike_train(base_filename, simplified=False)
    phase_histogram(resampled_signal, spike_trains)
#
# fname = r'E:\sniffer_data_spring_2019\ParameterTest_OE2_022219_odors.smr'
# trains = dp.get_spike_train(fname, False)
# data = dp.get_data(fname, 'U1_2LFP', False)
# print()
# data.as_array = scipy.signal.decimate(np.squeeze(data.as_array()), 100)
# data.sampling_rate = data.sampling_rate/100
# phase_histogram(data, trains)