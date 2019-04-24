import spike2_data_puller as dp
import numpy as np
import power_analyses as pa
import matplotlib.pyplot as plt
from quantities import s, dimensionless
import csv


def clip_data(data, fs, start, end):
    """
    returns the data over a time interval

    Parameters
    ----------
    data: The data to be cropped
    fs: the sampling rate of data (in Hz)
    start: The time at which the crop should start (in seconds)
    end: The time at which the crop should end (in seconds)

    Returns
    ----------
    data: a subsection of data from time start to end
    """
    sf = int(np.floor(fs * start))
    ff = int(np.floor(fs * end))
    return data[sf:ff]


# def get_max_sniff(data, fs, start_cut=5, end_cut=15):
#     freq, power = pa.welch_psd(data, fs, False)
#
#     max_pow = -11111
#     max_freq = -111111
#     for i in range(len(freq)):
#         if power[i] > max_pow and start_cut < freq[i] < end_cut:
#             max_pow = power[i]
#             max_freq = freq[i]
#         elif freq[i] > end_cut:
#             break
#
#     return max_freq,


def integrate_sniff_power(data, fs, start_cut=7, end_cut=12):
    """
    returns power of the frequencies in range start_cut to end_cut

    Parameters
    ----------
    data: The data whose power is to be calculated
    fs: the sampling rate of data (in Hz)
    start_cut: The lower bound of the frequency range (in Hz)
    end_cut: The upper bound of the frequency range (in Hz)

    Returns
    ----------
    adder: The approximate result of the integration of the welch PSD (Power Spectral Density)
        of data from start_cut to end_cut
    """
    freq, power = pa.welch_psd(data, fs, False, len_seg=256*4)
    c = 0
    adder = 0
    for i in range(len(freq)):
        if start_cut < freq[i] < end_cut:
            adder = adder + (power[i] * float(freq[i+1]-freq[i]))
            c = c + 1
    return adder


def get_power_for_data(data, fs, start_list, time_range):
    """
    returns a list of the powers of data for time windows indicated by start_list over a
    frequency range specified in integrate_sniff_power

    Parameters
    ----------
    data: The data whose power is to be calculated
    fs: the sampling rate of data (in Hz)
    start_list: a list of the starting time of each time window (each element in seconds)
    time_range: the width of each time window (in s)

    Returns
    ----------
    ps: A list of the power for each time window in the order of start_list.
    """
    ps = []
    for st in start_list:
        e = float(st) + time_range
        if int(np.floor(fs * e)) < len(data):
            d_temp = clip_data(data, fs, st, e)
            power = integrate_sniff_power(d_temp, fs)
            ps.append(power)
    return ps


def lst_sum(l1, l2):
    """ Adds l1 and l2 element-wise and returns the result """
    assert len(l1) == len(l2)
    r = []
    for i in range(len(l1)):
        r.append(l1[i]+l2[i])
    return np.array(r)


def lst_div(l1, d):
    """ Divides each element in l1 by d and returns the result """
    r = []
    for i in range(len(l1)):
        r.append(l1[i]/d)
    return np.array(r)


def write_csv(a):
    """
    writes a 2D array into a csv file

    Parameters
    ----------
    a: The 2D array to be written

    Returns nothing, writes a csv file named 'habituation_data.csv
    """
    with open('habituation_data.csv', mode='w') as hab_file:
        employee_writer = csv.writer(hab_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for i in range(np.size(a,1)):
            employee_writer.writerow(a[:, i])


def pows_on_many_data(file_names, type, time_range=10):
    """
    for each file in file_names, determines the power for each window after an event, and returns a matrix with
    those powers. Does not include events from where the window goes out of bounds.

    Parameters
    ----------
    file_names: A list of file names from where the data is collected
    type: a string that specifies what data is to be analysed. eg: 'Sniff' or 'LFP1'
    time_range: the width of the window considered when getting the data

    Returns
    ----------
    final_pow: A matrix whose rows are the individual file's powers, and columns are the data for each event

    Requires each file to have the same number of events occur. Groups events together in the order of
    appearance, and not by event name.
    """

    print('start')
    ie, w = dp.get_events(file_names[0])
    print(ie)
    l = len(ie) - 1
    final_pow = np.zeros((l, len(file_names)))
    i = 0
    for name in file_names:
        data, fs = dp.get_data(name, type)
        event_times, event_labels = dp.get_events(name)
        powers = get_power_for_data(data, fs, event_times, time_range)
        final_pow[:, i] = powers
        print('pow = ', powers)
        i = i + 1

    out = np.mean(final_pow, axis=1)

    plt.bar((range(l)), out)
    plt.show()

    return final_pow


file_names = [r'E:\sniffer_data_spring_2019\ParameterTest_OE1_041619_odors.smr',
              r'E:\sniffer_data_spring_2019\ParameterTest_OE2_041619_odors.smr',
              r'E:\sniffer_data_spring_2019\ParameterTest_OE3_041619_odors.smr',

              r'E:\sniffer_data_spring_2019\ParameterTest_OE1_041719_odors.smr',
              r'E:\sniffer_data_spring_2019\ParameterTest_OE2_041719_odors.smr',
              r'E:\sniffer_data_spring_2019\ParameterTest_OE3_041719_odors.smr',

              r'E:\sniffer_data_spring_2019\ParameterTest_OE1_041819_odors.smr',
              r'E:\sniffer_data_spring_2019\ParameterTest_OE2_041819_odors.smr',
              r'E:\sniffer_data_spring_2019\ParameterTest_OE3_041819_odors.smr',

              r'E:\sniffer_data_spring_2019\ParameterTest_OE1_041919_odors.smr',
              r'E:\sniffer_data_spring_2019\ParameterTest_OE2_041919_odors.smr',
              r'E:\sniffer_data_spring_2019\ParameterTest_OE3_041919_odors.smr']

write_csv(pows_on_many_data(file_names, 'Sniff'))
