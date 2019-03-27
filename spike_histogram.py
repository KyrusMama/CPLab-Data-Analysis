import spike2_data_puller as dp
import numpy as np
import matplotlib.pyplot as plt
import math
from quantities import s, dimensionless
#Written by Kyrus Mama


def spikes_events_histogram(file_name, index=-1, baseline=[b'BLANK', b'mineral oil'], bins=-1, bins_btw=5):
    """
    builds a histogram which displays spike frequency in a time window file the spikes in file_name.
    Draws vertical lines at times where events occur.


    Parameters
    ----------
    file_name: The name of the file from which you wish to retrieve the data. Must include the path.
        eg: r'E:\sniffer_data_spring_2019\ParameterTest_OE2_022219_odors.smr'
    index: The index of the set of spikes that you want to plot in the histogram.
        if index is negative, it combines all the spikes (except the 0 case) and plots them.
    baseline: a list of all the baseline event labels. This is used to draw a black line for baseline events,
        and a red line for the rest.
    bins: The number of bins the user wants in the histogram.
        if negative, calculates number of bins using bins_btw.
    bins_btw: The number of bins that the user wants per event.
        If the events are evenly spaced, this becomes the number of bins between two events.
        If bins is non-negative, this value is ignored.

    Returns
    -------
    n: A list of the number of elements in each bin.

    Throws
    -------
    FileNotFoundError: No such file or directory file_name if file_name does not exit.
    """
    trains = dp.get_spike_train(file_name)
    event_times, event_labels= dp.get_events(file_name)

    if index < 0:
        displayed_times = trains[1]
        for i in range(2, len(trains)):
            displayed_times = np.concatenate([displayed_times, trains[i]])
    else:
        displayed_times = trains[index]

    if bins < 1:
        bins = math.ceil(bins_btw * len(event_times))

    (n, q, w) = plt.hist(displayed_times, bins)
    for j in range(len(event_times)):
        if event_labels[j] in baseline:
            col = 'k'
        else:
            col = 'r'
        plt.axvline(x=event_times[j], color=col)
    print(event_labels)
    plt.show()
    return n


def averaged_spike_events_histogram(name_list, index=-1, baseline=[b'BLANK', b'mineral oil'], bins= -1, bins_btw=5, subset = False):
    """
    builds a histogram which displays spike frequency of spikes in a time window averaged across spikes from all files in name_list
    Draws vertical lines at times where events occur.


    Parameters
    ----------
    name_list: The list of file names from which you wish to retrieve the data. Must include the path.
        eg: [r'E:\sniffer_data_spring_2019\ParameterTest_OE2_022219_odors.smr']
    index: The index of the set of spikes that you want to plot in the histogram.
        if index is negative, it combines all the spikes (except the 0 case) and plots them.
    baseline: a list of all the baseline event labels. This is used to draw a black line for baseline events,
        and a red line for the rest.
    bins: The number of bins the user wants in the histogram.
        if negative, calculates number of bins using bins_btw.
    bins_btw: The number of bins that the user wants per event.
        If the events are evenly spaced, this becomes the number of bins between two events.
        If bins is non-negative, this value is ignored.

    Returns
    -------
    n: A list of the number of elements in each bin.

    Throws
    -------
    FileNotFoundError: No such file or directory file_name if file_name does not exit.
    """
    all_display_times = np.zeros(0)
    for file_name in name_list:
        trains = dp.get_spike_train(file_name)
        event_times, event_labels = dp.get_events(file_name)

        if index < 0:
            displayed_times = trains[1]
            for i in range(2, len(trains)):
                displayed_times = np.concatenate([displayed_times, trains[i]])
        else:
            displayed_times = trains[index]

        all_display_times = np.concatenate([all_display_times, displayed_times])

    if bins < 1:
        bins = math.ceil(bins_btw * len(event_times))

    if subset:
        # subset_display_times = np.random.choice(all_display_times, int(len(all_display_times) / len(name_list)),
        #                                         replace=False)
        subset_display_times = all_display_times[::len(name_list)]
    else:
        subset_display_times = all_display_times

    (n, q, w) = plt.hist(subset_display_times, bins)
    for j in range(len(event_times)):
        if event_labels[j] in baseline:
            col = 'k'
        else:
            col = 'r'
        plt.axvline(x=event_times[j], color=col)
    print(event_labels)
    plt.show()
    return n


def break_by_event(spike_times, event_times, event_names, before_event, after_event):
    """
    helper function
    groups spikes up based on which event they are near


    Parameters
    ----------
    spike_times: A list of floats each of which represents the timing of a spike
    event_times: A list of float lists each of which represents the timing of a particular type of event
    event_names: A list of event names, which correspond to the elements of event times. Must have the same length as
        event_times. event_times[i] is a list of times that the event event_names[i] occurred.
    before_event: A float. A spike is considered to be near and before an event if its timing is less than the
        events timing and if (the events timing) - (its timing) is less than before_event
    after_event: A float. A spike is considered to be near and after an event if its timing is greater than the
        events timing and if (its timing) - (the events timing) is less than after_event

    Returns
    -------
    lst: A list of numpy arrays, with the same length as event_names . Each numpy array contains
        a list of spike timings that are near the events of the corresponding event_names
    """
    lst = []
    for e in range(len(event_names)):
        sie = np.zeros(0)
        for ei in event_times[e]:
            relative_times = spike_times*s - ei
            t_spikes = relative_times[np.where(np.logical_and(- before_event < relative_times, relative_times < after_event))]
            sie = np.concatenate([sie, t_spikes])
        # sie = np.random.choice(sie, int(len(sie)/len(event_times[e])))
        # print(len(event_times[e]))
        sie = sie[::len(event_times[e])]
        lst.append(sie)
    return lst


def group_events(event_times, event_names, grouped_event_times=[], grouped_event_names=[], skip_last_event=False):
    """
    helper function
    groups the timing of events based on the event names.
    Can be used to prepare for break_by_event. The returned lists satisfy both event_times and event_names conditions.

    Parameters
    ----------
    event_times: The list of times that different events took place
    event_names: The list of names of events, corresponding to event_times. Must have the same length as event_times.
        event_names[i] happened at event_times[i]
    grouped_event_times: A list of float lists. The outer list contains a list of times for each event.
    grouped_event_names: A list of event names that corresponds to grouped_event_times. Must have the same length as
        grouped_event_times. Must not contain duplicates.
        grouped_event_names[i] happened at grouped_event_times[i][0], ... grouped_event_times[i][n],
        where n=len(grouped_event_times[i])
    skip_last_event: if True, do not include the last event in event_names in the returned lists.

    Returns
    -------
    grouped_event_times, grouped_event_names
    after adding the groupings from event_times and event_names

    Throws
    -------
    ValueError grouped_event_names must not contain duplicates if grouped_event_names contains duplicates.
    """
    assert len(event_times) == len(event_names)
    assert len(grouped_event_times) == len(grouped_event_names)

    for i in range(len(event_names) - (1 if skip_last_event else 0)):
        if event_names[i] in grouped_event_names:
            loc = grouped_event_names.index(event_names[i])
            grouped_event_times[loc].append(event_times[i])
        else:
            grouped_event_names.append(event_names[i])
            grouped_event_times.append([event_times[i]])

    return grouped_event_times, grouped_event_names


def grouped_spikes_events_histogram(file_name, *args, index=-1, baseline=[b'BLANK', b'mineral oil'],
                                    bins_btw=5, before_event=30.*s, after_event=40.*s, skip_last_event=False):
    """
    for each unique event denoted by event label, display a histogram which displays spike frequency in a time window
    around the occurrence of that event.
    Draws vertical lines at times where events occur.
    If multiple events of the same type occur, average them together

    Parameters
    ----------
    file_name: The name of the file from which you wish to retrieve the data. Must include the path.
        eg: r'E:\sniffer_data_spring_2019\ParameterTest_OE2_022219_odors.smr'
    index: The index of the set of spikes that you want to plot in the histogram.
        if index is negative, it combines all the spikes (except the 0 case) and plots them.
    baseline: a list of all the baseline event labels. This is used to draw a black line for baseline events,
        and a red line for the rest.
    bins_btw: The number of bins that the user wants per event.
    before_event: A float. A spike is considered to be near and before an event if its timing is less than the
        events timing and if (the events timing) - (its timing) is less than before_event
    after_event: A float. A spike is considered to be near and after an event if its timing is greater than the
        events timing and if (its timing) - (the events timing) is less than after_event
    skip_last_event: if True, do not include the last event in event_names in the returned lists.

    Returns nothing

    Throws FileNotFoundError: No such file or directory file_name if file_name does not exit.
    """
    trains = dp.get_spike_train(file_name)
    event_times, event_labels = dp.get_events(file_name)

    grouped_times, grouped_labels = group_events(event_times, event_labels, skip_last_event=skip_last_event)

    ttal = [0]
    for arg in args:
        tta = event_times[len(event_times)-1] + before_event + after_event
        ttal.append(tta)
        event_times, event_labels = dp.get_events(arg)
        event_times = event_times + tta
        grouped_times, grouped_labels = group_events(event_times, event_labels, grouped_times, grouped_labels,
                                                     skip_last_event=skip_last_event)

    if index < 0:
        displayed_times = trains[1]
        for i in range(2, len(trains)):
            displayed_times = np.concatenate([displayed_times, trains[i]])
    else:
        displayed_times = trains[index]

    c = 1
    for arg in args:
        tta2 = ttal[c]
        c = c+1
        trains = dp.get_spike_train(arg)
        trains = trains*s + tta2
        if index < 0:
            for i in range(1, len(trains)):
                displayed_times = np.concatenate([displayed_times, trains[i]])
        else:
            displayed_times = np.concatenate([displayed_times, trains[index]])

    lst = break_by_event(displayed_times, grouped_times, grouped_labels, before_event, after_event)

    bins = bins_btw
    cols = len(lst)

    subplots = []
    plt.figure(1)
    yu_lim = 0
    yd_lim = len(max(lst, key=len))
    for i in range(cols):
        subplots.append(plt.subplot(1, cols, i+1))
        n, bbb, ppp = plt.hist(lst[i], bins)
        yu_lim = max(max(n), yu_lim)
        yd_lim = min(min(n), yd_lim)
        tr = ""+str(grouped_labels[i])+"\n"+str(len(grouped_times[i]))+" time" + \
             ("" if len(grouped_times[i]) == 1 else "s")
        plt.title(tr)
        if grouped_labels[i] in baseline:
            col = 'k'
        else:
            col = 'r'
        plt.axvline(x=0, color=col)

    for subs in subplots:
        subs.set_ylim([max(yd_lim - 100, 0), yu_lim+100])
        if subs!=subplots[0]:
            subs.set_yticklabels([])

    plt.show()


fname = r'E:\sniffer_data_spring_2019\ParameterTest_OE2_022219_odors.smr'
name_list = [r'E:\sniffer_data_spring_2019\ParameterTest_OE2_022119_odors.smr',
             r'E:\sniffer_data_spring_2019\ParameterTest_OE1_022119_odors.smr']
# (spikes_events_histogram(fname, bins_btw=30))
# averaged_spike_events_histogram(name_list, bins_btw=40, subset=True)
# print(dp.get_spike_train(fname))
# grouped_spikes_events_histogram(fname, bins_btw=30, skip_last_event=True)
# grouped_spikes_events_histogram(name_list[0], name_list[1], bins_btw=30, skip_last_event=True)
