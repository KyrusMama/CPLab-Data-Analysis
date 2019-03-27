from neo.io.spike2io import Spike2IO
import numpy as np
#Written by Kyrus Mama

def get_data(file_name, data_name, id_no=-1, simplified=True):
    """
    returns the first instance of a signal identified by data_name from the file file_name.

    Parameters
    ----------
    file_name: The name of the file from which you wish to retrieve the data. Must include the path.
        eg: r'E:\sniffer_data_spring_2019\ParameterTest_OE2_022219_odors.smr'
    data_name: The name of the signal which you wish to retrieve
    simplified: if False then returns the entire neo signal

    Returns
    -------
    data, fr:
    data: A numpy array of all the data in the desired signal
        if data_name does not exit in the file returns []
    fr: The sampling rate of the desired data, in Hz.
        if data_name does not exit in the file returns 0

    Throws
    -------
    FileNotFoundError: No such file or directory file_name if file_name does not exit.
    """
    spike2_reader = Spike2IO(file_name)

    blocks = spike2_reader.read()
    assert len(blocks) == 1
    block, = blocks

    segments = block.segments
    assert len(segments) == 1
    segment = segments[0]

    for sig in segment.analogsignals:
        if sig.annotations['channel_names'][0] == data_name or sig.annotations['channel_ids'][0] == id_no - 1:
            prnt = "channel : " + sig.annotations['channel_names'][0] + " with id no : " + \
                  str(sig.annotations['channel_ids'][0]+1) + " found"
            print(prnt)
            if simplified:
                return np.squeeze(sig.as_array()), sig.sampling_rate
            else:
                return sig
    return [], 0


def get_spike_train(file_name, simplified=True):
    """
    returns a list of all the spike-train times from the file file_name.

    Parameters
    ----------
    file_name: The name of the file from which you wish to retrieve the data. Must include the path.
        eg: r'E:\sniffer_data_spring_2019\ParameterTest_OE2_022219_odors.smr'
    simplified: if False then returns the entire neo spike train

    Returns
    -------
    trains: A list in which each element contains a list of spike times (in seconds) for a type of spike.

    Throws
    -------
    FileNotFoundError: No such file or directory file_name if file_name does not exit.
    """
    spike2_reader = Spike2IO(file_name)

    blocks = spike2_reader.read()
    assert len(blocks) == 1
    block, = blocks

    segments = block.segments
    assert len(segments) == 1
    segment = segments[0]

    trains = []
    spike_trains = segment.spiketrains
    if not simplified:
        return spike_trains
    for i in range(len(spike_trains)):
        train = spike_trains[i]
        trains.append(np.squeeze(train.as_array()))
    return trains


def get_events(file_name):
    """
    returns the first list of events in file file_name, along with the times at which they occur.

    Parameters
    ----------
    file_name: The name of the file from which you wish to retrieve the data. Must include the path.
        eg: r'E:\sniffer_data_spring_2019\ParameterTest_OE2_022219_odors.smr'

    Returns
    -------
    times, labels:
    times: a list of the times at which each event occurred (in seconds)
    labels: a list of the labels allocated to each event

    Throws
    -------
    FileNotFoundError: No such file or directory file_name if file_name does not exit.
    """
    spike2_reader = Spike2IO(file_name)

    blocks = spike2_reader.read()
    assert len(blocks) == 1
    block, = blocks

    segments = block.segments
    assert len(segments) == 1
    segment = segments[0]

    events = segment.events
    assert len(events) >= 1
    event = segment.events[0]
    return event.times, event.labels

