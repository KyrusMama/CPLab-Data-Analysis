from neo.io.spike2io import Spike2IO
import matplotlib.pyplot as plt
import numpy as np
import power_analyses as pa

fname = r'E:\sniffer_data_spring_2019\ParameterTest_OE2_022219_odors.smr'

spike2_reader = Spike2IO(fname)

blocks = spike2_reader.read()
assert len(blocks) == 1
block, = blocks
print(block)

segments = block.segments
assert len(segments) >= 1
segment = segments[0]

for sig in segment.analogsignals:
    print(sig.annotations['channel_names'][0])

assert len(segment.analogsignals) >= 1
analog_signal = segment.analogsignals[6]
unitless_data = np.squeeze(analog_signal.as_array())
# plt.plot(analog_signal.times[:10000], unitless_data[:10000])

# pa.welch_psd(unitless_data, analog_signal.sampling_rate)

print(analog_signal.sampling_rate, analog_signal.t_start, analog_signal.t_stop, analog_signal.annotations, analog_signal.annotations['channel_names'][0])
# for signal in segment.analogsignals:
#     print(signal.annotations)



events = segment.events
assert len(events) >= 1
event = events[0]

# for event in segment.events:
#     print(event.annotations)
#     print(event.times)
#     print(event.labels)

#
# print("spike trains:")
#
# spiketrains = segment.spiketrains
# if len(spiketrains):
#     train = spiketrains[5]
#     print(len(train), train.annotations, train.times)
#     print(np.squeeze(train.as_array()))
#     # for train in spiketrains:
#     #     print(train.annotations)
#
#     plt.hist(np.squeeze(train.as_array()), 100)
#     xcoords = segment.events[0].times
#     for xc in xcoords:
#         plt.axvline(x=xc, color='r')
#     plt.show()
