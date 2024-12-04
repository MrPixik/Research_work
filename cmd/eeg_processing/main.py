import mne
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.use('TkAgg')

data_path = r'../../static/eeg_data/panina1.set'

# Importing data and cropping for
raw = mne.io.read_raw_eeglab(data_path, preload=True)
# raw.crop(tmax=60).load_data()
# fig1 = raw.plot(duration=30, remove_dc=False)
# plt.show()

# raw_filtered_low_freq_drifts = raw.copy().filter(l_freq=1.,h_freq=None)
# fig2 = raw_filtered_low_freq_drifts.plot(duration=30,remove_dc=False)
# fig2.suptitle("Filtered low frequency")


# raw.plot()
# plt.show()

spectrum = raw.compute_psd()
# spectrum.plot(average=False, picks="data", exclude="bads", amplitude=False)
# plt.show()


spectrum.plot_topomap()
plt.show()
# spectrum.plot_topo()
# plt.show()
