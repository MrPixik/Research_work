import mne
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')

data_path = r'../../static/eeg_data/panina1.set'

#Importing data and cropping for
raw = mne.io.read_raw_eeglab(data_path, preload=True)
raw.crop(tmax=60).load_data()

ssp_projectors = raw.info["projs"]
raw.del_proj()