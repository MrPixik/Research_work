import mne
from mne.preprocessing import ICA
from app.eeg.lib.paths import *
from tqdm import tqdm


def __preprocessing(path):

    # Importing eeg_data
    raw = mne.io.read_raw_eeglab(path, preload=True)
    raw.load_data()

    # Freq filter
    raw_filtered_freq = raw.filter(l_freq=16.0, h_freq=22.0)

    raw_filtered_freq.set_channel_types({'E127': 'eog'})

    # ICA filter
    ica = ICA(n_components=20, random_state=97)
    ica.fit(raw_filtered_freq)

    eog_indices, _ = ica.find_bads_eog(raw_filtered_freq, threshold=0.5)
    ica.exclude = eog_indices

    raw_clean = ica.apply(raw_filtered_freq)

    return raw_clean


def preprocess_aud():
    mne.set_log_level('ERROR')
    for i in tqdm(range(1, DATASETS_NUM), desc="aud processor"):

        file_path = get_raw_data_path(AUD_PACKAGE_NAME, i, Path(__file__))

        data = __preprocessing(file_path)

        # Экспорт данных в формат .set
        save_path = get_preprocessed_data_path(AUD_PACKAGE_NAME, i, Path(__file__))
        data.save(save_path, overwrite=True)
