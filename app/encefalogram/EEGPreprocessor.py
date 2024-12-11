import mne

data_path = r'../../static/eeg_data/raw/panina1.set'
output_path = r'../../static/eeg_data/preprocessed/preprocessed_data_eeg.fif'


class EEGPreprocessor:
    @staticmethod
    def preprocessing():
        # Importing data
        raw = mne.io.read_raw_eeglab(data_path, preload=True)
        raw.load_data()

        # Low freq filter
        raw_preprocessed = raw.copy().filter(l_freq=1., h_freq=None)

        # Notch filter
        freqs = [25.07]
        raw_preprocessed.notch_filter(freqs=freqs, picks='all', method='fir')

        return raw_preprocessed

    @staticmethod
    def preprocessing_save():
        data = EEGPreprocessor.preprocessing()
        # Экспорт данных в формат .set
        data.save(output_path, overwrite=True)
