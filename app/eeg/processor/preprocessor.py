import mne
from mne.preprocessing import ICA
from app.eeg.lib.paths import *
from tqdm import tqdm
import warnings


def __preprocessing(path):

    # Importing eeg_data
    raw = mne.io.read_raw_eeglab(path, preload=True)
    # raw.crop(tmax=180).load_data()
    raw.load_data()

    # Freq filter
    raw_filtered = raw.filter(l_freq=16.0, h_freq=22.0)

    possible_eog_channels = [['E127', 'E126'], ['E24', 'E3'], ['E27', 'E2'], ['E23', 'E9'], ['E22', 'E14'], ['E26', 'E8']] 

    for eog_channel in possible_eog_channels:
        try:
            
            raw_temp = raw_filtered.copy()
            raw_temp.set_channel_types({eog_channel[0]: 'eog'})
            raw_temp.set_channel_types({eog_channel[1]: 'eog'})

            # ICA
            ica = ICA(n_components=20, random_state=97)

            with warnings.catch_warnings(record=True) as caught_warnings:
                warnings.simplefilter("always")
                ica.fit(raw_temp)

                if any("did not converge" in str(w.message) for w in caught_warnings):
                    print(f"[!] ICA не сошёлся на каналах {eog_channel}")
                    continue


            eog_indices, _ = ica.find_bads_eog(raw_temp, threshold=0.5)
            ica.exclude = eog_indices

            raw_clean = ica.apply(raw_temp)

            return raw_clean
    
        except Exception as e:
            print(f"Ошибка с каналами {eog_channel}: {e}")
            continue

    raise RuntimeError("ICA не сошёлся ни на одном из EOG-каналов.")


def print_all_data_time():
    for i in range(1, DATASETS_NUM+1):

        file_path = get_raw_data_path(AUD_PACKAGE_NAME, i, Path(__file__))
    
        raw = mne.io.read_raw_eeglab(file_path, preload=True)
        raw.load_data()
        duration = raw.times[-1] 
        print(f"Номер: {i:d} Длительность записи: {duration:.2f} секунд")
    for i in range(1, DATASETS_NUM+1):

        file_path = get_raw_data_path(AUD_PACKAGE_NAME, i, Path(__file__))
    
        raw = mne.io.read_raw_eeglab(file_path, preload=True)
        raw.load_data()
        duration = raw.times[-1] 
        print(f"Номер: {i:d} Длительность записи: {duration:.2f} секунд")



def preprocess_aud():
    mne.set_log_level('ERROR')
    for i in tqdm(range(1, DATASETS_NUM+1), desc="aud processor"):

        file_path = get_raw_data_path(AUD_PACKAGE_NAME, i, Path(__file__))

        data = __preprocessing(file_path)

        # Экспорт данных в формат .set
        save_path = get_preprocessed_data_path(AUD_PACKAGE_NAME, i, Path(__file__))
        data.save(save_path, overwrite=True)
        
        
def preprocess_aud_norm():
    mne.set_log_level('ERROR')
    for i in tqdm(range(1, DATASETS_NUM+1), desc="aud-norm processor"):

        file_path = get_raw_data_path(AUD_NORM_PACKAGE_NAME, i, Path(__file__))

        data = __preprocessing(file_path)

        save_path = get_preprocessed_data_path(AUD_NORM_PACKAGE_NAME, i, Path(__file__))
        data.save(save_path, overwrite=True)
