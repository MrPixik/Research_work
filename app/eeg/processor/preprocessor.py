import mne
from mne.preprocessing import ICA
from mne_icalabel import label_components
from app.eeg.lib.paths import *
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np


def __preprocessing(path):

    # Importing eeg_data
    raw = mne.io.read_raw_eeglab(path, preload=True)
    raw.load_data()
    
    # Low frequesy filter            
    raw = raw.filter(l_freq=1, h_freq=45.0)
    # raw_before = raw.copy()
    
    # raw.plot(duration=30, n_channels=30, scalings='auto', title='Сырые данные ЭЭГ (только низкочастотная частотная фильтрация)')
    
    # ICA
    ica = ICA(
    n_components=0.99,
    max_iter="auto",
    method="infomax",
    random_state=97,
    fit_params=dict(extended=True),
    )
    ica.fit(raw)
    
    # ica.plot_components()
    
    ic_labels = label_components(raw, ica, method="iclabel")
    
    # print(ic_labels)
    # print(ic_labels["labels"])
    # ica.plot_properties(raw, picks=[0, 12, 14], verbose=False)
    
    labels = ic_labels["labels"]
    probas = ic_labels['y_pred_proba']
    
    exclude_idx = [
        idx
        for idx, (lbl, p) in enumerate(zip(labels, probas))
        if (lbl not in ['brain', 'other']) and (p > 0.5)
    ]
    # print(f"Excluding these ICA components: {exclude_idx}")
    
    
    
    ica.apply(raw, exclude=exclude_idx)
    # ica.plot_overlay(raw, exclude=exclude_idx)
    
    # raw.plot(duration=30, n_channels=30, scalings='auto', title='Очищенные данные EEG')
    # raw_before.compute_psd(fmin=1, fmax=40).plot(average=True, picks='eeg')
    # raw.compute_psd(fmin=1, fmax=40).plot(average=True, picks='eeg')
    # input()
    
    # raw.compute_psd().plot_topomap()

    return raw


def print_all_data_time():
    for i in range(1, DATASETS_NUM+1):

        file_path = get_raw_data_path(AUD_PKGNAME, i, Path(__file__))
    
        raw = mne.io.read_raw_eeglab(file_path, preload=True)
        raw.load_data()
        duration = raw.times[-1] 
        print(f"Номер: {i:d} Длительность записи: {duration:.2f} секунд")
    for i in range(1, DATASETS_NUM+1):

        file_path = get_raw_data_path(AUD_PKGNAME, i, Path(__file__))
    
        raw = mne.io.read_raw_eeglab(file_path, preload=True)
        raw.load_data()
        duration = raw.times[-1] 
        print(f"Номер: {i:d} Длительность записи: {duration:.2f} секунд")



def preprocess_aud():
    mne.set_log_level('ERROR')
    for i in tqdm(range(1, DATASETS_NUM+1), desc="aud processor"):

        file_path = get_raw_data_path(AUD_PKGNAME, i, Path(__file__))

        data = __preprocessing(file_path)

        # Экспорт данных в формат .set
        save_path = get_preprocessed_data_path(AUD_PKGNAME, i, Path(__file__))
        data.save(save_path, overwrite=True)
        
def preprocess_aud_norm():
    mne.set_log_level('ERROR')
    for i in tqdm(range(1, DATASETS_NUM+1), desc="aud-norm processor"):

        file_path = get_raw_data_path(AUD_NORM_PKGNAME, i, Path(__file__))

        data = __preprocessing(file_path)

        save_path = get_preprocessed_data_path(AUD_NORM_PKGNAME, i, Path(__file__))
        data.save(save_path, overwrite=True)