import mne
from mne.preprocessing import ICA
from mne_icalabel import label_components
from app.eeg.lib.paths import *
from tqdm import tqdm
import warnings
import matplotlib.pyplot as plt


# def __preprocessing(path):

#     # Importing eeg_data
#     raw = mne.io.read_raw_eeglab(path, preload=True)
#     raw.load_data()


#     possible_eog_channels = [['E127', 'E126'], ['E24', 'E3'], ['E27', 'E2'], ['E23', 'E9'], ['E22', 'E14'], ['E26', 'E8']] 

#     for eog_channel in possible_eog_channels:
#         try:
            
#             raw_temp = raw.copy()
#             raw_temp.set_channel_types({eog_channel[0]: 'eog'})
#             raw_temp.set_channel_types({eog_channel[1]: 'eog'})
            
#             # Low frequesy filter            
#             raw_temp_low_freq_filtred = raw_temp.filter(l_freq=1, h_freq=40.0)

#             # ICA
#             ica = ICA(n_components=20, random_state=97)

#             with warnings.catch_warnings(record=True) as caught_warnings:
#                 warnings.simplefilter("always")
#                 ica.fit(raw_temp_low_freq_filtred)

#                 if any("did not converge" in str(w.message) for w in caught_warnings):
#                     print(f"[!] ICA не сошёлся на каналах {eog_channel}")
#                     continue


#             eog_indices, _ = ica.find_bads_eog(raw_temp_low_freq_filtred, threshold=2)
#             ica.exclude = eog_indices

#             # # 1. Сколько компонентов исключено:
#             # n_excluded = len(ica.exclude)
#             # print(f"Исключено компонентов: {n_excluded}")

#             # # 2. Общее число компонентов в модели ICA:
#             # n_total = ica.n_components_
#             # print(f"Всего компонентов в модели: {n_total}")

#             # # 3. Сколько компонентов осталось:
#             # n_remaining = n_total - n_excluded
#             # print(f"Осталось компонентов: {n_remaining}")
            
#             raw_ica_accepted = ica.apply(raw_temp_low_freq_filtred)
            
#             raw_ica_accepted.set_eeg_reference(ref_channels='average')
            
#             raw_ica_accepted.plot(duration=10, n_channels=30, scalings='auto', title='Очищенные данные EEG')
#             input("Нажмите Enter, чтобы закрыть окно и выйти")

#             return raw_ica_accepted
    
#         except Exception as e:
#             print(f"Ошибка с каналами {eog_channel}: {e}")
#             continue

#     raise RuntimeError("ICA не сошёлся ни на одном из EOG-каналов.")

# def __preprocessing(path):

#     # Importing eeg_data
#     raw = mne.io.read_raw_eeglab(path, preload=True)
#     raw.load_data()

#     # Low frequesy filter            
#     raw = raw.filter(l_freq=1, h_freq=45.0)

#     # ICA
#     ica = ICA(n_components=20, random_state=97)
#     ica.fit(raw)
    
#     component_dict  = label_components(raw, ica, method='iclabel')
#        # Берём массив вероятностей «eye blink»
#     proba = component_dict['y_pred_proba']
    


#     # Составляем список индексов, где вероятность > 0.5
#     eye_idx = [i for i, p in enumerate(proba) if p > 0.85]
#     ica.exclude = eye_idx

#     # 1) Глобальный обзор
#     # ica.plot_components(inst=raw)

#     # 2) Только помеченные артефакты
#     ica.plot_components(picks=ica.exclude, inst=raw)

#     # 4) Наложение исходного/очищенного
#     # ica.plot_overlay(inst=raw, exclude=ica.exclude)

#     raw_clean = ica.apply(raw)

#     # # 1. Сколько компонентов исключено:
#     # n_excluded = len(ica.exclude)
#     # print(f"Исключено компонентов: {n_excluded}")

#     # # 2. Общее число компонентов в модели ICA:
#     # n_total = ica.n_components_
#     # print(f"Всего компонентов в модели: {n_total}")

#     # # 3. Сколько компонентов осталось:
#     # n_remaining = n_total - n_excluded
#     # print(f"Осталось компонентов: {n_remaining}")
#     # raw_clean.plot(duration=10, n_channels=30, scalings='auto', title='Очищенные данные EEG')

#     return raw_clean


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
        if (lbl not in ['brain', 'other']) #and (p > 0.5)
    ]
    # print(f"Excluding these ICA components: {exclude_idx}")
    
    
    
    ica.apply(raw, exclude=exclude_idx)
    # ica.plot_overlay(raw, exclude=exclude_idx)
    
    # raw.plot(duration=30, n_channels=30, scalings='auto', title='Очищенные данные EEG')
    # raw_before.compute_psd(fmin=1, fmax=40).plot(average=True, picks='eeg')
    # raw.compute_psd(fmin=1, fmax=40).plot(average=True, picks='eeg')
    # input()

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
    # for i in range(1, DATASETS_NUM+1):

        file_path = get_raw_data_path(AUD_PKGNAME, i, Path(__file__))

        data = __preprocessing(file_path)

        # Экспорт данных в формат .set
        save_path = get_preprocessed_data_path(AUD_PKGNAME, i, Path(__file__))
        data.save(save_path, overwrite=True)
        
def preprocess_aud_norm():
    mne.set_log_level('ERROR')
    for i in tqdm(range(1, DATASETS_NUM+1), desc="aud-norm processor"):
    # for i in range(1, DATASETS_NUM+1):

        file_path = get_raw_data_path(AUD_NORM_PKGNAME, i, Path(__file__))

        data = __preprocessing(file_path)

        save_path = get_preprocessed_data_path(AUD_NORM_PKGNAME, i, Path(__file__))
        data.save(save_path, overwrite=True)