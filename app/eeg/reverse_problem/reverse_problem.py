import mne
from mne.preprocessing import ICA
from mne.datasets import fetch_fsaverage
from mne.minimum_norm import apply_inverse_epochs
from mne import read_labels_from_annot
from app.eeg.lib.paths import *
from app.eeg.lib.eeg_channels import *
import numpy as np
import pandas as pd
from tqdm import tqdm
import os


SUBJECT = 'fsaverage'



def reverse_problem_inactive_brodmann(eeg_data,
                                     n_components=0.9999,
                                     method='sLORETA',
                                     snr=3.0,
                                     recalculate=False,
                                     inactive_percentile=10.0,
                                     verbose=False):
    """
    Разбивает EEG-данные ICA-компоненты.
    Для каждой из компонент решает обратную задачу и возвращает
    список полей Бродмана с наименьшей активностью.
    """

    # ICA
    ica = mne.preprocessing.ICA(n_components=n_components,
                                random_state=97,
                                max_iter='auto')
    ica.fit(eeg_data)
    sources = ica.get_sources(eeg_data).get_data()


    #Загрузка обратного оператора для решения обратной задачи
    if recalculate:
        __recalculate_constants(eeg_data_info=eeg_data.info)

    inv = mne.minimum_norm.read_inverse_operator(get_inverse_operator_path(Path(__file__)))
    lambda2 = 1.0 / (snr ** 2)

    # Загрузка меток полей Бродмана
    labels_all = mne.read_labels_from_annot(
        subject='fsaverage',
        parc='PALS_B12_Brodmann',
        hemi='both'
    )
    labels_brodmann = [lbl for lbl in labels_all if lbl.name.startswith('Brodmann.')]



    inactive_labels_per_component = []

    #Цикл по компонентам
    for idx in tqdm(range(ica.n_components_), desc="Brodmann classification"):
        # Создание временного ряда из компоненты (тот же формат, в котором записана egg_data)
        topo = ica.get_components()[:, idx][:, None]
        info_one = mne.create_info(ica.ch_names,
                                   eeg_data.info['sfreq'],
                                   ch_types='eeg')
        raw_comp = mne.io.RawArray(topo @ sources[idx][None, :], info_one)
        raw_comp.set_montage(eeg_data.get_montage())
        raw_comp.set_eeg_reference('average', projection=True)

        # Выделение эпохи из временного ряда (этот формат нужен для решения обратной задачи в будущем)
        events = mne.make_fixed_length_events(raw_comp,
                                              duration=raw_comp.times[-1])
        epochs = mne.Epochs(raw_comp,
                            events,
                            tmin=0,
                            tmax=raw_comp.times[-1],
                            baseline=None,
                            preload=True)

        # Решение обратной задачи и уреднение за весь промежуток времени
        stc = apply_inverse_epochs(epochs, inv, lambda2, method)[0]
        data_mean = stc.data.mean(axis=1)
        n_lh = len(stc.vertices[0])

        # Вычисление средней активации для каждого поля Бродмана
        label_acts = {}
        all_verts = np.hstack(stc.vertices)
        for lbl in labels_brodmann:
            verts = np.intersect1d(lbl.vertices, all_verts)
            if verts.size == 0:
                continue
            idxs = []
            for v in verts:
                if v in stc.vertices[0]:
                    pos = np.where(stc.vertices[0] == v)[0][0]
                else:
                    pos = n_lh + np.where(stc.vertices[1] == v)[0][0]
                idxs.append(pos)
            label_acts[lbl.name] = data_mean[idxs].mean()
            
            
        if verbose:
            print(f"\nКомпонента {idx}:")
            print("----------------------------")
            print("{:<20} {:<10}".format("Область Бродмана", "Активация"))
            print("----------------------------")
            for area, value in sorted(label_acts.items(), key=lambda x: x[1]):
                print("{:<20} {:<10.4f}".format(area, value))
            print("----------------------------")
            

        # Определяем порог по неактивности и отбираем метки
        values = np.array(list(label_acts.values()))
        thresh = np.percentile(values, inactive_percentile)
        inactive = [name for name, val in label_acts.items() if val <= thresh]
        inactive_labels_per_component.append(inactive)
        
        if verbose:
            print(f"Неактивные области (нижние {inactive_percentile}%): {inactive}")
            print("==============================================")


    return inactive_labels_per_component



def brodman_for_band(eeg_data, band_name):
    """
    Обрезает предобработанные данные по частотным диапазонам из lib.eeg_channels.BANDS.
    Возвращает результаты из reverse_problem_inactive_brodmann для каждого из них.
    """
    l_freq, h_freq = BANDS[band_name]
    preprocessed_data = eeg_data.filter(l_freq=l_freq, h_freq=h_freq)
    
    
    
    return reverse_problem_inactive_brodmann(preprocessed_data)

def brodman_for_id(id):
    """
    Создает цикл по частотным диапазонам из lib.eeg_channels.BANDS.
    Сохраняет полученные результаты в .xlsx файлы.
    """
    experiments = [AUD_PKGNAME, AUD_NORM_PKGNAME]
    
    for experiment in experiments:
        
        # 1. Загрузка предобработанных данных
        data_path = get_preprocessed_data_path(
            experiment, id, Path(__file__)
        )
        eeg_data = mne.io.read_raw_fif(data_path, preload=True)
        eeg_data.load_data()
        
        # Массив с названиями всех полей Бродманна
        labels_all = read_labels_from_annot(
            subject='fsaverage',
            parc='PALS_B12_Brodmann',
            hemi='both',
            sort=True
        )
        
        brodmann_labels = [lbl.name for lbl in labels_all
                        if lbl.name.startswith('Brodmann.')]
        # 2. Цикл по частотным диапазонам
        for band in tqdm(BANDS, desc=experiment + " " + str(id)):
            
            
            res = brodman_for_band(eeg_data, band)
            
            
            # Создаем словарь-счётчик
            brodmann_count = {label: 0 for label in brodmann_labels}

            # Подсчитываем количество вхождений каждой метки в res
            for component in res:
                for label in component:
                    if label in brodmann_count:
                        brodmann_count[label] += 1

            
            # Преобразуем словарь в DataFrame
            df = pd.DataFrame(list(brodmann_count.items()), columns=['Brodmann Area', 'Count'])
            df = df.sort_values(by='Count', ascending=False)
            
            # Сохраняем в Excel
            output_path = get_brodmann_data_path(Path(__file__), id, experiment, band)
            
            # Создаем все директории, если они не существуют
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            df.to_excel(output_path, index=False)

def brodmann():
    """
    Основная функция для решения задачи локализации неактивных областей по полям Бродмана.
    Работает по вложенной цепочке семейства функций.
    brodman()
    brodman_for_id()
    brodman_for_band()
    reverse_problem_inactive_brodmann()
    """
  
    mne.set_log_level('WARNING')
    

    fs_dir = Path(fetch_fsaverage(verbose=False))
    subjects_dir = str(fs_dir.parent)
    mne.utils.set_config('SUBJECTS_DIR', subjects_dir, set_env=True)
    
    for i in range(1,DATASETS_NUM + 1):
        brodman_for_id(i)

def __recalculate_constants(eeg_data_info):
    # Создание БЕМ модели
    bem_model = mne.make_bem_model(
        subject=SUBJECT, 
        ico=4,
        conductivity=(0.3, 0.006, 0.3),
    )
    bem = mne.make_bem_solution(
        bem_model,
        verbose=False
    )
    # Создание источникового пространства
    src = mne.setup_source_space(
        SUBJECT,
        spacing='oct4',
        n_jobs=12,
        verbose=False
    )
    
    fwd = mne.make_forward_solution(
        eeg_data_info,
        trans=SUBJECT,
        src=src,
        bem=bem,
        eeg=True,
        meg=False,
        mindist=5.0,
        n_jobs=12
    )
    
    mne.write_bem_solution('fsaverage-bem.fif', bem)
    mne.write_source_spaces('fsaverage-src.fif', src)
    mne.write_forward_solution('fsaverage-fwd.fif', fwd)

    # 5. Шумовая ковариация и inverse-оператор  
    cov = mne.make_ad_hoc_cov(
        eeg_data_info,
        verbose=False
    )
    inv = mne.minimum_norm.make_inverse_operator(
        eeg_data_info, fwd, cov, loose=0.2,
        verbose=False
    )
    mne.minimum_norm.write_inverse_operator('fsaverage-inv.fif', inv)

"""DEPRECIATED"""
def __reverce_problem_for_inactive_zones(eeg_data, n_components=30, method='sLORETA', snr=3.0, recalculate=False):

    # ICA
    ica = mne.preprocessing.ICA(
        n_components=n_components, random_state=97, max_iter='auto'
    )
    ica.fit(eeg_data)
    
    # Получаем компоненты ICA в виде mne.Raw, а затем извлекаем из них данные в виде (n_components, n_times)
    sources = ica.get_sources(eeg_data).get_data()

    if recalculate:
        __recalculate_constants(
            eeg_data_info=eeg_data.info
        )
    
    inv = mne.minimum_norm.read_inverse_operator(get_inverse_operator_path(Path(__file__)))
    
    lambda2 = 1.0 / (snr ** 2)

    # Обратная задача и усреднение для каждой ICA-компоненты
    stcs_avg = []
    for idx in tqdm(range(n_components), desc="reverse problem solution"):
        
        # Канальный сигнал компоненты
        topo = ica.get_components()[:, idx][:, None]
        # Создаём Info-объект — метаинформацию о данных
        info_one = mne.create_info(
            ica.ch_names,
            eeg_data.info['sfreq'],
            ch_types='eeg'
            )
        raw_comp = mne.io.RawArray(topo @ sources[idx][None, :], info_one)
        raw_comp.set_montage(eeg_data.get_montage())
        raw_comp.set_eeg_reference('average', projection=True)

        # 6.2. Epochs для inverse
        events = mne.make_fixed_length_events(
            raw_comp, duration=raw_comp.times[-1]
        )
        epochs = mne.Epochs(
            raw_comp, events, tmin=0, tmax=raw_comp.times[-1],
            baseline=None, preload=True
        )

        # 6.3. Решаем обратную задачу
        stc = mne.minimum_norm.apply_inverse_epochs(
            epochs, inv, lambda2, method
        )[0]

        data_mean = stc.data.mean(axis=1)
        
        
        stc_avg = mne.SourceEstimate(
            data_mean[:, None], vertices=stc.vertices,
            tmin=0.0, tstep=1.0, subject=SUBJECT
        )
        stcs_avg.append(stc_avg)

    return stcs_avg

def __reverse_problem_for_band(preprocessed_data, band_name):
    
    l_freq, h_freq = BANDS[band_name]
    preprocessed_data = preprocessed_data.filter(l_freq=l_freq, h_freq=h_freq)
    
    preprocessed_data.set_eeg_reference('average', projection=True)
    
    
    
    return __reverce_problem_for_inactive_zones(preprocessed_data)

def __reverse_problem_for_id(id):
    """
    Делает расчет обратной задачи для всех каналов для обоих экспериментов aud и aud-norm.
    После этого сохраняет его в static/reverse_solution/{id}/{experiment_type}/{band}
    """
    
    experiments = [AUD_PKGNAME, AUD_NORM_PKGNAME]
    
    for experiment in experiments:
        
        # Загрузка предобработанных данных
        data_path = get_preprocessed_data_path(
            experiment, id, Path(__file__)
        )
        eeg_data = mne.io.read_raw_fif(data_path, preload=True)
        eeg_data.load_data()
        
        
        # Цикл по частотным диапазонам
        for band in tqdm(BANDS, desc=experiment + " " + str(id)):
            
            stcs_avg = __reverse_problem_for_band(eeg_data, band)
            
            for component_id, stc_avg in enumerate(stcs_avg):
                brain = mne.viz.plot_source_estimates(
                    stc_avg, 
                    subject=SUBJECT,
                    hemi='both',
                    surface='white',
                    background='white',
                    cortex='low_contrast',
                    time_viewer=False,
                    show_traces=False,
                    verbose=False
                )

                # Сохраняем изображение сразу в файл
                save_path = get_solution_data_path_by_band(
                    Path(__file__),
                    id,
                    experiment,
                    band,
                    component_id
                )
                
                # Создаем все директории, если они не существуют
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                
                brain.save_image(save_path)
                
                brain.close()

def reverse_problem():
    mne.set_log_level('WARNING')
    

    fs_dir = Path(fetch_fsaverage(verbose=False))
    subjects_dir = str(fs_dir.parent)
    mne.utils.set_config('SUBJECTS_DIR', subjects_dir, set_env=True)
    
    for i in range(1,DATASETS_NUM + 1):
        __reverse_problem_for_id(i)
