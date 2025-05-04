import mne
from mne.preprocessing import ICA
from mne.datasets import fetch_fsaverage
from app.eeg.lib.paths import *
from app.eeg.lib.eeg_channels import *
from tqdm import tqdm
import numpy as np
import os


def reverce_problem_for_inactive_zones(eeg_data, n_components=30, method='dSPM', snr=3.0, recalculate=False):
    """
    Выполняет ICA и обратную задачу для предобработанных raw-данных,
    усредняя результат по всему временному интервалу. При этом автоматически
    загружает стандартную модель fsaverage.

    Параметры:
    -----------
    n_components : int
        Количество ICA-компонент для извлечения (по умолчанию 15).
    method : str
        Метод обратного решения ('MNE', 'dSPM', 'sLORETA').
    snr : float
        Отношение сигнал/шум, влияет на параметр регуляризации.

    Возвращает:
    ------------
    stcs_avg : list of mne.SourceEstimate
        Список статических SourceEstimate (усредненная активность)
        по каждой ICA-компоненте.
    """
    # 0. Загружаем fsaverage и определяем subjects_dir правильно
    fs_dir = Path(fetch_fsaverage(verbose=False))
    subjects_dir = str(fs_dir.parent)  # убираем дублирование 'fsaverage'
    subject = 'fsaverage'

    # 2. ICA
    ica = mne.preprocessing.ICA(
        n_components=n_components, random_state=97, max_iter='auto'
    )
    ica.fit(eeg_data)
    # ica.plot_components(show=True)
    
    # Получаем компоненты ICA в виде mne.Raw, а затем извлекаем из них данные в виде (n_components, n_times)
    sources = ica.get_sources(eeg_data).get_data()

    # 3. Монтаж
    # eeg_data.set_montage(eeg_data.get_montage())

    if recalculate:
        recalculate_constants(
            subject_dir=subjects_dir,
            subject=subject,
            eeg_data_info=eeg_data.info
        )
    
    # bem = mne.read_bem_solution(get_bem_solution_path(Path(__file__)))
    # src = mne.read_source_spaces(get_source_spaces_path(Path(__file__)))
    # fwd = mne.read_forward_solution(get_forward_solution_path(Path(__file__)))
    inv = mne.minimum_norm.read_inverse_operator(get_inverse_operator_path(Path(__file__)))
    
    lambda2 = 1.0 / (snr ** 2)

    # 6. Обратная задача и усреднение для каждой ICA-компоненты
    stcs_avg = []
    for idx in tqdm(range(n_components), desc="reverse problem solution"):
        
        # 6.1. Канальный сигнал компоненты
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

        # 6.4. Усредняем по времени и создаём статический SourceEstimate
        # data_mean = stc.data.mean(axis=1)  # усреднённая по времени активность
        # mean = np.mean(data_mean)
        # std = np.std(data_mean)
        # z_score = (data_mean - mean) / std

        # # Инвертируем так, чтобы меньшие значения z_score (т.е. "менее активные") стали более значимыми
        # z_score_inv = -z_score

        # # Optionally: можно отсечь или выделить только 10% самых неактивных
        # # threshold = np.percentile(z_score_inv, 90)
        # # z_score_inv[z_score_inv < threshold] = 0

        # stc_avg = mne.SourceEstimate(
        #     z_score_inv[:, None], vertices=stc.vertices,
        #     tmin=0.0, tstep=1.0, subject=subject
        # )
        data_mean = stc.data.mean(axis=1)
        
        
        # инверсия
        data_mean = stc.data.mean(axis=1)       # shape (n_vertices,)
        max_val = data_mean.max()
        data_inv = max_val - data_mean
        
        stc_avg = mne.SourceEstimate(
            data_inv[:, None], vertices=stc.vertices,
            tmin=0.0, tstep=1.0, subject=subject
        )
        stcs_avg.append(stc_avg)

    return stcs_avg


def recalculate_constants(subjects_dir, subject, eeg_data_info):
    # Создание БЕМ модели
    bem_model = mne.make_bem_model(
        subject=subject, ico=4,
        conductivity=(0.3, 0.006, 0.3),
        subjects_dir=subjects_dir
    )
    bem = mne.make_bem_solution(
        bem_model,
        verbose=False
    )
    # Создание источникового пространства
    src = mne.setup_source_space(
        subject,
        spacing='oct4',
        subjects_dir=subjects_dir,
        n_jobs=12,
        verbose=False
    )
    
    fwd = mne.make_forward_solution(
        eeg_data_info,
        trans='fsaverage',
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

def reverse_problem_for_band(preprocessed_data, band_name):
    
    
    if band_name in BANDS.keys():
        l_freq, h_freq = BANDS[band_name]
        preprocessed_data = preprocessed_data.filter(l_freq=l_freq, h_freq=h_freq)
        preprocessed_data.set_eeg_reference('average', projection=True)
        
        return reverce_problem_for_inactive_zones(preprocessed_data)
    
    
def reverse_problem_for_all_bands(id):
    """
    Делает расчет обратной задачи для всех каналов для обоих экспериментов aud и aud-norm.
    После этого сохраняет его в static/reverse_solution/{id}/{experiment_type}/{band}

    Параметры:
    -----------
    id : int (1-28)
        номер эксперимента
    """
    #Отключаем логирование
    mne.set_log_level('WARNING')
    
    os.environ['PYVISTA_OFF_SCREEN'] = 'true'
    mne.viz.set_3d_backend('pyvistaqt')
    
    experiments = [AUD_PKGNAME, AUD_NORM_PKGNAME]
    
    for experiment in experiments:
        
        # 1. Загрузка предобработанных данных
        data_path = get_preprocessed_data_path(
            experiment, id, Path(__file__)
        )
        eeg_data = mne.io.read_raw_fif(data_path, preload=True)
        eeg_data.load_data()
        
        # 2. Образание по частотам
        for band in tqdm(BANDS, desc=experiment + " " + str(id)):
            stcs_avg = reverse_problem_for_band(eeg_data, band)
            
            subject='fsaverage'
            fs_dir = Path(fetch_fsaverage(verbose=False))
            subjects_dir = str(fs_dir.parent)
            
            for component_id, stc_avg in enumerate(stcs_avg):
                brain = mne.viz.plot_source_estimates(
                    stc_avg, 
                    subject=subject, 
                    subjects_dir=subjects_dir,
                    hemi='both',
                    surface='white',
                    background='white',
                    cortex='low_contrast',
                    time_viewer=False,
                    show_traces=False,
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