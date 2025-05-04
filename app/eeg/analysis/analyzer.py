import mne
import matplotlib
import matplotlib.pyplot as plt
from app.eeg.lib.eeg_channels import *
from app.eeg.lib.paths import *
from tqdm import tqdm
import numpy as np


def _select_bands(selected_band_name):
    # Создание нового словаря с выбранным каналом
    return {selected_band_name: BANDS[selected_band_name]}
def _load_data():
    path = get_preprocessed_data_path(AUD_PKGNAME, 1, Path(__file__))
    eeg_data = mne.io.read_raw(path, preload=True)
    eeg_data.load_data()
    return eeg_data

def _get_glob_minmax(spectrum, freq_lims):

    # Извлечение данных PSD и частот
    psds = spectrum.get_data()  # Массив формы (n_channels, n_freqs)
    freqs = spectrum.freqs

    # Диапазон допустимых частот
    fmin, fmax = freq_lims

    # Индексы частот в целевом диапазоне
    idx_band = np.logical_and(freqs >= fmin, freqs <= fmax)

    # Выбор данных PSD для диапазона
    psds_band = psds[:, idx_band]

    # Расчет глобальных min и max
    psds_band_uv = psds_band * 1e12

    vmin = np.percentile(psds_band_uv, 0) 
    vmax = np.percentile(psds_band_uv, 100)

    return (vmin,vmax)

def _extract_psd_for_segment(spectrum_segment, freq_lims):
    # Извлекаем данные PSD для текущего сегмента
    psds = spectrum_segment.get_data()  # Массив формы (n_channels, n_freqs)
    freqs = spectrum_segment.freqs

    # Фильтруем данные по частотному диапазону
    fmin, fmax = freq_lims
    idx_band = np.logical_and(freqs >= fmin, freqs <= fmax)
    psds_band = psds[:, idx_band]

    # Конвертируем в μV²/Hz
    psds_band_uv = psds_band * 1e12

    return psds_band_uv

def _normalize_psd(psds_band_uv, vlim, threshold=1):
    vmin,vmax = vlim
    psds_band_norm = (psds_band_uv - vmin) / (vmax - vmin)
    psds_band_norm = np.clip(psds_band_norm, 0, 1)
    psds_band_mean = np.mean(psds_band_norm, axis=1)
    psds_band_mean_binary = np.where(psds_band_mean >= threshold, 1, 0)
    return psds_band_mean_binary

def _plot_topomap(fig, axes, raw_info, psd_data, vlim, colorbar=False):

    im, _ = mne.viz.plot_topomap(
        psd_data,
        raw_info,
        vlim=vlim,
        axes=axes,
        # image_interp='nearest',
        show=False
    )
    if colorbar:
        cbar = fig.colorbar(im, ax=axes)
        cbar.set_label('Power (μV²/Hz)')
    return fig

def save_topomap_by_time(title, freq_lims, update_interval=1, threshold=1, normalize=False):
    """
    Сохраняет топографические карты по времени с возможностью выбрать частотный диапазон и интервал обновления.

    :param title: название графика
    :param freq_lims: частостный интервал
    :param update_interval: интервал времени в секундах для обновления карты
    :param threshold: пороговое значение для отображения топографической карты [0,1]
    :param normalize: нормировать данные
    """
    mne.set_log_level('ERROR')
    
    data = _load_data()

    # Словарь с названием + частотный диапазон
    band_dict = {title: freq_lims}

    # Вычисляем значения глобальных MIN и MAX
    spectrum = data.compute_psd()
    vlim = _get_glob_minmax(spectrum, freq_lims)

    duration = int(data.times[-1])
    output_dir = get_topomap_imgs_path(title, Path(__file__))
    # Создаем список временных точек для обработки
    time_points = np.arange(update_interval, duration, update_interval)
    
    # Оборачиваем цикл в tqdm
    for curr_time in tqdm(time_points, 
                         desc=f"Создание топокарт ({title})",
                         unit="кадр"):

        # Обрезаем данные
        data_copy = data.copy()
        data_segment = data_copy.crop(curr_time - update_interval, curr_time)

        # Извлечение PSD с фильтрацией по частотному диапазону
        spectrum_segment = data_segment.compute_psd()
        psds_segment = _extract_psd_for_segment(spectrum_segment, freq_lims)


        if normalize:
            psds_band_mean = _normalize_psd(psds_segment, vlim, threshold)
            plot_lims = (0,1)
        else:
            psds_band_mean = np.mean(psds_segment, axis=1)
            psds_band_mean = np.where(psds_band_mean < (vlim[1] - vlim[0]) * threshold, 0, psds_band_mean)
            plot_lims = vlim

        # Построение и отрисовка топографической карты
        fig, ax = plt.subplots()
        fig = _plot_topomap(fig, ax, data.info, psds_band_mean, plot_lims, colorbar=True)
        plt.draw()

        # Сохранение графика
        formatted_time = f"{curr_time:05.2f}"
        filename = f"{output_dir}/topomap_" + "_".join(band_dict.keys()) + "_" + formatted_time + ".png"
        plt.savefig(filename)

        plt.close()




