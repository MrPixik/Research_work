import mne
import matplotlib
import matplotlib.pyplot as plt
from internal.constants.eeg_channels import *
from internal.constants.paths import *
import numpy as np
matplotlib.use('TkAgg')


class EEG_analyzer:

    @staticmethod
    def __select_bands(selected_band_name):
        # Создание нового словаря с выбранным каналом
        return {selected_band_name: BANDS[selected_band_name]}
    @staticmethod
    def __load_data():
        PREPROCESSED_EEG_DATA_PATH = get_preprocessed_eeg_data_path(Path(__file__))
        eeg_data = mne.io.read_raw(PREPROCESSED_EEG_DATA_PATH, preload=True)
        eeg_data.load_data()
        return eeg_data

    @staticmethod
    def _get_glob_minmax(spectrum, freq_lims, threshold=1):

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

        vmin = np.percentile(psds_band_uv, 5)  # 5-й перцентиль
        vmax = np.percentile(psds_band_uv, 95)  # 95-й перцентиль

        return (vmin,vmax)

    @staticmethod
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

    @staticmethod
    def _normalize_psd(psds_band_uv, vlim, threshold=1):
        vmin,vmax = vlim
        psds_band_norm = (psds_band_uv - vmin) / (vmax - vmin)
        psds_band_mean = np.mean(psds_band_norm, axis=1)
        psds_band_mean_binary = np.where(psds_band_mean >= threshold, 1, 0)
        return psds_band_mean_binary

    @staticmethod
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

    @staticmethod
    def save_topomap_by_time(title, freq_lims, update_interval=1, threshold=1, normalize=False):
        """
        Сохраняет топографические карты по времени с возможностью выбрать частотный диапазон и интервал обновления.

        :param title: название графика
        :param freq_lims: частостный интервал
        :param update_interval: интервал времени в секундах для обновления карты
        :param threshold: пороговое значение для отображения топографической карты [0,1]
        :param normalize: нормировать данные
        """

        data = EEG_analyzer.__load_data()

        # Словарь с названием + частотный диапазон
        band_dict = {title: freq_lims}

        # Вычисляем значения глобальных MIN и MAX
        spectrum = data.compute_psd()
        vlim = EEG_analyzer._get_glob_minmax(spectrum, freq_lims)

        duration = int(data.times[-1])
        output_dir = get_topomap_path(title)
        curr_time = 1.0
        while curr_time < duration:

            # Обрезаем данные
            data_copy = data.copy()
            data_segment = data_copy.crop(curr_time - 1.0, curr_time)

            # Извлечение PSD с фильтрацией по частотному диапазону
            spectrum_segment = data_segment.compute_psd()
            psds_segment = EEG_analyzer._extract_psd_for_segment(spectrum_segment, freq_lims)


            if normalize:
                psds_band_mean = EEG_analyzer._normalize_psd(psds_segment, vlim)
                plot_lims = (0,1)
            else:
                psds_band_mean = np.mean(psds_segment, axis=1)
                psds_band_mean = np.where(psds_band_mean < (vlim[1] - vlim[0]) * threshold, 0, psds_band_mean)
                plot_lims = vlim

            # Построение и отрисовка топографической карты
            fig, ax = plt.subplots()
            fig = EEG_analyzer._plot_topomap(fig, ax, data.info, psds_band_mean, plot_lims, colorbar=True)
            plt.draw()

            # Сохранение графика
            formatted_time = f"{curr_time:05.2f}"
            filename = f"{output_dir}/topomap_" + "_".join(band_dict.keys()) + "_" + formatted_time + ".png"
            plt.savefig(filename)

            plt.close()
            curr_time += update_interval



