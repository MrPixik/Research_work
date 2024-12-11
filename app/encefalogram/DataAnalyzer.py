import mne
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
matplotlib.use('TkAgg')



class DataAnalyzer:
    data_path = r'../../static/eeg_data/preprocessed/preprocessed_data_eeg.fif'
    delta = 'Delta (0-4 Hz)'
    theta = 'Theta (4-8 Hz)'
    alpha = 'Alpha (8-12 Hz)'
    beta = 'Beta (12-30 Hz)'
    gamma = 'Gamma (30-45 Hz)'
    bands = {delta: (0, 4), theta: (4, 8),
             alpha: (8, 12), beta: (12, 30),
             gamma: (30, 45)}

    @staticmethod
    def __select_bands(selected_band_names):
        # Создание нового словаря с выбранными каналами
        selected_bands = {name: DataAnalyzer.bands[name] for name in selected_band_names}
        return selected_bands
    @staticmethod
    def __load_data():
        raw = mne.io.read_raw(DataAnalyzer.data_path, preload=True)
        raw.load_data()
        return raw
    @staticmethod
    def show_topomap():

        data = DataAnalyzer.__load_data()
        spectrum = data.compute_psd()

        spectrum.plot_topomap()
        plt.show()

    @staticmethod
    def plot_topomap_by_band(bands):
        """
        Строит топографическую карту для заданного частотного диапазона.

        :param bands: list with band names (alpha,beta, etc.)
        """

        data = DataAnalyzer.__load_data()

        spectrum = data.compute_psd()

        band_dict = DataAnalyzer.__select_bands(bands)
        spectrum.plot_topomap(bands=band_dict)
        plt.show()

    @staticmethod
    def save_topomap_by_time(bands, update_interval=1, output_dir='./'):
        """
        Строит анимацию топографической карты по времени с возможностью выбрать интервал обновления.

        :param bands: list with band names (alpha, beta, etc.)
        :param update_interval: интервал времени в секундах для обновления карты.
        :param save_images: если True, сохраняет изображения вместо отображения анимации.
        :param output_dir: директория для сохранения изображений (если save_images=True).
        """
        data = DataAnalyzer.__load_data()

        # Выбираем нужные частотные диапазоны
        band_dict = DataAnalyzer.__select_bands(bands)

        duration = int(data.times[-1])

        curr_time = 1.0
        while curr_time < duration:

            data_copy = data.copy()
            data_segment = data_copy.crop(curr_time - 1.0, curr_time)
            spectrum = data_segment.compute_psd()

            # Создаем фигуру и оси для графика
            fig, ax = plt.subplots()

            fig = spectrum.plot_topomap(bands=band_dict, axes=ax)

            # Удаление цветовой шкалы
            colorbar = fig.axes[-1]
            colorbar.remove()

            # Явная отрисовка графика, чтобы не сохранять пустое изображение
            plt.draw()


            # Создаем имя файла для сохранения

            filename = f"{output_dir}/topomap_" + "_".join(band_dict.keys()) + f"_{curr_time}.png"
            plt.savefig(filename)  # Сохраняем текущий кадр

            plt.close()
            curr_time += update_interval

