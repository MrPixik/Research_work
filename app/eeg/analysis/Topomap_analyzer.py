import cv2.dnn
import cv2
import os
import re
import math
import matplotlib.pyplot as plt
from internal.constants.eeg_channels import *
from internal.constants.paths import *
from tqdm import tqdm


class Topomap_analyzer:

    @staticmethod
    def __load_image(image_address):
        """
        Загружает изображение с помощью OpenCV.

        :param image_address (str): Путь к изображению.

        Возвращает: numpy.ndarray: Загруженное изображение в формате BGR. Возвращает None, если изображение не найдено.
        """
        image = cv2.imread(image_address)

        if image is None:
            print("Ошибка: Не удалось загрузить изображение")
        return image

    @staticmethod
    def __crop_topomap_image(image):
        """
        Обрезает изображение, оставляя только область, содержащую топографическую карту.

        :param image (numpy.ndarray): Входное изображение (grayscale).

        Возвращает numpy.ndarray: Обрезанное изображение.
        """

        # Инициализация границ топографической карты (значения подобраны так, чтобы уменьшить график только до рабочей области)
        Y_MAX, X_MAX = 440, 470
        Y_MIN, X_MIN = 60, 110

        # Обрезка изображения
        return image[Y_MIN:Y_MAX, X_MIN:X_MAX]

    @staticmethod
    def __count_occupied_percentage(binary_img):
        # Находим все контуры
        contours, hierarchy = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Инициализируем переменную для хранения общей площади
        total_area = 0

        # Проходимся по всем контурам и добавляем их площади
        for contour in contours:
            total_area += cv2.contourArea(contour)

        full_area = math.pi * math.pow(TOPOMAP_DIAM_PIXELS, 2) / 4

        return (total_area / full_area) * 100

    @staticmethod
    def inactive_area_img(image_addr):
        """
        Отображает неактивную область мозга на бинарном изображении

        :param image_addr : Путь к изображению топографической карты.
        """

        img = Topomap_analyzer.__load_image(image_addr)

        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        cropped_image = Topomap_analyzer.__crop_topomap_image(gray_img)

        # Создание маски для того, чтобы убрать белый фон
        _, threshold_mask = cv2.threshold(cropped_image, 254, 255, cv2.THRESH_BINARY_INV)

        thresholded_img = cropped_image.copy()

        # Наложение маски на изображение
        thresholded_img[threshold_mask == 0] = 0

        _, thresholded_img = cv2.threshold(thresholded_img, 235, 255, cv2.THRESH_BINARY)


        return thresholded_img
        # # Нахождение контура
        # contours, hierarchy = cv2.findContours(thresholded_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #
        # largest_contour = max(contours, key=cv2.contourArea)
        #
        # # Вычисление площади
        # area = cv2.contourArea(largest_contour)
        # print(f"Площадь неактивной области: {area} пикселей")
        #
        # # Рисование контура на изображении
        # img_with_contour = cv2.cvtColor(thresholded_img, cv2.COLOR_GRAY2BGR)
        # cv2.drawContours(img_with_contour, [largest_contour], -1, (0, 255, 0), 1)

        # cv2.imshow("Контур", img_with_contour)
        # cv2.waitKey(0)

    @staticmethod
    def create_video_of_inactive_areas(band):
        # Путь к директориям с изображениями и видео
        topomap_origin_path = TOPOMAP_ORIGIN_PATH.get(band)
        topomap_processed_path = TOPOMAP_PROCESSED_PATH.get(band)
        output_video_path = TOPOMAP_VIDEO_PATH.get(band) + '/' + band + r'.mp4'

        # Функция для извлечения чисел с плавающей точкой из имени файла
        def extract_float_numbers(file_name):
            match = re.search(r'(\d+\.\d+)', file_name)  # Ищем число с точкой
            return float(match.group()) if match else float('inf')  # Преобразуем в float

        # Получаем и сортируем файлы с учётом числовых значений
        files = sorted(
            [f for f in os.listdir(topomap_origin_path) if f.endswith('.png')],
            key=extract_float_numbers
        )

        # Чтение первого изображения для получения размеров видео
        first_image_path = os.path.join(topomap_origin_path, files[0])
        frame = cv2.imread(first_image_path)

        if frame is None:
            print(f"Не удалось прочитать изображение: {first_image_path}")
            exit()

        height, width = 380, 360  # Размеры изображения
        size = (width, height)
        fps = 50
        total_frames = len(files)

        # Создание объекта VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Кодек (можно заменить на 'mp4v' для .mp4)
        out = cv2.VideoWriter(output_video_path, fourcc, fps, size)

        occupied_percentages = []
        time_intervals = []
        t = 1.0
        with tqdm(total=total_frames, desc="Processing video") as pbar:

            # Обработка изображений
            for filename in files:
                file_path = os.path.join(topomap_origin_path, filename)
                processed_frame = Topomap_analyzer.inactive_area_img(file_path)

                # path = os.path.join(topomap_processed_path,file_name)
                # cv2.imwrite(path, processed_frame)

                occupied_percentages.append(Topomap_analyzer.__count_occupied_percentage(processed_frame))
                time_intervals.append(t)
                # Записываем обработанный кадр в выходное видео
                if len(processed_frame.shape) == 2:  # Если изображение имеет два измерения (градации серого)
                    processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_GRAY2BGR)

                t += 0.02
                out.write(processed_frame)
                pbar.update(1)

        # Среднее значение
        average = sum(occupied_percentages) / len(occupied_percentages)

        print(average)

        # Построение графика
        plt.figure(figsize=(10, 6))  # Настраиваем размер графика
        plt.plot(time_intervals, occupied_percentages, label="Процент неактивной области", color="b")

        # Добавляем подписи и заголовок

        plt.xlabel("Время (с)", fontsize=14)
        plt.ylabel("Процент неактивной области площади (%)", fontsize=14)

        # Настройка осей и сетки
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)

        # Легенда
        plt.legend(fontsize=12)

        # plt.show()
        out.release()
        cv2.destroyAllWindows()
