import cv2.dnn
from skimage.util import img_as_float
from skimage.segmentation import active_contour, mark_boundaries
import cv2
import numpy as np
from scipy.cluster.vq import kmeans, vq
from skimage.transform import resize
from matplotlib import pyplot as plt


class Topomap_analysis:

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
    def show_inactive_area(image_addr):
        """
        Вычисляет и отображает площадь неактивной (темной) области на топографической карте.

        :param image_addr : Путь к изображению топографической карты.
        """

        # Обрезание картинки, чтобы оптимизировать предобработку
        img = Topomap_analysis.__load_image(image_addr)

        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        cropped_image = Topomap_analysis.__crop_topomap_image(gray_img)

        # Создание маски для того, чтобы убрать белый фон
        _, threshold_mask = cv2.threshold(cropped_image, 254, 255, cv2.THRESH_BINARY_INV)

        thresholded_img = cropped_image.copy()

        # Наложение маски на изображение
        thresholded_img[threshold_mask == 0] = 0

        _, thresholded_img = cv2.threshold(thresholded_img, 235, 255, cv2.THRESH_BINARY)

        # Нахождение контура
        contours, hierarchy = cv2.findContours(thresholded_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        largest_contour = max(contours, key=cv2.contourArea)

        # Вычисление площади
        area = cv2.contourArea(largest_contour)
        print(f"Площадь неактивной области: {area} пикселей")

        # Рисование контура на изображении
        img_with_contour = cv2.cvtColor(thresholded_img, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(img_with_contour, [largest_contour], -1, (0, 255, 0), 1)

        cv2.imshow("Контур", img_with_contour)
        cv2.waitKey(0)

        # return area