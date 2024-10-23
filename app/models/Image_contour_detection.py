import cv2
import cv2.dnn
import numpy as np
from matplotlib import pyplot as plt


class Image_contour_detection:
    def __load_image(self, image_address):
        image = cv2.imread(image_address)

        if image is None:
            print("Ошибка: Не удалось загрузить изображение")
        return image


    def marr_hildreth_edge_detection(self,image, sigma, threshold):
        max_brightness = np.max(image)
        threshold_value = threshold * max_brightness
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Шаг 1: Применить размытие Гаусса
        blurred_image = cv2.GaussianBlur(gray_image, (3, 3), sigma)

        # Шаг 2: Вычислить Лапласиан Гаусса
        log_image = cv2.Laplacian(blurred_image, cv2.CV_64F)

        # Шаг 3: Найти нулевые пересечения
        # Вычисляем абсолютное значение Лапласиана и нормализуем
        abs_log = np.absolute(log_image)
        abs_log = (abs_log / np.max(abs_log)) * 255

        # Применяем пороговое значение, чтобы выделить только значимые пересечения
        edges = np.uint8(abs_log > threshold_value) * 255

        return edges
    def canny_edge_detection(self, image, aperture_size, threshold1, threshold2):
        return cv2.Canny(image, threshold1, threshold2, aperture_size)
    def sobel_edge_detection(self, image, kernel_size, threshold):
        max_brightness = np.max(image)
        threshold_value = threshold * max_brightness

        gray_frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sobelx = cv2.Sobel(gray_frame, cv2.CV_64F, 1, 0, ksize=kernel_size)  # производная по X
        sobely = cv2.Sobel(gray_frame, cv2.CV_64F, 0, 1, ksize=kernel_size)  # производная по Y
        processed_frame = cv2.sqrt(sobelx ** 2 + sobely ** 2)
        processed_frame = cv2.normalize(processed_frame, None, 0, 255, cv2.NORM_MINMAX)

        edges = np.uint8(processed_frame > threshold_value) * 255

        return edges
