import cv2.dnn
from skimage.util import img_as_float
from skimage.segmentation import active_contour, mark_boundaries
import cv2
import numpy as np
from scipy.cluster.vq import kmeans, vq
from skimage.transform import resize
from matplotlib import pyplot as plt


class Image_contour_detection:
    thresholdOtsu = "Otsu"
    thresholdAdaptive = "Adaptive"
    def __load_image(self, image_address):
        image = cv2.imread(image_address)

        if image is None:
            print("Ошибка: Не удалось загрузить изображение")
        return image

    def __threshold_image(self, image, threshold_method):
        if threshold_method == self.thresholdOtsu:
            _, binary = cv2.threshold(image.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif threshold_method == self.thresholdAdaptive:
            binary = cv2.adaptiveThreshold(image.astype(np.uint8), 255,
                                                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                    cv2.THRESH_BINARY_INV, blockSize=11, C=2)
        else:
            _, binary = cv2.threshold(image.astype(np.uint8), 0, 255, cv2.THRESH_BINARY)
        return binary
    def __preprocess_image(self, original_image):
        gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        # Билинейная интерполяция через сжатие изображения
        downsampled = cv2.pyrDown(gray_image)
        upsampled = cv2.pyrUp(downsampled)

        # Морфологическое замыкание и размыкание радиусом 7 пикселя
        kernel = np.ones((7, 7), np.uint8)
        opening = cv2.morphologyEx(upsampled, cv2.MORPH_OPEN, kernel)
        closing = cv2.morphologyEx(upsampled, cv2.MORPH_CLOSE, kernel)

        # Среднее значение между открытием и закрытием
        combined = cv2.addWeighted(opening, 0.5, closing, 0.5, 0)

        return combined

    def denoise(self,image_path, tolerance=0.1, tau=0.125, tv_weight=100):
        """ An implementation of the Rudin-Osher-Fatemi (ROF) denoising model
            using the numerical procedure presented in Eq. (11) of A. Chambolle
            (2005). Implemented using periodic boundary conditions.

            Input: noisy input image (grayscale), initial guess for U, weight of
            the TV-regularizing term, steplength, tolerance for the stop criterion

            Output: denoised and detextured image, texture residual. """

        im = cv2.imread(image_path)
        im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
        m, n = im.shape  # size of noisy image

        # initialize
        U = np.copy(im)
        Px = im  # x-component to the dual field
        Py = im  # y-component of the dual field
        error = 1

        while (error > tolerance):
            Uold = U

            # gradient of primal variable
            GradUx = np.roll(U, -1, axis=1) - U  # x-component of U's gradient
            GradUy = np.roll(U, -1, axis=0) - U  # y-component of U's gradient

            # update the dual varible
            PxNew = Px + (tau / tv_weight) * GradUx  # non-normalized update of x-component (dual)
            PyNew = Py + (tau / tv_weight) * GradUy  # non-normalized update of y-component (dual)
            NormNew = np.maximum(1, np.sqrt(PxNew ** 2 + PyNew ** 2))

            Px = PxNew / NormNew  # update of x-component (dual)
            Py = PyNew / NormNew  # update of y-component (dual)

            # update the primal variable
            RxPx = np.roll(Px, 1, axis=1)  # right x-translation of x-component
            RyPy = np.roll(Py, 1, axis=0)  # right y-translation of y-component

            DivP = (Px - RxPx) + (Py - RyPy)  # divergence of the dual field.
            U = im + tv_weight * DivP  # update of the primal variable

            # update of error
            error = np.linalg.norm(U - Uold) / np.sqrt(n * m);
        U = U.astype(np.uint8)
        cv2.imshow("denoised",U)
        cv2.imshow("residual",im - U)
        cv2.waitKey()

        return U, im - U  # denoised image and texture residual

    from skimage.transform import resize
    import numpy as np
    import cv2
    import matplotlib.pyplot as plt
    from scipy.cluster.vq import kmeans, vq

    def classrerizarion(self, image_path):
        image = cv2.imread(image_path)
        steps = 200

        dx, dy, _ = image.shape
        block_size_x = dx // steps
        block_size_y = dy // steps

        features = []
        for x in range(steps):
            for y in range(steps):
                x_start, x_end = x * block_size_x, (x + 1) * block_size_x
                y_start, y_end = y * block_size_y, (y + 1) * block_size_y

                # Проверка на непустой блок перед расчетом среднего
                block = image[x_start:x_end, y_start:y_end]
                if block.size > 0:
                    R = np.mean(block[:, :, 0])
                    G = np.mean(block[:, :, 1])
                    B = np.mean(block[:, :, 2])
                    features.append([R, G, B])

        # Преобразуем список признаков в массив
        features = np.array(features, dtype="f")

        if len(features) == 0:
            print("Нет доступных данных для кластеризации")
            return

        centroids, variance = kmeans(features, 3)
        code, distance = vq(features, centroids)

        codeim = code.reshape(steps, steps)
        codeim = resize(codeim, image.shape[:2], order=0, preserve_range=True, anti_aliasing=False)

        plt.figure()
        plt.imshow(codeim)
        plt.show()

    def graph_based_segmentation(self, image_path):
        # Загрузка изображения
        image = cv2.imread(image_path)

        # Отображение оригинального изображения
        plt.figure()
        plt.subplot(231)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # Обратно в RGB для matplotlib
        plt.title("Оригинальное изображение")

        # Получение размеров изображения
        rows, cols, _ = image.shape

        # Инициализация маски и моделей
        mask = np.zeros((rows, cols), np.uint8)
        bgr_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)

        # Определение прямоугольника внутри границ изображения
        x0, y0 = 0, 0
        x1, y1 = cols-1, rows-1  # Ограничиваем размеры, чтобы они не выходили за пределы
        rect = (x0, y0, x1 - x0, y1 - y0)

        # Применение GrabCut
        cv2.grabCut(image, mask, rect, bgr_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)

        # Получение бинарной маски переднего плана
        # Преобразование маски: выделяем только пиксели переднего плана
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        segmented_image = image * mask2[:, :, np.newaxis]

        # Отображение результатов
        plt.subplot(232)
        plt.imshow(mask, cmap='gray')
        plt.title("Маска GrabCut")

        plt.subplot(233)
        plt.imshow(cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB))
        plt.title("Сегментированное изображение")

        plt.show()

        # Блок для использования active_contours с конутром полученным через сегментацию

        # binary_mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        #
        # # Находим контуры на бинарной маске
        # contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #
        # # Проверка, что хотя бы один контур найден
        # if len(contours) == 0:
        #     raise ValueError("Контуры не найдены на маске.")
        #
        # # Используем самый большой контур
        # largest_contour = max(contours, key=cv2.contourArea)
        #
        # # Преобразуем контур в нужный формат для active_contour (N, 2)
        # init_snake = largest_contour[:, 0, :].astype(float)
        #
        # # Применение функции active_contour
        # snake = active_contour(img_as_float(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)),
        #                        init_snake, alpha=0.015, beta=10, gamma=0.001)
        #
        # # Визуализация результатов
        # plt.figure()
        # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        # plt.plot(init_snake[:, 0], init_snake[:, 1], '--r', lw=2, label='Начальный контур')
        # plt.plot(snake[:, 0], snake[:, 1], '-b', lw=2, label='Active Contour')
        # plt.legend()
        # plt.title("Active Contours")
        # plt.show()

    def active_contours(self, image):
        # Загрузка изображения
        original_image = cv2.imread(image)
        if original_image is None:
            raise FileNotFoundError("Изображение не найдено. Проверьте путь.")

        # Конвертация изображения в оттенки серого и нормализация
        gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        image_float = img_as_float(gray_image)

        # Бинаризация изображения и нахождение контуров
        _, binary = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Проверка наличия контуров
        if contours:
            init_snake = contours[0].squeeze()

            # Если контур одномерный, преобразовать его в нужную форму
            if init_snake.ndim == 1 and len(init_snake) % 2 == 0:
                init_snake = init_snake.reshape(-1, 2)
            elif init_snake.ndim != 2 or init_snake.shape[1] != 2:
                raise ValueError("Контур имеет неправильную форму. Ожидался массив размером (N, 2).")

            # Применение алгоритма Active Contours
            snake = active_contour(image_float, init_snake, alpha=0.015, beta=10, gamma=0.001)

            # Отображение результата
            fig, ax = plt.subplots(figsize=(7, 7))
            ax.imshow(gray_image, cmap=plt.cm.gray)
            ax.plot(init_snake[:, 0], init_snake[:, 1], '--r', lw=2, label='Initial Snake')
            ax.plot(snake[:, 0], snake[:, 1], '-b', lw=3, label='Final Snake')
            ax.set_title("Active Contour (Snake)")
            ax.legend()
            plt.show()
        else:
            print("Контуры не найдены.")

    def contour_tracing(self, image
                        , threshold_method):

        # image = cv2.imread(image_path)


        preprocessed_image = self.marr_hildreth_edge_detection(self.__preprocess_image(image),5)
        # cv2.imshow("preprocessed_image", preprocessed_image)
        # cv2.waitKey()

        binary = self.__threshold_image(preprocessed_image, threshold_method)

        # kernel = np.ones((5, 5), np.uint8)
        # morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # output = original_image.copy()
        # cv2.drawContours(output, contours, -1, (0,255,0),2)
        mask = np.zeros_like(image)

        # Фильтрация контуров по площади (например, оставляем только крупные контуры)
        min_area = 300  # Задайте минимальную площадь для контура
        main_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

        # Рисуем отфильтрованные контуры на маске
        cv2.drawContours(mask, main_contours, -1, 255, thickness=cv2.FILLED)

        # Наложение маски на исходное изображение
        result = cv2.bitwise_and(image, mask)
        return cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    def marr_hildreth_edge_detection(self,image, sigma):
        # Шаг 1: Применить размытие Гаусса
        blurred_image = cv2.GaussianBlur(image, (3, 3), sigma)

        # Шаг 2: Вычислить Лапласиан Гаусса
        log_image = cv2.Laplacian(blurred_image, cv2.CV_64F)

        # Шаг 3: Найти нулевые пересечения
        # Вычисляем абсолютное значение Лапласиана и нормализуем
        abs_log = np.absolute(log_image)
        abs_log = (abs_log / np.max(abs_log)) * 255

        return abs_log
        # edges = self.__threshold_image(abs_log, threshold_method)
        #
        # return edges
    def canny_edge_detection(self, image, aperture_size, threshold1, threshold2):
        return cv2.Canny(image, threshold1, threshold2, aperture_size)
    def sobel_edge_detection(self, image, kernel_size, threshold_method):
        # max_brightness = np.max(image)
        # threshold_value = threshold * max_brightness

        gray_frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sobelx = cv2.Sobel(gray_frame, cv2.CV_64F, 1, 0, ksize=kernel_size)  # производная по X
        sobely = cv2.Sobel(gray_frame, cv2.CV_64F, 0, 1, ksize=kernel_size)  # производная по Y
        processed_frame = cv2.sqrt(sobelx ** 2 + sobely ** 2)
        processed_frame = cv2.normalize(processed_frame, None, 0, 255, cv2.NORM_MINMAX)

        processed_frame = self.__threshold_image(processed_frame, threshold_method)

        return processed_frame
