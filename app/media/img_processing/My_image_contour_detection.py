import cv2
import cv2.dnn
import numpy as np
from matplotlib import pyplot as plt


class My_Image_contour_detection:
    def __load_image(self, image_address):
        # print("Loading image from",image_address)
        image = cv2.imread(image_address)

        if image is None:
            print("Ошибка: Не удалось загрузить изображение")
        return image

    def __pixel_from(self, image, x, y):
        if (y >= 0 and y < image.shape[0] and x >= 0 and x < image.shape[1]):
            return image[y, x]
        else:
            return 0

    def __RGB_to_HSV(self, image):
        height, width = image.shape[:2]

        hsv = np.zeros((height, width, 3))
        for y in range(height):
            for x in range(width):
                b, g, r = image[y, x] / 255

                # h, s, v = hue, saturation, value
                cmax = max(r, g, b)  # maximum of r, g, b
                cmin = min(r, g, b)  # minimum of r, g, b
                diff = cmax - cmin  # diff of cmax and cmin
                h, s = -1, -1

                # if cmax and cmax are equal then h = 0
                if cmax == cmin:
                    h = 0

                # if cmax equal r then compute h
                elif cmax == r:
                    h = (60 * ((g - b) / diff) + 360) % 360

                # if cmax equal g then compute h
                elif cmax == g:
                    h = (60 * ((b - r) / diff) + 120) % 360

                # if cmax equal b then compute h
                elif cmax == b:
                    h = (60 * ((r - g) / diff) + 240) % 360

                # if cmax equal zero
                if cmax == 0:
                    s = 0
                else:
                    s = (diff / cmax) * 100

                # compute v
                v = cmax * 100

                hsv[y, x] = [int(h), int(s), int(v)]
        return hsv

    def __remove_color_range(self, image_rgb, min_hue, max_hue):
        image = image_rgb.copy()
        image_hsv = self.__RGB_to_HSV(image_rgb)
        for i in range(image_rgb.shape[0]):
            for j in range(image_rgb.shape[1]):
                if image_hsv[i, j, 0] >= min_hue and image_hsv[i, j, 0] <= max_hue:  # Зеленые оттенки в HSV
                    image[i, j] = (150, 150, 150)  # Замена на серый цвет
        return image

    def __RGB_to_gray(self, image):
        height, width = image.shape[:2]
        gray_img = np.zeros((height, width), dtype=np.uint8)

        for i in range(height):
            for j in range(width):
                gray_img[i, j] = np.dot(image[i, j], [0.299, 0.587, 0.114])
        return gray_img

    def __gaussian_kernel(self, size, sigma):
        # Kernel center calculation
        center = size // 2

        kernel = np.zeros((size, size))

        const = 1 / 2 * np.pi * sigma ** 2

        for x in range(size):
            for y in range(size):
                kernel[x][y] = const * np.exp(-((x - center) ** 2 + (y - center) ** 2) / (2 * sigma ** 2))
        # Kernel matrix normalisation
        kernel = kernel / np.sum(kernel)

        return kernel

    def __dilate(self, image, kernel):

        kernel_height, kernel_width = kernel.shape
        pad_height = kernel_height // 2
        pad_width = kernel_width // 2
        padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant',
                              constant_values=0)
        dilated_image = np.zeros_like(image)

        for i in range(pad_height, padded_image.shape[0] - pad_height):
            for j in range(pad_width, padded_image.shape[1] - pad_width):
                if np.any(padded_image[i - pad_height:i + pad_height + 1, j - pad_width:j + pad_width + 1] & kernel):
                    dilated_image[i - pad_height, j - pad_width] = 1

        return dilated_image

    def __erode(self, image, kernel):

        kernel_height, kernel_width = kernel.shape
        pad_height = kernel_height // 2
        pad_width = kernel_width // 2
        padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant',
                              constant_values=0)
        eroded_image = np.zeros_like(image)

        for i in range(pad_height, padded_image.shape[0] - pad_height):
            for j in range(pad_width, padded_image.shape[1] - pad_width):
                if np.all(padded_image[i - pad_height:i + pad_height + 1, j - pad_width:j + pad_width + 1] & kernel):
                    eroded_image[i - pad_height, j - pad_width] = 1

        return eroded_image

    def __morphology_close(self, image, kernel):

        dilated_image = self.__dilate(image, kernel)
        closed_image = self.__erode(dilated_image, kernel)
        return closed_image

    def __convolve(self, image, kernel):
        height, width = image.shape[:2]
        kernel_height, kernel_width = kernel.shape[:2]

        # Определяем отступы для краевых пикселей
        pad_height = kernel_height // 2
        pad_width = kernel_width // 2

        # Дополняем изображение нулями по краям
        padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant')

        # Если изображение цветное, добавляем паддинг для каналов
        if len(image.shape) == 3:
            padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width), (0, 0)), mode='constant')

        filtered_image = np.zeros_like(image, dtype=np.float32)

        for i in range(height):
            for j in range(width):
                if len(image.shape) == 3:
                    for c in range(image.shape[2]):
                        window = padded_image[i:i + kernel_height, j:j + kernel_width, c] * kernel
                        filtered_image[i, j, c] = np.sum(window)
                else:
                    window = padded_image[i:i + kernel_height, j:j + kernel_width] * kernel
                    filtered_image[i, j] = np.sum(window)

        # Обрезаем значения по диапазону [0, 255] и переводим в целочисленный тип
        filtered_image = filtered_image.astype(np.int64, copy=False)

        return filtered_image

    def __gaussian_blur(self, image, kernel_size, sigma):
        # Генерация ядра Гаусса
        kernel = self.__gaussian_kernel(kernel_size, sigma)

        # Применение фильтра Гаусса через свертку
        blurred_image = self.__convolve(image, kernel)

        return blurred_image

    def __sobel_operator(self, image, edge_threshold):
        height, width = image.shape[:2]

        filtered_image = np.zeros((height, width), dtype=np.uint8)

        for y in range(height):
            for x in range(width):
                # left column
                top_left = self.__pixel_from(image=image, x=x - 1, y=y - 1)
                left = self.__pixel_from(image=image, x=x - 1, y=y)
                bottom_left = self.__pixel_from(image=image, x=x - 1, y=y + 1)

                # Top & Bottom
                above = self.__pixel_from(image=image, x=x, y=y - 1)
                below = self.__pixel_from(image=image, x=x, y=y + 1)

                # Right column
                top_right = self.__pixel_from(image=image, x=x + 1, y=y - 1)
                right = self.__pixel_from(image=image, x=x + 1, y=y)
                bottom_right = self.__pixel_from(image=image, x=x + 1, y=y + 1)

                # Apply the Sobel operator
                # -1 0 1
                # -2 0 2    Horizontal
                # -1 0 1
                horizontal_gradient = -(top_left + 2 * left + bottom_left) + (top_right + 2 * right + bottom_right)

                # Apply the Sobel operator
                #  1 2 1
                #  0 0 0    Vertical
                # -1-2-1
                vertical_gradient = top_right + 2 * above + top_left - (bottom_right + 2 * below + bottom_left)

                # Calculate the gradient magnitude
                gradient_magnitude = (horizontal_gradient ** 2 + vertical_gradient ** 2) ** 0.5

                if gradient_magnitude > edge_threshold:
                    filtered_image[y, x] = 255
                else:
                    filtered_image[y, x] = 0

        return filtered_image

    def __zero_crossing(self, image, threshold):
        # Рассчитываем пороговое значение в зависимости от максимальной яркости изображения
        max_brightness = np.max(image)
        threshold_value = threshold * max_brightness

        # Создаем выходной массив той же формы, что и входное изображение, заполненный нулями
        zero_crossings = np.zeros_like(image)

        # Размеры изображения
        rows, cols = image.shape

        # Обрабатываем каждую точку изображения, кроме границ (1 до rows-1 и 1 до cols-1)
        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                # Смотрим соседние точки (по горизонтали, вертикали и диагоналям)
                neighbors = [image[i - 1, j], image[i + 1, j], image[i, j - 1], image[i, j + 1],
                             image[i - 1, j - 1], image[i - 1, j + 1], image[i + 1, j - 1], image[i + 1, j + 1]]

                # Проверяем пересечение через ноль и превышение порога
                if (image[i, j] > 0 and any(n < 0 for n in neighbors) or
                    image[i, j] < 0 and any(n > 0 for n in neighbors)) and \
                        np.abs(image[i, j]) >= threshold_value:
                    zero_crossings[i, j] = 255

        return zero_crossings

    def edges_marr_hildreth(self, image_address, sigma, threshold):
        image = self.__load_image(image_address)
        gray_image = self.__RGB_to_gray(image)
        LoG_kernel_size = int(2 * (np.ceil(3 * sigma)) + 1)

        x, y = np.meshgrid(np.arange(-LoG_kernel_size / 2 + 1, LoG_kernel_size / 2 + 1),
                           np.arange(-LoG_kernel_size / 2 + 1, LoG_kernel_size / 2 + 1))

        normal = 1 / (2.0 * np.pi * sigma ** 2)

        LoG_kernel = ((x ** 2 + y ** 2 - (2.0 * sigma ** 2)) / sigma ** 4) * \
                     np.exp(-(x ** 2 + y ** 2) / (2.0 * sigma ** 2)) / normal  # LoG filter

        # applying filter
        log = self.__convolve(image=gray_image, kernel=LoG_kernel)
        log = log.astype(np.int64, copy=False)
        # computing zero crossing
        zero_crossing = self.__zero_crossing(log, threshold=threshold)

        filtered_image = np.clip(zero_crossing, 0, 255).astype(np.uint8)

        # plotting images
        fig = plt.figure(figsize=(10, 10))

        a = fig.add_subplot(2, 2, 1)
        imgplot = plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        a.set_title('Original Image')
        a = fig.add_subplot(2, 2, 2)
        imgplot = plt.imshow(cv2.cvtColor(gray_image, cv2.COLOR_BGR2RGB))
        a.set_title('Gray Image')
        a = fig.add_subplot(2, 2, 3)
        imgplot = plt.imshow(log, cmap='gray')
        a.set_title('Laplasian of Gaussian blur \nthreshold = ' + str(threshold))
        a = fig.add_subplot(2, 2, 4)
        imgplot = plt.imshow(filtered_image, cmap='gray')
        string = 'Zero crossing \nsigma = '
        string += (str(sigma))
        a.set_title(string)
        plt.show()

    def edges_marr_hildreth_result_only(self, image, sigma, threshold):
        if isinstance(image, str):
            image = self.__load_image(image)
        gray_image = self.__RGB_to_gray(image)
        LoG_kernel_size = int(2 * (np.ceil(3 * sigma)) + 1)

        x, y = np.meshgrid(np.arange(-LoG_kernel_size / 2 + 1, LoG_kernel_size / 2 + 1),
                           np.arange(-LoG_kernel_size / 2 + 1, LoG_kernel_size / 2 + 1))

        normal = 1 / (2.0 * np.pi * sigma ** 2)

        LoG_kernel = ((x ** 2 + y ** 2 - (2.0 * sigma ** 2)) / sigma ** 4) * \
                     np.exp(-(x ** 2 + y ** 2) / (2.0 * sigma ** 2)) / normal  # LoG filter

        # applying filter
        log = self.__convolve(image=gray_image, kernel=LoG_kernel)
        log = log.astype(np.int64, copy=False)
        # computing zero crossing
        zero_crossing = self.__zero_crossing(log, threshold=threshold)

        filtered_image = np.clip(zero_crossing, 0, 255).astype(np.uint8)
        return filtered_image

    def sobel_edge_detection(self, image_address, kernel_size, sigma, threshold):
        image = self.__load_image(image_address)
        gray_image = self.__RGB_to_gray(image)
        blurred_image = self.__gaussian_blur(gray_image, kernel_size, sigma)
        edges = self.__sobel_operator(blurred_image, threshold)

        kernel = np.ones((3, 3), np.uint8)
        closed_edges = self.__morphology_close(edges, kernel)

        # plotting images
        fig = plt.figure(figsize=(13, 8))

        a = fig.add_subplot(2, 3, 1)
        imgplot = plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        a.set_title('Original Image')
        a = fig.add_subplot(2, 3, 2)
        imgplot = plt.imshow(gray_image, cmap='gray')
        a.set_title('Gray Image')
        a = fig.add_subplot(2, 3, 3)
        imgplot = plt.imshow(blurred_image, cmap='gray')
        a.set_title('Gaussian blur \n sigma =' + str(sigma) + '\nkernel size = ' + str(kernel_size))
        a = fig.add_subplot(2, 3, 4)
        imgplot = plt.imshow(edges, cmap='gray')
        string = 'Sobel Operator \n threshold = '
        string += (str(threshold))
        a.set_title(string)
        a = fig.add_subplot(2, 3, 5)
        imgplot = plt.imshow(closed_edges, cmap='gray')
        a.set_title('Morphology close')
        plt.show()

    def sobel_edge_detection_result_only(self, image, kernel_size, sigma, threshold):
        if isinstance(image, str):
            image = self.__load_image(image)
        gray_image = self.__RGB_to_gray(image)
        blurred_image = self.__gaussian_blur(gray_image, kernel_size, sigma)
        edges = self.__sobel_operator(blurred_image, threshold)

        kernel = np.ones((3, 3), np.uint8)
        closed_edges = self.__morphology_close(edges, kernel)
        return closed_edges

    def rcr_sobel_edge_detection(self, image_address, kernel_size, sigma, threshold, min_hue, max_hue):
        image = self.__load_image(image_address)
        colour_removed_image = self.__remove_color_range(image, min_hue, max_hue)

        gray_image = self.__RGB_to_gray(colour_removed_image)

        blurred_image = self.__gaussian_blur(gray_image, kernel_size, sigma)

        edges = self.__sobel_operator(blurred_image, threshold)

        kernel = np.ones((3, 3), np.uint8)
        closed_edges = self.__morphology_close(edges, kernel)

        # plotting images
        fig = plt.figure(figsize=(12, 8))

        a = fig.add_subplot(2, 3, 1)
        imgplot = plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        a.set_title('Original Image')
        a = fig.add_subplot(2, 3, 2)
        imgplot = plt.imshow(cv2.cvtColor(colour_removed_image, cv2.COLOR_BGR2RGB))
        a.set_title('Green color removed Image')
        a = fig.add_subplot(2, 3, 3)
        imgplot = plt.imshow(cv2.cvtColor(gray_image, cv2.COLOR_BGR2RGB))
        a.set_title('Gray Image')
        a = fig.add_subplot(2, 3, 4)
        imgplot = plt.imshow(blurred_image, cmap='gray')
        a.set_title('Gaussian blur \n sigma =' + str(sigma) + '\nkernel size = ' + str(kernel_size))
        a = fig.add_subplot(2, 3, 5)
        imgplot = plt.imshow(edges, cmap='gray')
        string = 'Sobel Operator \n threshold = '
        string += (str(threshold))
        a.set_title(string)
        a = fig.add_subplot(2, 3, 6)
        imgplot = plt.imshow(closed_edges, cmap='gray')
        a.set_title('Morphology close')
        plt.show()

    def rcr_sobel_edge_detection_result_only(self, image, kernel_size, sigma, threshold, min_hue, max_hue):
        if isinstance(image, str):
            image = self.__load_image(image)
        colour_removed_image = self.__remove_color_range(image, min_hue, max_hue)

        gray_image = self.__RGB_to_gray(colour_removed_image)

        blurred_image = self.__gaussian_blur(gray_image, kernel_size, sigma)

        edges = self.__sobel_operator(blurred_image, threshold)

        kernel = np.ones((3, 3), np.uint8)
        closed_edges = self.__morphology_close(edges, kernel)
        return edges, closed_edges

    def canny_edge(self, image_address, sigma, gauss_kernel_size, th1, th2):
        image = self.__load_image(image_address)

        gray = self.__RGB_to_gray(image)
        gauss = self.__gaussian_blur(gray, gauss_kernel_size, sigma)

        kernel, kern_size = np.array(
            [[-1, -1, -1], [0, 0, 0], [1, 1, 1]]), 3  # edge detection
        gx, gy = np.zeros_like(
            gauss, dtype=float), np.zeros_like(gauss, dtype=float)

        for i in range(gauss.shape[0] - (kern_size - 1)):
            for j in range(gauss.shape[1] - (kern_size - 1)):
                window = gauss[i:i + kern_size, j:j + kern_size]
                gx[i, j], gy[i, j] = np.sum(
                    window * kernel.T), np.sum(window * kernel)

        magnitude = np.sqrt(gx ** 2 + gy ** 2)
        epsilon = 1e-10
        theta = ((np.arctan(gy / (gx + epsilon))) / np.pi) * 180  # перевод радианов в градусы
        nms = np.copy(magnitude)

        theta[theta < 0] += 180

        # non maximum suppression; quantization and suppression done in same step
        for i in range(theta.shape[0] - (kern_size - 1)):
            for j in range(theta.shape[1] - (kern_size - 1)):
                if (theta[i, j] <= 22.5 or theta[i, j] > 157.5):
                    if (magnitude[i, j] <= magnitude[i - 1, j]) and (magnitude[i, j] <= magnitude[i + 1, j]):
                        nms[i, j] = 0
                if (theta[i, j] > 22.5 and theta[i, j] <= 67.5):
                    if (magnitude[i, j] <= magnitude[i - 1, j - 1]) and (magnitude[i, j] <= magnitude[i + 1, j + 1]):
                        nms[i, j] = 0
                if (theta[i, j] > 67.5 and theta[i, j] <= 112.5):
                    if (magnitude[i, j] <= magnitude[i + 1, j + 1]) and (magnitude[i, j] <= magnitude[i - 1, j - 1]):
                        nms[i, j] = 0
                if (theta[i, j] > 112.5 and theta[i, j] <= 157.5):
                    if (magnitude[i, j] <= magnitude[i + 1, j - 1]) and (magnitude[i, j] <= magnitude[i - 1, j + 1]):
                        nms[i, j] = 0

        weak, strong = np.copy(nms), np.copy(nms)

        # weak edges
        weak[weak < th1] = 0
        weak[weak > th2] = 0

        # strong edges
        strong[strong < th2] = 0
        strong[strong > th2] = 255

        fig = plt.figure(figsize=(12, 8))
        # plotting multiple images
        a = fig.add_subplot(2, 3, 1)
        imgplot = plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        a.set_title('Original Image')
        a = fig.add_subplot(2, 3, 2)
        imgplot = plt.imshow(gray, cmap='gray')
        a.set_title('Gray Image')
        a = fig.add_subplot(2, 3, 3)
        imgplot = plt.imshow(gauss, cmap='gray')
        a.set_title('Gaussian blur \n sigma =' + str(sigma) + '\nkernel size = ' + str(gauss_kernel_size))
        a = fig.add_subplot(2, 3, 4)
        imgplot = plt.imshow(magnitude, cmap='gray')
        a.set_title('Magnitude')
        a = fig.add_subplot(2, 3, 5)
        imgplot = plt.imshow(weak, cmap='gray')
        a.set_title('Weak edges \nmin th = ' + str(th1) + 'max th = ' + str(th2))
        a = fig.add_subplot(2, 3, 6)
        imgplot = plt.imshow(strong, cmap='gray')
        a.set_title('Strong edges')
        plt.show()

    def canny_edge_result_only(self, image, sigma, gauss_kernel_size, th1, th2):
        if isinstance(image, str):
            image = self.__load_image(image)

        gray = self.__RGB_to_gray(image)
        gauss = self.__gaussian_blur(gray, gauss_kernel_size, sigma)

        kernel, kern_size = np.array(
            [[-1, -1, -1], [0, 0, 0], [1, 1, 1]]), 3  # edge detection
        gx, gy = np.zeros_like(
            gauss, dtype=float), np.zeros_like(gauss, dtype=float)

        for i in range(gauss.shape[0] - (kern_size - 1)):
            for j in range(gauss.shape[1] - (kern_size - 1)):
                window = gauss[i:i + kern_size, j:j + kern_size]
                gx[i, j], gy[i, j] = np.sum(
                    window * kernel.T), np.sum(window * kernel)

        magnitude = np.sqrt(gx ** 2 + gy ** 2)
        epsilon = 1e-10
        theta = ((np.arctan(gy / (gx + epsilon))) / np.pi) * 180  # перевод радианов в градусы
        nms = np.copy(magnitude)

        theta[theta < 0] += 180

        # non maximum suppression; quantization and suppression done in same step
        for i in range(theta.shape[0] - (kern_size - 1)):
            for j in range(theta.shape[1] - (kern_size - 1)):
                if (theta[i, j] <= 22.5 or theta[i, j] > 157.5):
                    if (magnitude[i, j] <= magnitude[i - 1, j]) and (magnitude[i, j] <= magnitude[i + 1, j]):
                        nms[i, j] = 0
                if (theta[i, j] > 22.5 and theta[i, j] <= 67.5):
                    if (magnitude[i, j] <= magnitude[i - 1, j - 1]) and (magnitude[i, j] <= magnitude[i + 1, j + 1]):
                        nms[i, j] = 0
                if (theta[i, j] > 67.5 and theta[i, j] <= 112.5):
                    if (magnitude[i, j] <= magnitude[i + 1, j + 1]) and (magnitude[i, j] <= magnitude[i - 1, j - 1]):
                        nms[i, j] = 0
                if (theta[i, j] > 112.5 and theta[i, j] <= 157.5):
                    if (magnitude[i, j] <= magnitude[i + 1, j - 1]) and (magnitude[i, j] <= magnitude[i - 1, j + 1]):
                        nms[i, j] = 0

        weak, strong = np.copy(nms), np.copy(nms)

        # weak edges
        weak[weak < th1] = 0
        weak[weak > th2] = 0

        # strong edges
        strong[strong < th2] = 0
        strong[strong > th2] = 255

        return weak, strong

    def test_MH(self, image):
        gray_image = self.__RGB_to_gray(image=image)
        log = self.edges_marr_hildreth(gray_image, 5, 0.02)
        return log
