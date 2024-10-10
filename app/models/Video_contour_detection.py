import cv2
from tqdm import tqdm
import numpy as np
import os
from app.models.Image_contour_detection import Image_contour_detection


class Video_contour_detection:
    sobel_save_path = r'static/videos/result/sobel_edge'
    mh_save_path = r'static/videos/result/marr_hildreth'
    canny_save_path = r'static/videos/result/canny_edge'

    def __generate_full_filepath(self, origin_video_addr, method_name, **kwargs):
        if method_name == "SOBEL_EDGE":
            path = self.sobel_save_path
        elif method_name == "MARR_HILDRETH":
            path = self.mh_save_path
        elif method_name == "CANNY_EDGE":
            path = self.canny_save_path

        # Извлекаем имя оригинального файла без расширения
        base_name = os.path.splitext(os.path.basename(origin_video_addr))[0]

        # Формируем имя файла с параметрами функции
        params_str = "_".join([f"{k}={v}" for k, v in kwargs.items()])
        filename = f"{base_name}_{method_name}_{params_str}.mp4"

        # Возвращаем полный путь к новому файлу
        return os.path.join(path, filename)
    def __generate_2_full_filepath(self, origin_video_addr, method_name, **kwargs):
        if method_name == "SOBEL_EDGE":
            path = self.sobel_save_path
        elif method_name == "MARR_HILDRETH":
            path = self.mh_save_path
        elif method_name == "CANNY_EDGE":
            path = self.canny_save_path

        # Извлекаем имя оригинального файла без расширения
        base_name = os.path.splitext(os.path.basename(origin_video_addr))[0]

        # Формируем имя файла с параметрами функции
        params_str = "_".join([f"{k}={v}" for k, v in kwargs.items()])
        filename1 = f"{base_name}_{method_name}_1_{params_str}.mp4"
        filename2 = f"{base_name}_{method_name}_2_{params_str}.mp4"

        # Возвращаем полный путь к новому файлу
        return os.path.join(path, filename1), os.path.join(path, filename2)
    def sobel_edge_detection(self, origin_video_addr, kernel_size,sigma, threshold):
        cd = Image_contour_detection()
        cap = cv2.VideoCapture(origin_video_addr)

        if not cap.isOpened():
            print(f"Ошибка: не удалось открыть видео {origin_video_addr}")
            return

        # Получаем информацию о видео
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        dest_video_addr = self.__generate_full_filepath(origin_video_addr, "SOBEL_EDGE",
                                                        kernel_size=kernel_size, sigma=sigma, threshold=threshold)

        # Проверка расширения файла
        file_extension = os.path.splitext(origin_video_addr)[-1].lower()
        if file_extension == '.mp4':
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        elif file_extension == '.avi':
            fourcc = cv2.VideoWriter_fourcc(*'XVID')

        out = cv2.VideoWriter(dest_video_addr, fourcc, fps, (frame_width, frame_height))

        with tqdm(total=total_frames, desc="Processing video") as pbar:
            while True:
                # Читаем кадр
                ret, frame = cap.read()

                if not ret:
                    break

                processed_frame =  cd.sobel_edge_detectionResultOnly(frame,kernel_size,sigma,threshold)

                processed_frame = (processed_frame * 255).astype(np.uint8)
                # cv2.imshow("Live", processed_frame)
                # key = cv2.waitKey(1)
                #
                # if key == ord("q"):
                #     break
                # Записываем обработанный кадр в выходное видео
                if len(processed_frame.shape) == 2:  # Если изображение имеет два измерения (градации серого)
                    processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_GRAY2BGR)

                out.write(processed_frame)
                pbar.update(1)


        cap.release()
        out.release()
        cv2.destroyAllWindows()
    def rcr_sobel_edge_detection(self, origin_video_addr, dest_video_addr1,dest_video_addr2, kernel_size, sigma, threshold, min_hue, max_hue):
        cd = Image_contour_detection()
        cap = cv2.VideoCapture(origin_video_addr)

        if not cap.isOpened():
            print(f"Ошибка: не удалось открыть видео {origin_video_addr}")
            return

        # Получаем информацию о видео
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Проверка расширения файла
        file_extension = os.path.splitext(origin_video_addr)[-1].lower()
        if file_extension == '.mp4':
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        elif file_extension == '.avi':
            fourcc = cv2.VideoWriter_fourcc(*'XVID')

        out1 = cv2.VideoWriter(dest_video_addr1, fourcc, fps, (frame_width, frame_height))
        out2 = cv2.VideoWriter(dest_video_addr2, fourcc, fps, (frame_width, frame_height))

        with tqdm(total=total_frames, desc="Processing video") as pbar:
            while True:
                # Читаем кадр
                ret, frame = cap.read()

                if not ret:
                    break

                processed_frame1, processed_frame2 = cd.edges_marr_hildreth_result_only(frame, kernel_size, sigma, threshold, min_hue, max_hue)

                # cv2.imshow("Live", processed_frame)
                # key = cv2.waitKey(1)
                #
                # if key == ord("q"):
                #     break

                # Записываем обработанный кадр в выходное видео
                if len(processed_frame.shape) == 2:  # Если изображение имеет два измерения (градации серого)
                    processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_GRAY2BGR)

                out1.write(processed_frame1)
                out2.write(processed_frame2)
                pbar.update(1)

        cap.release()
        out1.release()
        out2.release()
        cv2.destroyAllWindows()
    def edges_marr_hildreth(self, origin_video_addr, sigma, threshold):
        cd = Image_contour_detection()
        cap = cv2.VideoCapture(origin_video_addr)

        if not cap.isOpened():
            print(f"Ошибка: не удалось открыть видео {origin_video_addr}")
            return

        # Получаем информацию о видео
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Проверка расширения файла
        file_extension = os.path.splitext(origin_video_addr)[-1].lower()
        if file_extension == '.mp4':
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        elif file_extension == '.avi':
            fourcc = cv2.VideoWriter_fourcc(*'XVID')

        dest_video_addr = self.__generate_full_filepath(origin_video_addr, "MARR_HILDRETH",
                                                        sigma=sigma, threshold=threshold)

        out = cv2.VideoWriter(dest_video_addr, fourcc, fps, (frame_width, frame_height))

        with tqdm(total=total_frames, desc="Processing video") as pbar:
            while True:
                # Читаем кадр
                ret, frame = cap.read()

                if not ret:
                    break

                processed_frame =  cd.edges_marr_hildreth_result_only(frame,sigma,threshold)

                # cv2.imshow("Live", processed_frame)
                # key = cv2.waitKey(1)
                #
                # if key == ord("q"):
                #     break

                # Записываем обработанный кадр в выходное видео
                if len(processed_frame.shape) == 2:  # Если изображение имеет два измерения (градации серого)
                    processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_GRAY2BGR)

                out.write(processed_frame)
                pbar.update(1)


        cap.release()
        out.release()
        cv2.destroyAllWindows()
    def canny_edge(self, origin_video_addr, sigma, gauss_kernel_size, th1, th2):
        cd = Image_contour_detection()
        cap = cv2.VideoCapture(origin_video_addr)

        if not cap.isOpened():
            print(f"Ошибка: не удалось открыть видео {origin_video_addr}")
            return

        # Получаем информацию о видео
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Проверка расширения файла
        file_extension = os.path.splitext(origin_video_addr)[-1].lower()
        if file_extension == '.mp4':
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        elif file_extension == '.avi':
            fourcc = cv2.VideoWriter_fourcc(*'XVID')

        dest_video_addr1, dest_video_addr2 = self.__generate_2_full_filepath(origin_video_addr, "CANNY_EDGE",
                                                        sigma=sigma, gauss_kernel_size=gauss_kernel_size, th1=th1, th2=th2)

        out1 = cv2.VideoWriter(dest_video_addr1, fourcc, fps, (frame_width, frame_height))
        out2 = cv2.VideoWriter(dest_video_addr2, fourcc, fps, (frame_width, frame_height))

        with tqdm(total=total_frames, desc="Processing video") as pbar:
            while True:
                # Читаем кадр
                ret, frame = cap.read()

                if not ret:
                    break

                processed_frame1, processed_frame2 =  cd.canny_edge_result_only(frame, sigma, gauss_kernel_size, th1, th2)

                # cv2.imshow("Live", processed_frame)
                # key = cv2.waitKey(1)
                #
                # if key == ord("q"):
                #     break

                # Конвертация кадра в 8-битный формат перед преобразованием цвета
                processed_frame1 = cv2.convertScaleAbs(processed_frame1)
                processed_frame2 = cv2.convertScaleAbs(processed_frame2)

                # Преобразование серого изображения в цветное
                processed_frame1 = cv2.cvtColor(processed_frame1, cv2.COLOR_GRAY2BGR)
                processed_frame2 = cv2.cvtColor(processed_frame2, cv2.COLOR_GRAY2BGR)

                out1.write(processed_frame1)
                out2.write(processed_frame2)
                pbar.update(1)


        cap.release()
        out1.release()
        out2.release()
        cv2.destroyAllWindows()
