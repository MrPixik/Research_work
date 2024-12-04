import cv2
import time
from tqdm import tqdm
import numpy as np
import pandas as pd
import os
from app.models.Image_contour_detection import *

class Video_contour_detection:
    sobel_save_path = r'static/videos/result/sobel_edge'
    mh_save_path = r'static/videos/result/marr_hildreth'
    canny_save_path = r'static/videos/result/canny_edge'

    def __get_video_information(self, cap, video_path):
        # Получаем информацию о видео
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_size = (frame_width, frame_height)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Проверка расширения файла
        file_extension = os.path.splitext(video_path)[-1].lower()
        if file_extension == '.mp4':
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        elif file_extension == '.avi':
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
        return frame_size, fps, total_frames, fourcc
    def __generate_filepaths(self, origin_video_addr, method_name, num_files=1, **kwargs):
        path = {
            "SOBEL_EDGE": self.sobel_save_path,
            "MARR_HILDRETH": self.mh_save_path,
            "CANNY_EDGE": self.canny_save_path
        }.get(method_name)

        if path is None:
            raise ValueError(f"Unknown method name: {method_name}")

        base_name = os.path.splitext(os.path.basename(origin_video_addr))[0]
        params_str = "_".join([f"{k}={v}" for k, v in kwargs.items()])

        filenames = [f"{base_name}_{method_name}_{i + 1}_{params_str}.mp4" for i in range(num_files)]
        return [os.path.join(path, filename) for filename in filenames]
    def sobel_edges(self, origin_video_addr, kernel_size, threshold):
        cd = Image_contour_detection()
        cap = cv2.VideoCapture(origin_video_addr)

        if not cap.isOpened():
            print(f"Ошибка: не удалось открыть видео {origin_video_addr}")
            return

        frame_size, fps, total_frames, fourcc = self.__get_video_information(cap, origin_video_addr)

        dest_video_addr = self.__generate_filepaths(origin_video_addr, "SOBEL_EDGE",
                                                    kernel_size=kernel_size, threshold=threshold)[0]
        out = cv2.VideoWriter(dest_video_addr, fourcc, fps, frame_size)

        with tqdm(total=total_frames, desc="Processing video") as pbar:
            while True:
                ret, frame = cap.read()

                if not ret:
                    break


                processed_frame = cd.sobel_edge_detection(frame, kernel_size, cd.thresholdAdaptive)

                if len(processed_frame.shape) == 2:  # Если изображение имеет два измерения (градации серого)
                    processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_GRAY2BGR)

                out.write(processed_frame)
                pbar.update(1)

        cap.release()
        out.release()
        cv2.destroyAllWindows()
    def marr_hildreth_edges(self, origin_video_addr, sigma, threshold):
        cd = Image_contour_detection()
        cap = cv2.VideoCapture(origin_video_addr)

        if not cap.isOpened():
            print(f"Ошибка: не удалось открыть видео {origin_video_addr}")
            return

        frame_size, fps, total_frames, fourcc = self.__get_video_information(cap, origin_video_addr)

        dest_video_addr = self.__generate_filepaths(origin_video_addr, "MARR_HILDRETH",
                                                        sigma=sigma, threshold=threshold)[0]

        out = cv2.VideoWriter(dest_video_addr, fourcc, fps, frame_size)

        with tqdm(total=total_frames, desc="Processing video") as pbar:
            while True:
                # Читаем кадр
                ret, frame = cap.read()

                if not ret:
                    break

                processed_frame = cd.marr_hildreth_edge_detection(frame,sigma,threshold)

                # Записываем обработанный кадр в выходное видео
                if len(processed_frame.shape) == 2:  # Если изображение имеет два измерения (градации серого)
                    processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_GRAY2BGR)

                out.write(processed_frame)
                pbar.update(1)


        cap.release()
        out.release()
        cv2.destroyAllWindows()
    def canny_edges(self, origin_video_addr, aperture_size, th1, th2):
        cd = Image_contour_detection()
        cap = cv2.VideoCapture(origin_video_addr)

        if not cap.isOpened():
            print(f"Ошибка: не удалось открыть видео {origin_video_addr}")
            return

        frame_size, fps, total_frames, fourcc = self.__get_video_information(cap, origin_video_addr)

        dest_video_addr = self.__generate_filepaths(origin_video_addr, "CANNY_EDGE",
                                                        aperture_size=aperture_size, th1=th1, th2=th2)[0]

        out = cv2.VideoWriter(dest_video_addr, fourcc, fps, frame_size)


        with tqdm(total=total_frames, desc="Processing video") as pbar:
            while True:
                # Читаем кадр
                ret, frame = cap.read()

                if not ret:
                    break

                processed_frame1= cd.canny_edge_detection(frame, aperture_size, th1, th2)

                # Конвертация кадра в 8-битный формат перед преобразованием цвета
                processed_frame1 = cv2.convertScaleAbs(processed_frame1)

                # Преобразование серого изображения в цветное
                processed_frame1 = cv2.cvtColor(processed_frame1, cv2.COLOR_GRAY2BGR)

                out.write(processed_frame1)
                pbar.update(1)


        cap.release()
        out.release()
        cv2.destroyAllWindows()
