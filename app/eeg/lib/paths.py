from pathlib import Path
from app.eeg.lib.eeg_channels import *

AUD_PACKAGE_NAME = 'aud'
AUD_NORM_PACKAGE_NAME = 'aud-norm'
DATASETS_NUM = 28

TOPOMAP_ORIGIN_PATH = {ALPHA: r'../../static/media/topomap_images/original/alpha',
                       BETA: r'../../static/media/topomap_images/original/beta',
                       GAMMA: r'../../static/media/topomap_images/original/gamma',
                       DELTA: r'../../static/media/topomap_images/original/delta',
                       THETA: r'../../static/media/topomap_images/original/theta',
                       CUSTOM: r'../../static/media/topomap_images/original/custom'}

TOPOMAP_PROCESSED_PATH = {ALPHA: r'../../static/media/topomap_images/processed/alpha',
                          BETA: r'../../static/media/topomap_images/processed/beta',
                          GAMMA: r'../../static/media/topomap_images/processed/gamma',
                          DELTA: r'../../static/media/topomap_images/processed/delta',
                          THETA: r'../../static/media/topomap_images/processed/theta'}

TOPOMAP_VIDEO_PATH = {ALPHA: r'../../static/media/topomap_video',
                      BETA: r'../../static/media/topomap_video',
                      GAMMA: r'../../static/media/topomap_video',
                      DELTA: r'../../static/media/topomap_video',
                      THETA: r'../../static/media/topomap_video'}


def get_preprocessed_data_path(package_name: str, num: int, current_file_path: Path, marker: str = "static") -> Path:
    """
    Возвращает путь до файла PREPROCESSED_EEG_DATA_PATH, автоматически находя корневую папку проекта.

    :param package_name: Имя папки (aud или aud-norm)
    :param num: Номер файла
    :param current_file_path: Путь к текущему исполняемому файлу (обычно Path(__file__)).
    :param marker: Папка-маркер, внутри которой лежит raw-data
    :return: Полный путь до файла RAW_EEG_DATA_PATH.
    """

    # Определяем корневую папку проекта
    project_root = find_project_root(current_file_path.resolve(), marker)

    file_name = str(num) + '.fif'

    # Формируем путь до файла с данными
    preprocessed_eeg_data_path = project_root / 'static' / 'eeg_data' / 'preprocessed' / package_name / file_name

    return preprocessed_eeg_data_path


def get_raw_data_path(package_name: str, num: int, current_file_path: Path, marker: str = "static") -> Path:
    """
    Возвращает путь до файла RAW_EEG_DATA_PATH, автоматически находя корневую папку проекта.

    :param package_name: Имя папки (aud или aud-norm)
    :param num: Номер файла
    :param current_file_path: Путь к текущему исполняемому файлу (обычно Path(__file__)).
    :param marker: Папка-маркер, внутри которой лежит raw-data
    :return: Полный путь до файла RAW_EEG_DATA_PATH.
    """
    # Определяем корневую папку проекта
    project_root = find_project_root(current_file_path.resolve(), marker)
    
    file_name = str(num) + '.set'

    # Формируем путь до файла с данными
    preprocessed_eeg_data_path = project_root / 'static' / 'eeg_data' / 'raw' / package_name / file_name

    return preprocessed_eeg_data_path


def get_topomap_path(key):
    # Если ключ есть в словаре, возвращаем соответствующее значение
    if key in TOPOMAP_ORIGIN_PATH:
        return TOPOMAP_ORIGIN_PATH[key]
    # Если ключа нет, возвращаем значение для CUSTOM
    else:
        return TOPOMAP_ORIGIN_PATH['CUSTOM']


def find_project_root(current_path: Path, marker: str = "README.md") -> Path:
    """
    Ищет корневую папку проекта, поднимаясь вверх по дереву директорий,
    пока не найдет файл-маркер (например, README.md).
    """
    for parent in current_path.parents:
        if (parent / marker).exists():
            return parent
    raise FileNotFoundError(f"Корневая папка проекта не найдена. Маркер: {marker}")


