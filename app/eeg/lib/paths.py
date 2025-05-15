from pathlib import Path
from app.eeg.lib.eeg_channels import *


# Константы для эксперемента
DATASETS_NUM = 27

# Константы названий директорий
APP_PKG_MARKER      = 'static'
AUD_PKGNAME         = 'aud'
AUD_NORM_PKGNAME    = 'aud-norm'

# Константы названий файлов
# BEM_SOLUTION_FILENAME       = 'fsaverage-bem.fif'
# SOURCE_SPACES_FILENAME      = 'fsaverage-src.fif'
# FORWARD_SOLUTION_FILENAME   = 'fsaverage-fwd.fif'
INVERSE_OPERATOR_FILENAME   = 'fsaverage-inv.fif'

_TOPOMAP_ORIGIN_PACKAGE_NAME =  {ALPHA: r'alpha',
                                BETA1: r'beta',
                                GAMMA: r'gamma',
                                DELTA: r'delta',
                                THETA: r'theta',
                                CUSTOM: r'custom'}

TOPOMAP_PROCESSED_PATH = {ALPHA: r'../../static/media/topomap_images/processed/alpha',
                          BETA1: r'../../static/media/topomap_images/processed/beta',
                          GAMMA: r'../../static/media/topomap_images/processed/gamma',
                          DELTA: r'../../static/media/topomap_images/processed/delta',
                          THETA: r'../../static/media/topomap_images/processed/theta'}

TOPOMAP_VIDEO_PATH = {ALPHA: r'../../static/media/topomap_video',
                      BETA1: r'../../static/media/topomap_video',
                      GAMMA: r'../../static/media/topomap_video',
                      DELTA: r'../../static/media/topomap_video',
                      THETA: r'../../static/media/topomap_video'}


def find_parent_dir(current_path: Path, marker: str = "README.md") -> Path:
    """
    Ищет корневую папку проекта, поднимаясь вверх по дереву директорий,
    пока не найдет файл-маркер (например, README.md).
    """
    for parent in current_path.parents:
        if (parent / marker).exists():
            return parent
    raise FileNotFoundError(f"Корневая папка проекта не найдена. Маркер: {marker}")


def get_raw_data_path(experiment_name: str, num: int, current_file_path: Path) -> Path:
    """
    Возвращает путь до файла RAW_EEG_DATA_PATH, автоматически находя корневую папку проекта.

    :param package_name: Имя папки (aud или aud-norm)
    :param num: Номер файла
    :param current_file_path: Путь к текущему исполняемому файлу (обычно Path(__file__)).
    :return: Полный путь до файла RAW_EEG_DATA_PATH.
    """
    # Определяем корневую папку проекта
    project_root = find_parent_dir(current_file_path.resolve(), APP_PKG_MARKER)
    
    file_name = str(num) + '.set'

    # Формируем путь до файла с данными
    preprocessed_eeg_data_path = project_root / 'static' / 'eeg_data' / 'raw' / experiment_name / file_name

    return preprocessed_eeg_data_path


def get_preprocessed_data_path(experiment_name: str, num: int, current_file_path: Path) -> Path:
    """
    Возвращает путь до файла PREPROCESSED_EEG_DATA_PATH, автоматически находя корневую папку проекта.

    :param package_name: Имя папки (aud или aud-norm)
    :param num: Номер файла
    :param current_file_path: Путь к текущему исполняемому файлу (обычно Path(__file__)).
    :return: Полный путь до файла RAW_EEG_DATA_PATH.
    """

    # Определяем корневую папку проекта
    project_root = find_parent_dir(current_file_path.resolve(), APP_PKG_MARKER)

    file_name = str(num) + '_eeg.fif'

    # Формируем путь до файла с данными
    preprocessed_eeg_data_path = project_root / 'static' / 'eeg_data' / 'preprocessed' / experiment_name / file_name

    return preprocessed_eeg_data_path


def get_topomap_img_path(current_file_path: Path, experiment_type: str,  id: int)-> Path:
    """Возвращает путь до папки с изображениями топографических карт по ключу

    :param key: Название канала (ALPHA,BETA,GAMMA... из пакета app.eeg.lib.eeg_channels)
    :param current_file_path: Путь к текущему исполняемому файлу (обычно Path(__file__)).
    :return: Полный путь до папки с топографическими картами.
    """
     # Определяем корневую папку проекта
    app_pkg = find_parent_dir(current_file_path.resolve(), APP_PKG_MARKER)
    
    file_name = 'experiment_N' + str(id) + '_' + '.png'
    
    topomap_img_path = app_pkg / 'static' / 'results' / 'topomaps' / experiment_type  / file_name 
    return topomap_img_path
    

def get_solution_data_path_by_band(current_file_path: Path, id: int, experiment_type: str, band: str, component_id: int) -> Path:
    app_pkg = find_parent_dir(current_file_path.resolve(), APP_PKG_MARKER)

    file_name = str(component_id) + '.png'

    solution_data_path = app_pkg / 'static' / 'results' / 'reverse_solution' / str(id) / experiment_type / band / file_name

    return solution_data_path

def get_brodmann_data_path(current_file_path: Path, id: int, experiment_type: str, band: str) -> Path:
    app_pkg = find_parent_dir(current_file_path.resolve(), APP_PKG_MARKER)

    file_name = band + '.xlsx'

    brodmann_data_path = app_pkg / 'static' / 'results' / 'brodmann' / str(id) / experiment_type / file_name

    return brodmann_data_path

# def get_bem_solution_path(current_file_path: Path) -> Path:
#     # Определяем корневую папку проекта
#     app_pkg = find_parent_dir(current_file_path.resolve(), APP_PKG_MARKER)

#     file_name = BEM_SOLUTION_FILENAME

#     # Формируем путь до файла с данными static/reverse_solution/{id}/{experiment_type}/{band}
#     bem_solution_path = app_pkg / 'static' / 'reverse_problem' / file_name

#     return bem_solution_path

# def get_source_spaces_path(current_file_path: Path) -> Path:
#     # Определяем корневую папку проекта
#     app_pkg = find_parent_dir(current_file_path.resolve(), APP_PKG_MARKER)

#     file_name = SOURCE_SPACES_FILENAME

#     # Формируем путь до файла с данными static/reverse_solution/{id}/{experiment_type}/{band}
#     source_spaces_path = app_pkg / 'static' / 'reverse_problem' / file_name

#     return source_spaces_path

# def get_forward_solution_path(current_file_path: Path) -> Path:
#     # Определяем корневую папку проекта
#     app_pkg = find_parent_dir(current_file_path.resolve(), APP_PKG_MARKER)

#     file_name = FORWARD_SOLUTION_FILENAME

#     # Формируем путь до файла с данными static/reverse_solution/{id}/{experiment_type}/{band}
#     forward_solution_path = app_pkg / 'static' / 'reverse_problem' / file_name

#     return forward_solution_path

def get_inverse_operator_path(current_file_path: Path) -> Path:
    # Определяем корневую папку проекта
    app_pkg = find_parent_dir(current_file_path.resolve(), APP_PKG_MARKER)

    file_name = INVERSE_OPERATOR_FILENAME

    # Формируем путь до файла с данными static/reverse_solution/{id}/{experiment_type}/{band}
    inverse_operator_path = app_pkg / 'static' / 'reverse_problem' / file_name

    return inverse_operator_path

