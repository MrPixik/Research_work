from app.eeg.analysis.analyzer import *
from pathlib import Path
# from app.eeg.lib.eeg_channels import *
from app.eeg.processor.preprocessor import *
from mne.datasets import fetch_fsaverage
from app.eeg.reverse_problem.reverse_problem import *


if __name__ == '__main__':
    mne.set_log_level('ERROR')
    # Preprocessing
    # preprocess_aud()
    # preprocess_aud_norm()
    
    # Reverse problem solution
    for i in range(1,28):
        reverse_problem_for_all_bands(i)
    
    
    # stcs_avg = reverce_problem_for_inactive_zones()

    # subject='fsaverage'
    # fs_dir = Path(fetch_fsaverage(verbose=True))
    # subjects_dir = str(fs_dir.parent)  # <- именно на папку выше

    # mne.viz.set_3d_backend('pyvistaqt')
    # for idx, stc_avg in enumerate(stcs_avg):
    #     brain = stc_avg.plot(
    #     subject=subject,
    #     subjects_dir=subjects_dir,
    #     hemi='both',
    #     surface='white',
    #     background='white',
    #     cortex='low_contrast',
    #     )

    #     # Сохраняем изображение сразу в файл
    #     brain.save_image(f'component_{idx+1:02d}.png')
        # print(f"Построение компоненты {idx + 1}/{len(stcs_avg)}")
        # stc_avg.plot(
        #     subject=subject,
        #     subjects_dir=subjects_dir,
        #     hemi='both',            # два полушария
        #     surface='white',
        #     time_label='Средняя активность',
        #     title=f'ICA-компонента {idx+1} (усредн.)',
        #     size=(800, 600),        # размер окна
        #     background='white',     # фон
        #     cortex='low_contrast'   # отображение коры
        # )
        # input("Нажмите Enter, чтобы закрыть окно и выйти")
        
    
    
    
    
    
    # print_all_data_time()
    # save_topomap_by_time(title=ALPHA, freq_lims=(16, 22), update_interval=0.2, threshold=1, normalize=True)
    # EEG_analyzer.save_topomap_by_time(band=BETA, update_interval=0.02)
    # EEG_analyzer.save_topomap_by_time(band=GAMMA, update_interval=0.02)
    # EEG_analyzer.save_topomap_by_time(band=DELTA, update_interval=0.02)
    # EEG_analyzer.save_topomap_by_time(band=THETA, update_interval=0.02)


    # Topomap_analyzer.create_video_of_inactive_areas(band=ALPHA)
    # Topomap_analyzer.create_video_of_inactive_areas(band=BETA)
    # Topomap_analyzer.create_video_of_inactive_areas(band=GAMMA)
    # Topomap_analyzer.create_video_of_inactive_areas(band=DELTA)
    # Topomap_analyzer.create_video_of_inactive_areas(band=THETA)

    # EEG_analyzer.show_topomap()