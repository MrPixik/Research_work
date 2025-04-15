# from app.eeg.analysis.EEG_analyzer import *
# from app.eeg.lib.eeg_channels import *
from app.eeg.processor.preprocessor import *


if __name__ == '__main__':

    preprocess_aud()
    preprocess_aud_norm()
    # print_all_data_time()
    # EEG_analyzer.save_topomap_by_time(title=ALPHA, freq_lims=(8, 12), update_interval=0.02, threshold=0.5)
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