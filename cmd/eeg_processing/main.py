from app.encefalogram.EEGPreprocessor import *
from app.encefalogram.DataAnalyzer import *
from app.models.Topomap_analysis import *
from app.constants.eeg_channels import *

if __name__ == '__main__':
    # DataAnalyzer.save_topomap_by_time(band=ALPHA, update_interval=0.02)
    # DataAnalyzer.save_topomap_by_time(band=BETA, update_interval=0.02)
    # DataAnalyzer.save_topomap_by_time(band=GAMMA, update_interval=0.02)
    # DataAnalyzer.save_topomap_by_time(band=DELTA, update_interval=0.02)
    DataAnalyzer.save_topomap_by_time(band=THETA, update_interval=0.02)


    # Topomap_analysis.create_video_of_inactive_areas(band=ALPHA)
    # Topomap_analysis.create_video_of_inactive_areas(band=BETA)
    # Topomap_analysis.create_video_of_inactive_areas(band=GAMMA)
    # Topomap_analysis.create_video_of_inactive_areas(band=DELTA)
    # Topomap_analysis.create_video_of_inactive_areas(band=THETA)