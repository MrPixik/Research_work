from app.encefalogram.EEGPreprocessor import *
from app.encefalogram.DataAnalyzer import *

if __name__ == '__main__':
    # DataAnalyzer.show_topomap()
    bands = [DataAnalyzer.alpha]
    # DataAnalyzer.plot_topomap_by_band(bands)
    DataAnalyzer.save_topomap_by_time(bands, save_images=True, output_dir=r'../../static/media/topomap_images')

