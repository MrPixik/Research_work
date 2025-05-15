import mne
from app.eeg.lib.paths import *
from app.eeg.lib.eeg_channels import *
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import io


def _get_topomap_image(eeg_data, l_freq, h_freq, band):
    fig = eeg_data.plot_psd_topomap(
        bands=[(l_freq, h_freq, band)],
        normalize=True,
        ch_type='eeg',
        cmap='viridis_r',
        show=False
    )

    
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=300)
    buf.seek(0)
    plt.close(fig)
    return Image.open(buf)    

def _save_topomap_new(eeg_data, expriment_type, id):

    images = []

    for band in BANDS:
        l_freq, h_freq = BANDS[band]
        img = _get_topomap_image(eeg_data, l_freq, h_freq, band)
        images.append(img)

    # СОздание общей картинки
    widths, heights = zip(*(i.size for i in images))
    total_width = sum(widths)
    max_height = max(heights)

    combined = Image.new('RGB', (total_width, max_height), (255, 255, 255))

    x_offset = 0
    for img in images:
        combined.paste(img, (x_offset, 0))
        x_offset += img.size[0]

    # Сохраннеие
    topomap_path = get_topomap_img_path(
        Path(__file__),
        expriment_type,
        id,
    )
    combined.save(topomap_path)
    
def topomaps():
    mne.set_log_level('WARNING')
    for id in tqdm(range(1,28), desc="Topomaps creating"):
        
        experiments = [AUD_PKGNAME, AUD_NORM_PKGNAME]
        
        for experiment in experiments:
            
            # Загрузка предобработанных данных
            data_path = get_preprocessed_data_path(
                experiment, id, Path(__file__)
            )
            eeg_data = mne.io.read_raw_fif(data_path, preload=True)
            eeg_data.load_data()
            
            # Построение и сохранение топографической карты
            _save_topomap_new(eeg_data, experiment, id)