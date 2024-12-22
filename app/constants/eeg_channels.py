ALPHA = 'Alpha (8-12 Hz)'
BETA = 'Beta (12-30 Hz)'
GAMMA = 'Gamma (30-45 Hz)'
DELTA = 'Delta (0-4 Hz)'
THETA = 'Theta (4-8 Hz)'
BANDS = {DELTA: (0, 4), THETA: (4, 8),
         ALPHA: (8, 12), BETA: (12, 30),
         GAMMA: (30, 45)}

TOPOMAP_ORIGIN_PATH = {ALPHA: r'../../static/media/topomap_images/original/alpha',
                       BETA: r'../../static/media/topomap_images/original/beta',
                       GAMMA: r'../../static/media/topomap_images/original/gamma',
                       DELTA: r'../../static/media/topomap_images/original/delta',
                       THETA: r'../../static/media/topomap_images/original/theta'}

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

TOPOMAP_DIAM_PIXELS = 340
