from app.eeg.processor.preprocessor import *
from app.eeg.reverse_problem.reverse_problem import *
from app.eeg.topomap.topomap import *


if __name__ == '__main__':
    
    # Preprocessing
    # preprocess_aud()
    # preprocess_aud_norm()
    
    topomaps()

    brodmann()
    # reverse_problem()

