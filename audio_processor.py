import librosa
import numpy as np
from math import floor

def compute_melgram(audio_path):
    ''' Compute a mel-spectrogram and returns it in a shape of (1,1,96,1366), where
    96 == #mel-bins and 1366 == #time frame
    '''
    src, sr = librosa.load(audio_path, sr=12000)
    n_sample = src.shape[0]
    n_sample_fit = int(29.12*12000)

    if n_sample < n_sample_fit:
        src = np.hstack((src, np.zeros((int(29.12*12000) - n_sample,))))
    elif n_sample > n_sample_fit:
        src = src[(n_sample-n_sample_fit)/2:(n_sample+n_sample_fit)/2]
    logam = librosa.logamplitude
    melgram = librosa.feature.melspectrogram
    ret = logam(melgram(y=src, sr=12000, hop_length=256,
                        n_fft=512, n_mels=96)**2,
                ref_power=1.0)
    ret = ret[np.newaxis, np.newaxis, :]
    return ret


def compute_melgram_multiframe(audio_path, all_song=True):
    if all_song:
        DURA_TRASH = 0
    else:
        DURA_TRASH = 20

    src, sr = librosa.load(audio_path, sr=12000)
    n_sample = src.shape[0]
    n_sample_fit = int(29.12*12000)
    n_sample_trash = int(DURA_TRASH*12000)

    src = src[n_sample_trash:(n_sample-n_sample_trash)]
    n_sample=n_sample-2*n_sample_trash

    ret = np.zeros((0, 1, 96, 1366), dtype=np.float32)

    if n_sample < n_sample_fit:
        src = np.hstack((src, np.zeros((int(29.12*12000) - n_sample,))))
        logam = librosa.logamplitude
        melgram = librosa.feature.melspectrogram
        ret = logam(melgram(y=src, sr=12000, hop_length=256,
                            n_fft=512, n_mels=96)**2,
                    ref_power=1.0)
        ret = ret[np.newaxis, np.newaxis, :]

    elif n_sample > n_sample_fit:
        N=int(floor(n_sample/n_sample_fit))

        src_total=src

        for i in range(0,N):
            src = src_total[(i*n_sample_fit):(i+1)*(n_sample_fit)]

            logam = librosa.logamplitude
            melgram = librosa.feature.melspectrogram
            retI = logam(melgram(y=src, sr=12000, hop_length=256,
                                n_fft=512, n_mels=96)**2,
                        ref_power=1.0)
            retI = retI[np.newaxis, np.newaxis, :]
            ret = np.concatenate((ret, retI), axis=0)

    return ret