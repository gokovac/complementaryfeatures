# -*- coding: utf-8 -*-
import numpy as np
import librosa

def extract_fft_l_a2en(read_audio_path):

    x, fs = librosa.load(read_audio_path + '.flac',sr=16000) 
    S = np.abs(librosa.stft(x, n_fft=1728, hop_length=130, win_length=1728, window='blackman')) #any 2-D representation can be used to extract reg. energy feats.
    # X=np.array([3,5,7,11,13,15,20,25]) #combination of prime numbers is possible
    X=np.array([15]) #prime numbers
    rows, cols = S.shape
    total_size_f = np.sum(np.power(X,2), axis=0)
    reg_en = np.zeros([total_size_f])
    jump, count = 0, 0
    for k in range(0, len(X)):
                # time-freq. resolution can be adjusted differently (in that case, cols and rows should be divided by different numbers)
                L1 = int(np.floor(cols/X[k]))
                L2 = int(np.floor(rows/X[k]))
                num_win = X[k] ** 2
                Nn, Mn = rows, cols
                partsM = np.arange(0, Mn, L1)
                partsN = np.arange(0, Nn, L2)
                partsM = partsM[ :X[k]]
                partsN = partsN[ :X[k]]

                for j in range(0, len(partsN)):
                    for ix in range(0, len(partsM)):
                        reg_en[jump] = np.sum(np.sum(S[partsN[j]:partsN[j]+L2,partsM[ix]:partsM[ix]+L1]**2))
                        jump = jump + 1
                        
                z = num_win
                reg_en[count:count+z] = reg_en[count:count+z] / np.sum(reg_en[count:count+z])
                count = count +z
    
    S = librosa.amplitude_to_db(S, ref=1.0, amin=1e-30, top_db=None)
    
    return S, reg_en
