import torch
from fairseq.models.wav2vec import Wav2VecModel
import librosa
import numpy as np
import kaldiio
from kaldiio import ReadHelper
from scipy.io import wavfile
from collections import OrderedDict
import os
import sys
from decimal import Decimal

wav2vec_model_path = sys.argv[1]
segment_file = sys.argv[2]
ark_file = sys.argv[3]
wav2vec_ark = sys.argv[4]
wav2vec_scp = sys.argv[5]
write_utt2dur_opt = sys.argv[6]
write_num_frames_opt = sys.argv[7]
logdir = sys.argv[8]
JOB = sys.argv[9]

cp = torch.load(wav2vec_model_path, map_location='cuda:0')
model = Wav2VecModel.build_model(cp['args'], task=None)
model.load_state_dict(cp['model'])
model.eval()

def extract_features(ark_file):
    od = OrderedDict()
    d = kaldiio.load_ark(ark_file)
    num_frames = []
    for key, (rate, numpy_array) in d:
        data = torch.Tensor(numpy_array).view(1, -1)
        z = model.feature_extractor(data)
        c = model.feature_aggregator(z)
        c = c.squeeze().transpose(0, 1)
        num_frames.append(key + ' ' + str(c.size(0)) + ' \n')
        a = c.detach().numpy()
        od[key] = a
    return od, num_frames

def utt2dur(segment_file):
    wlines = []
    with open(segment_file, 'r') as fr:
        for line in fr:
            line = line.split()
            file_id = line[0]
            dur = Decimal(line[-1]) - Decimal(line[-2])
            dur = str(dur)
            dur = dur.rstrip('0').rstrip('.') if '.' in dur else dur
            wlines.append(file_id + ' ' + dur + ' \n')
    with open(logdir + '/utt2dur.' + JOB, 'w') as fw:
        fw.writelines(wlines)

def main():
    od, num_frames = extract_features(ark_file)
    kaldiio.save_ark(wav2vec_ark, od, scp=wav2vec_scp)
    if write_utt2dur_opt:
        utt2dur(segment_file)
    if write_num_frames_opt:
        with open(logdir + '/utt2num_frames.' + JOB, 'w') as fw:
            fw.writelines(num_frames)
    os.remove(ark_file)

if __name__ == "__main__":
    main()
