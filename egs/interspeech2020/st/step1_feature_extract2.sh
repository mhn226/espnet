#!/bin/bash

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

# general configuration
backend=pytorch # chainer or pytorch
stage=-1        # start from -1 if you need to start from data download
stop_stage=100
ngpu=1          # number of gpus ("0" uses cpu, otherwise use gpu)
debugmode=1
dumpdir=dump    # directory to dump full features
N=0             # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
verbose=0       # verbose option
resume=         # Resume the training from snapshot
seed=1          # seed to generate random number
# feature configuration
do_delta=false

train_config=conf/train.yaml
decode_config=conf/decode.yaml

# preprocessing related
case=tc
# tc: truecase
# lc: lowercase
# lc.rm: lowercase with punctuation removal

# Set this to somewhere where you want to put your data, or where
# someone else has already put it.  You'll want to change this
# if you're not on the CLSP grid.
#st_ted=/export/b08/inaguma/IWSLT
# st_ted=/n/rd11/corpora_8/iwslt18

# bpemode (unigram or bpe)
nbpe=134
#bpemode=bpe
# NOTE: nbpe=88 means character-level ST (lc.rm)
# NOTE: nbpe=106 means character-level ST (lc)
# NOTE: nbpe=134 means character-level ST (tc)

# exp tag
tag="" # tag for managing experiments.

. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=train_sp.fr
train_set_prefix=train_sp
train_dev=train_dev.fr
recog_set="mustc-tst-COMMON.fr mustc-tst-HE.fr"

feat_tr_dir=${dumpdir}/${train_set}; mkdir -p ${feat_tr_dir}
feat_dt_dir=${dumpdir}/${train_dev}; mkdir -p ${feat_dt_dir}
wav2vec_model_path=/home/getalp/nguyen35/fairseq_exp/wav2vec_large.pt

export PYTHONPATH=/home/getalp/nguyen35/fairseq

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    ### Task dependent. You have to design training and dev sets by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 1: Feature Generation"
    wav2vecdir=wav2vec

#    for x in mustc-dev mustc-tst-COMMON mustc-tst-HE; do
    for x in mustc-dev; do
        #######################################################
        ##################### WAV2VEC #########################
        #######################################################
        # extract audio segments for wav2vec script
        ./extract_segments.sh --nj 16 --write_utt2num_frames true data/${x} exp/make_wav2vec/${x} ${wav2vecdir}
        # Generate raw wav2vec features
        ./make_wav2vec_features.sh --nj 16 --write_utt2num_frames true --wav2vec-model-path ${wav2vec_model_path} data/${x} exp/make_wav2vec/${x} ${wav2vecdir} 
    done

    # speed-perturbed
#    utils/perturb_data_dir_speed.sh 0.9 data/mustc-train data/temp1
#    utils/perturb_data_dir_speed.sh 1.0 data/mustc-train data/temp2
#    utils/perturb_data_dir_speed.sh 1.1 data/mustc-train data/temp3
#    utils/combine_data.sh --extra-files utt2uniq data/train_sp data/temp1 data/temp2 data/temp3
#    rm -r data/temp1 data/temp2 data/temp3


    #######################################################
    ##################### WAV2VEC #########################
    #######################################################
    # extract audio segments for wav2vec script
#    ./extract_segments.sh --nj 16 --write_utt2num_frames true data/train_sp exp/make_wav2vec/train_sp ${wav2vecdir}
    # compute wav2vec features
#    ./make_wav2vec_features.sh --nj 16 --write_utt2num_frames true --wav2vec-model-path ${wav2vec_model_path} data/train_sp exp/make_wav2vec/train_sp ${wav2vecdir}



#    for lang in en fr; do
#        awk -v p="sp0.9-" '{printf("%s %s%s\n", $1, p, $1);}' data/mustc-train/utt2spk > data/train_sp/utt_map
#        utils/apply_map.pl -f 1 data/train_sp/utt_map <data/mustc-train/text.tc.${lang} >data/train_sp/text.tc.${lang}
#        awk -v p="sp1.0-" '{printf("%s %s%s\n", $1, p, $1);}' data/mustc-train/utt2spk > data/train_sp/utt_map
#        utils/apply_map.pl -f 1 data/train_sp/utt_map <data/mustc-train/text.tc.${lang} >>data/train_sp/text.tc.${lang}
#        awk -v p="sp1.1-" '{printf("%s %s%s\n", $1, p, $1);}' data/mustc-train/utt2spk > data/train_sp/utt_map
#        utils/apply_map.pl -f 1 data/train_sp/utt_map <data/mustc-train/text.tc.${lang} >>data/train_sp/text.tc.${lang}
#    done

    # Divide into source and target languages
#    for x in train_sp mustc-dev mustc-tst-COMMON mustc-tst-HE; do
#        local/divide_lang_fr.sh ${x}
#    done

#    for lang in en fr; do
#        if [ -d data/train_dev.${lang} ];then
#            rm -rf data/train_dev.${lang}
#        fi
#        cp -rf data/mustc-dev.${lang} data/train_dev.${lang}
#    done

#    for x in ${train_set_prefix} train_dev; do
        # remove utt having more than 3000 frames
        # remove utt having more than 400 characters
#        for lang in en fr; do
#            remove_longshortdata.sh --maxframes 3000 --maxchars 400 data/${x}.${lang} data/${x}.${lang}.tmp
#        done

        # Match the number of utterances between source and target languages
        # extract commocn lines
#        cut -f 1 -d " " data/${x}.en.tmp/text > data/${x}.fr.tmp/reclist1
#        cut -f 1 -d " " data/${x}.fr.tmp/text > data/${x}.fr.tmp/reclist2
#        comm -12 data/${x}.fr.tmp/reclist1 data/${x}.fr.tmp/reclist2 > data/${x}.fr.tmp/reclist

#        for lang in en fr; do
#            reduce_data_dir.sh data/${x}.${lang}.tmp data/${x}.fr.tmp/reclist data/${x}.${lang}
#            utils/fix_data_dir.sh --utt_extra_files "text.tc" data/${x}.${lang}
#        done
#        rm -rf data/${x}.*.tmp
#    done


    # Normalization 
    #compute-cmvn-stats scp:data/${train_set}/feats.scp data/${train_set}/cmvn.ark

    #dump.sh --cmd "$train_cmd" --nj 32 --do_delta $do_delta \
    #    data/${train_set}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/${train_set} ${feat_tr_dir}
    #dump.sh --cmd "$train_cmd" --nj 32 --do_delta $do_delta \
    #    data/${train_dev}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/${train_dev} ${feat_dt_dir}
    #for rtask in ${recog_set}; do
    #    feat_recog_dir=${dumpdir}/${rtask}; mkdir -p ${feat_recog_dir}
    #    dump.sh --cmd "$train_cmd" --nj 16 --do_delta $do_delta \
    #        data/${rtask}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/recog/${rtask} \
    #        ${feat_recog_dir}
    #done


fi

