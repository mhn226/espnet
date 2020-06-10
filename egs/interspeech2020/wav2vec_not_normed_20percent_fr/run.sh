#!/bin/bash

# Copyright 2019 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

# general configuration
backend=pytorch # chainer or pytorch
stage=5        # start from -1 if you need to start from data download
stop_stage=5
ngpu=2          # number of gpus ("0" uses cpu, otherwise use gpu)
debugmode=1
dumpdir=dump    # directory to dump full features
N=0             # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
verbose=0       # verbose option
resume=         # Resume the training from snapshot
seed=1          # seed to generate random number
# feature configuration
do_delta=false
cmvn=false

preprocess_config=
train_config=conf/train.yaml
decode_config=conf/decode.yaml

# decoding parameter
trans_model=model.acc.best # set a model to be used for decoding: 'model.acc.best' or 'model.loss.best'

# model average realted (only for transformer)
n_average=5                  # the number of ST models to be averaged
use_valbest_average=true     # if true, the validation `n_average`-best ST models will be averaged.
                             # if false, the last `n_average` ST models will be averaged.

# pre-training related
asr_model=
mt_model=

# type of acoustic features
wav2vec=true
idim_reduction=true
feature_type=wav2vec
wav2vec_pretrained_model=/home/getalp/nguyen35/fairseq_exp/wav2vec_large.pt

# preprocessing related
case=tc
# tc: truecase
# lc: lowercase
# lc.rm: lowercase with punctuation removal

# Set this to somewhere where you want to put your data, or where
# someone else has already put it.
speech_trans_corpus=/lium/corpus/speechtrans/ldc_aren

# source language related
src_lang=en
# target language related
tgt_lang=fr

# bpemode (unigram or bpe)
nbpe=""
bpemode=char

# exp tag
tag="" # tag for managing experiments.

. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=train_sp.fr
train_dev=train_dev.fr
train_set_prefix=train_sp
trans_set="mustc-tst-COMMON.fr mustc-tst-HE.fr"


feat_tr_dir=${dumpdir}/${train_set}/delta${do_delta}; mkdir -p ${feat_tr_dir}
feat_dt_dir=${dumpdir}/${train_dev}/delta${do_delta}; mkdir -p ${feat_dt_dir}
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    ### Task dependent. You have to design training and dev sets by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 1: Feature Generation"

    # speed-perturbed
    utils/perturb_data_dir_speed.sh 0.9 data/mustc-train data/temp1
    utils/perturb_data_dir_speed.sh 1.0 data/mustc-train data/temp2
    utils/perturb_data_dir_speed.sh 1.1 data/mustc-train data/temp3
    utils/combine_data.sh --extra-files utt2uniq data/train_sp data/temp1 data/temp2 data/temp3
    rm -r data/temp1 data/temp2 data/temp3

    if [ ${feature_type} = "wav2vec"  ]; then
        export PYTHONPATH=/home/getalp/nguyen35/fairseq
        echo "Feature type : pretrained wav2vec features (need gpu)"
        wav2vecdir=wav2vec_features
        mkdir -p ${wav2vecdir}
        for x in mustc-dev mustc-tst-COMMON mustc-tst-HE train_sp; do
            echo "Extracting segments from ${x}"
                                                            #  $1 (data)        $2 (model)           $3 (logdir)          $4 (features dir)
            local/make_wav2vec.sh --cmd "$train_cmd" --nj 16 --gpu ${ngpu} data/${x} ${wav2vec_pretrained_model} exp/make_wav2vec/${x}  ${wav2vecdir}
        done

    else
        fbankdir=fbank
        echo "Feature type : 80-dimensional fbank"
        # Generate the fbank features; by default 80-dimensional fbanks with pitch on each frame
        for x in mustc-dev mustc-tst-COMMON mustc-tst-HE train_sp; do
            steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj 32 --write_utt2num_frames true \
                data/${x} exp/make_fbank/${x} ${fbankdir}
        done
    fi

    for lang in ${src_lang} ${tgt_lang}; do
        awk -v p="sp0.9-" '{printf("%s %s%s\n", $1, p, $1);}' data/mustc-train/utt2spk > data/train_sp/utt_map
        utils/apply_map.pl -f 1 data/train_sp/utt_map <data/mustc-train/text.tc.${lang} >data/train_sp/text.tc.${lang}
        awk -v p="sp1.0-" '{printf("%s %s%s\n", $1, p, $1);}' data/mustc-train/utt2spk > data/train_sp/utt_map
        utils/apply_map.pl -f 1 data/train_sp/utt_map <data/mustc-train/text.tc.${lang} >>data/train_sp/text.tc.${lang}
        awk -v p="sp1.1-" '{printf("%s %s%s\n", $1, p, $1);}' data/mustc-train/utt2spk > data/train_sp/utt_map
        utils/apply_map.pl -f 1 data/train_sp/utt_map <data/mustc-train/text.tc.${lang} >>data/train_sp/text.tc.${lang}
    done


    # Divide into source and target languages
    for x in train_sp mustc-dev mustc-tst-COMMON mustc-tst-HE; do
        local/divide_lang_fr.sh ${x}
    done

    for lang in ${src_lang} ${tgt_lang}; do
        if [ -d data/train_dev.${lang} ];then
            rm -rf data/train_dev.${lang}
        fi
        cp -rf data/mustc-dev.${lang} data/train_dev.${lang}
    done

    for x in train_sp train_dev; do
        # remove utt having more than 3000 frames
        # remove utt having more than 400 characters
        for lang in en fr; do
            remove_longshortdata.sh --maxframes 3000 --maxchars 400 data/${x}.${lang} data/${x}.${lang}.tmp
        done

        # Match the number of utterances between source and target languages
        # extract commocn lines
        cut -f 1 -d " " data/${x}.en.tmp/text > data/${x}.fr.tmp/reclist1
        cut -f 1 -d " " data/${x}.fr.tmp/text > data/${x}.fr.tmp/reclist2
        comm -12 data/${x}.fr.tmp/reclist1 data/${x}.fr.tmp/reclist2 > data/${x}.fr.tmp/reclist

        for lang in en fr; do
            reduce_data_dir.sh data/${x}.${lang}.tmp data/${x}.fr.tmp/reclist data/${x}.${lang}
            utils/fix_data_dir.sh --utt_extra_files "text.tc" data/${x}.${lang}
        done
        rm -rf data/${x}.*.tmp
    done

    if [ ${cmvn} = "true"  ]; then
        # compute global CMVN
        compute-cmvn-stats scp:data/${train_set}/feats.scp data/${train_set}/cmvn.ark
    
        dump.sh --cmd "$train_cmd" --nj 32 --do_delta $do_delta \
            data/${train_set}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/${train_set} ${feat_tr_dir}
        dump.sh --cmd "$train_cmd" --nj 32 --do_delta $do_delta \
            data/${train_dev}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/${train_dev} ${feat_dt_dir}
        for rtask in ${trans_set}; do
            feat_trans_dir=${dumpdir}/${rtask}/delta${do_delta}; mkdir -p ${feat_trans_dir}
            dump.sh --cmd "$train_cmd" --nj 16 --do_delta $do_delta \
                data/${rtask}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/trans/${rtask} \
                ${feat_trans_dir}
        done
    else
        cp data/${train_set}/feats.scp ${feat_tr_dir}/
        cp data/${train_dev}/feats.scp ${feat_dt_dir}/
        for rtask in ${trans_set}; do
            feat_trans_dir=${dumpdir}/${rtask}/delta${do_delta}; mkdir -p ${feat_trans_dir}
            cp data/${rtask}/feats.scp ${feat_trans_dir}/feats.scp
        done
    fi
fi


dict=data/lang_1char/${train_set}_units_${case}.txt
nlsyms=data/lang_1char/non_lang_syms_${case}.txt
echo "dictionary: ${dict}"
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then

    ### Task dependent. You have to check non-linguistic symbols used in the corpus.
    echo "stage 2: Dictionary and Json Data Preparation"
    mkdir -p data/lang_1char/

    echo "make a non-linguistic symbol list for all languages"
    grep sp1.0 data/${train_set_prefix}*/text.${case} | cut -f 2- -d' ' | grep -o -P '&[^;]*;'| sort | uniq > ${nlsyms}
    cat ${nlsyms}

    echo "make a dictionary"
    echo "<unk> 1" > ${dict} # <unk> must be 1, 0 will be used for "blank" in CTC
    grep sp1.0 data/${train_set_prefix}.*/text.${case} | text2token.py -s 1 -n 1 -l ${nlsyms} | cut -f 2- -d" " | tr " " "\n" \
        | sort | uniq | grep -v -e '^\s*$' | awk '{print $0 " " NR+1}' >> ${dict}
    wc -l ${dict}

    echo "make json files"
    local/data2json.sh --nj 16 --feat ${feat_tr_dir}/feats.scp --text data/${train_set}/text.${case} --nlsyms ${nlsyms} \
        data/${train_set} ${dict} > ${feat_tr_dir}/data.${case}.json

    local/data2json.sh --feat ${feat_dt_dir}/feats.scp --text data/${train_dev}/text.${case} --nlsyms ${nlsyms} \
        data/${train_dev} ${dict} > ${feat_dt_dir}/data.${case}.json

    for rtask in ${trans_set}; do
        feat_trans_dir=${dumpdir}/${rtask}/delta${do_delta}
        local/data2json.sh --feat ${feat_trans_dir}/feats.scp --text data/${rtask}/text.${case} --nlsyms ${nlsyms} \
            data/${rtask} ${dict} > ${feat_trans_dir}/data.${case}.json
    done

    # update json (add source references)
    for x in ${train_set} ${train_dev}; do
        feat_dir=${dumpdir}/${x}/delta${do_delta}
        data_dir=data/$(echo ${x} | cut -f -1 -d ".").en
        local/update_json.sh --text ${data_dir}/text.${case} --nlsyms ${nlsyms} \
            ${feat_dir}/data.${case}.json ${data_dir} ${dict}
    done
fi

expname=${feature_type}_cmvn${cmvn}
expdir=exp/${expname}
mkdir -p ${expdir}

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "stage 4: Network Training"

    ${cuda_cmd} --gpu ${ngpu} ${expdir}/train.log \
        st_train.py \
        --config ${train_config} \
        --preprocess-conf ${preprocess_config} \
        --ngpu ${ngpu} \
        --wav2vec ${wav2vec} \
        --idim_reduction ${idim_reduction} \
        --backend ${backend} \
        --outdir ${expdir}/results \
        --tensorboard-dir tensorboard/${expname} \
        --debugmode ${debugmode} \
        --dict ${dict} \
        --debugdir ${expdir} \
        --minibatches ${N} \
        --seed ${seed} \
        --verbose ${verbose} \
        --resume ${resume} \
        --train-json ${feat_tr_dir}/data.${case}.json \
        --valid-json ${feat_dt_dir}/data.${case}.json \
        --enc-init ${asr_model} \
        --dec-init ${mt_model}
fi


if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo "stage 5: Decoding"
    if [[ $(get_yaml.py ${train_config} model-module) = *transformer* ]]; then
        # Average ST models
        if ${use_valbest_average}; then
            trans_model=model.val${n_average}.avg.best
            opt="--log ${expdir}/results/log"
        else
            trans_model=model.last${n_average}.avg.best
            opt="--log"
        fi
        average_checkpoints.py \
            ${opt} \
            --backend ${backend} \
            --snapshots ${expdir}/results/snapshot.ep.* \
            --out ${expdir}/results/${trans_model} \
            --num ${n_average}
    fi
    nj=16

    pids=() # initialize pids
    for ttask in ${trans_set}; do
    (
        decode_dir=decode_${ttask}_$(basename ${decode_config%.*})
        feat_trans_dir=${dumpdir}/${ttask}/delta${do_delta}

        # split data
        splitjson.py --parts ${nj} ${feat_trans_dir}/data.${case}.json

        #### use CPU for decoding
        ngpu=0

        ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
            st_trans.py \
            --config ${decode_config} \
            --ngpu ${ngpu} \
            --backend ${backend} \
            --batchsize 0 \
            --trans-json ${feat_trans_dir}/split${nj}utt/data.JOB.json \
            --result-label ${expdir}/${decode_dir}/data.JOB.json \
            --model ${expdir}/results/${trans_model}

        score_bleu.sh --case ${case} ${expdir}/${decode_dir} fr ${dict}
    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
    echo "Finished"
fi
