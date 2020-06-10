#!/bin/bash

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <set>"
    echo "e.g.: $0 dev"
    exit 1
fi

set=$1

# Copy stuff intoc its final locations [this has been moved from the format_data script]
# for En
mkdir -p data/${set}.en
for f in spk2utt utt2spk segments wav.scp feats.scp utt2num_frames; do
    if [ -f data/${set}/${f} ]; then
        sort data/${set}/${f} > data/${set}.en/${f}
    fi
done
if [ ${set} = "train_sp" ] || [ ${set} = "mustc-dev" ] || [ ${set} = "mustc-tst-COMMON" ] || [ ${set} = "mustc-tst-HE" ]; then
    sort data/${set}/text.tc.en > data/${set}.en/text  # dummy
    sort data/${set}/text.tc.en > data/${set}.en/text.tc
else
    sort data/${set}/text_noseg.tc.en > data/${set}.en/text_noseg.tc
    sort data/${set}/text_noseg.lc.en > data/${set}.en/text_noseg.lc
    sort data/${set}/text_noseg.lc.rm.en > data/${set}.en/text_noseg.lc.rm
fi
utils/fix_data_dir.sh --utt_extra_files "text.tc" data/${set}.en
if [ ${set} = "train_sp" ] || [ ${set} = "mustc-dev" ] || [ ${set} = "mustc-tst-COMMON" ] || [ ${set} = "mustc-tst-HE" ]; then
    if [ -f data/${set}.en/feats.scp ]; then
        utils/validate_data_dir.sh data/${set}.en || exit 1;
    else
        utils/validate_data_dir.sh --no-feats data/${set}.en || exit 1;
    fi
else
    if [ -f data/${set}.en/feats.scp ]; then
        utils/validate_data_dir.sh --no-text data/${set}.en || exit 1;
    else
        utils/validate_data_dir.sh --no-text --no-feats data/${set}.en || exit 1;
    fi
fi

# for Fr
mkdir -p data/${set}.fr
for f in spk2utt utt2spk segments wav.scp feats.scp utt2num_frames; do
    if [ -f data/${set}/${f} ]; then
        sort data/${set}/${f} > data/${set}.fr/${f}
    fi
done
if [ ${set} = "train_sp" ] || [ ${set} = "mustc-dev" ] || [ ${set} = "mustc-tst-COMMON" ] || [ ${set} = "mustc-tst-HE" ]; then
    sort data/${set}/text.tc.fr > data/${set}.fr/text  # dummy
    sort data/${set}/text.tc.fr > data/${set}.fr/text.tc
else
    sort data/${set}/text_noseg.tc.fr > data/${set}.fr/text_noseg.tc
    sort data/${set}/text_noseg.lc.fr > data/${set}.fr/text_noseg.lc
    sort data/${set}/text_noseg.lc.rm.fr > data/${set}.fr/text_noseg.lc.rm
fi
utils/fix_data_dir.sh --utt_extra_files "text.tc text.lc text.lc.rm" data/${set}.fr
if [ ${set} = "train_sp" ] || [ ${set} = "mustc-dev" ] || [ ${set} = "mustc-tst-COMMON" ] || [ ${set} = "mustc-tst-HE" ]; then
    if [ -f data/${set}.fr/feats.scp ]; then
        utils/validate_data_dir.sh data/${set}.fr || exit 1;
    else
        utils/validate_data_dir.sh --no-feats data/${set}.fr || exit 1;
    fi
else
    if [ -f data/${set}.fr/feats.scp ]; then
        utils/validate_data_dir.sh --no-text data/${set}.fr || exit 1;
    else
        utils/validate_data_dir.sh --no-text --no-feats data/${set}.fr || exit 1;
    fi
fi
