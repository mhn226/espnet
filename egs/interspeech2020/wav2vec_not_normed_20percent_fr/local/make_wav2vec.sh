#!/usr/bin/env bash

# Extract wav2vec features 
# fethi.bouagres@univ-lemans.fr

# Begin configuration section.
nj=4
cmd=run.pl
gpu=0
#fbank_config=conf/fbank.conf
#pitch_config=conf/pitch.conf
#pitch_postprocess_config=
#paste_length_tolerance=2
#compress=true

write_utt2num_frames=true  # If true writes utt2num_frames.
write_utt2dur=true
# End configuration section.

echo "$0 $@"  # Print the command line for logging.

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# -lt 2 ] || [ $# -gt 4 ]; then
  cat >&2 <<EOF
Usage: $0 [options] <data-dir> <wav2vec-model-path>  [<log-dir> [<wav2vec-dir>] ]
 e.g.: $0 data/train
Note: <log-dir> defaults to <data-dir>/log, and
      <wav2vec-dir> defaults to <data-dir>/wav2vec

Options:
  --nj <nj>                            # number of parallel jobs.
  --cmd <run.pl|queue.pl <queue opts>> # how to run jobs.
  --gpu <ngpu> 
EOF
   exit 1;
fi

data=$1
model=$2
if [ $# -ge 3 ]; then
  logdir=$3
else
  logdir=$data/log
fi

if [ $# -ge 4 ]; then
    featsdir=$4
else
    featsdir=wav2vec_features
fi

# make $featsdir an absolute pathname.
featsdir=`perl -e '($dir,$pwd)= @ARGV; if($dir!~m:^/:) { $dir = "$pwd/$dir"; } print $dir; ' $featsdir ${PWD}`

# make $logdir an absolute pathname.
logdir=`perl -e '($dir,$pwd)= @ARGV; if($dir!~m:^/:) { $dir = "$pwd/$dir"; } print $dir; ' $logdir ${PWD}`

echo "feature dir : ${featsdir}"
# use "name" as part of name of the archive.
name=`basename $data`

mkdir -p $logdir || exit 1;

mkdir -p $featsdir ||exit 1;

if [ -f $data/feats.scp ]; then
  mkdir -p $data/.backup
  echo "$0: moving $data/feats.scp to $data/.backup"
  mv $data/feats.scp $data/.backup
fi

scp=$data/wav.scp

required="$scp"

for f in $required; do
  if [ ! -f $f ]; then
    echo "$0: no such file $f"
    exit 1;
  fi
done

#utils/validate_data_dir.sh --no-text --no-feats $data || exit 1;

if [ ! -z "$pitch_postprocess_config" ]; then
  postprocess_config_opt="--config=$pitch_postprocess_config";
else
  postprocess_config_opt=
fi

if [ -f $data/spk2warp ]; then
  echo "$0 [info]: using VTLN warp factors from $data/spk2warp"
  vtln_opts="--vtln-map=ark:$data/spk2warp --utt2spk=ark:$data/utt2spk"
elif [ -f $data/utt2warp ]; then
  echo "$0 [info]: using VTLN warp factors from $data/utt2warp"
  vtln_opts="--vtln-map=ark:$data/utt2warp"
fi


if $write_utt2num_frames; then
    write_num_frames_opt="--write-num-frames=ark,t:$featsdir/utt2num_frames.JOB"
else
    write_num_frames_opt=
fi

if $write_utt2dur; then
    write_utt2dur_opt="--write-utt2dur=ark,t:$featsdir/utt2dur.JOB"
else
    write_utt2dur_opt=
fi


if [ -f $data/segments ]; then
  echo "$0 [info]: segments file exists: using that."
  split_segments=
  for n in $(seq $nj); do
    split_segments="$split_segments $logdir/segments.$n"
  done

  utils/split_scp.pl $data/segments $split_segments || exit 1;
  rm $logdir/.error 2>/dev/null

  wav2vec_feats="scp,p:$scp $logdir/segments.JOB"

  $cmd JOB=1:$nj $logdir/extract_segment_${name}.JOB.log \
    extract-segments --min-segment-length=0 $wav2vec_feats  ark:"$featsdir/speech_seg_$name.JOB.ark" \
   || exit 1;

  echo "$cmd JOB=1:$nj $logdir/make_raw_wav2vec_${name}.JOB.log \
      python3 local/wav2vec_feature_extract.py $model $logdir/segments.JOB $featsdir/speech_seg_$name.JOB.ark $featsdir/raw_wav2vec_$name.JOB.ark $featsdir/raw_wav2vec_$name.JOB.scp $write_utt2dur_opt $write_num_frames_opt $logdir JOB "

   $cmd JOB=1:$nj $logdir/make_raw_wav2vec_${name}.JOB.log \
   python3 local/wav2vec_feature_extract.py $model $logdir/segments.JOB $featsdir/speech_seg_$name.JOB.ark $featsdir/raw_wav2vec_$name.JOB.ark $featsdir/raw_wav2vec_$name.JOB.scp $write_utt2dur_opt $write_num_frames_opt $logdir JOB || exit 1;

else
  echo "$0: [info]: no segments file exists: assuming wav.scp indexed by utterance."
  split_scps=
  for n in $(seq $nj); do
    split_scps="$split_scps $logdir/wav_${name}.$n.scp"
  done

  utils/split_scp.pl $scp $split_scps || exit 1;

fi

if [ -f $logdir/.error.$name ]; then
  echo "$0: Error producing filterbank features for $name:"
  tail $logdir/make_wav2vec_${name}.1.log
  exit 1;
fi

# concatenate the .scp files together.
for n in $(seq $nj); do
  cat $featsdir/raw_wav2vec_$name.$n.scp || exit 1
done > $data/feats.scp || exit 1

if $write_utt2num_frames; then
  for n in $(seq $nj); do
    cat $logdir/utt2num_frames.$n || exit 1
  done > $data/utt2num_frames || exit 1
fi

if $write_utt2dur; then
  for n in $(seq $nj); do
    cat $logdir/utt2dur.$n || exit 1
  done > $data/utt2dur || exit 1
fi

# Store frame_shift and fbank_config along with features.
frame_shift=$(perl -ne 'if (/^--frame-shift=(\d+)/) {
                          printf "%.3f", 0.001 * $1; exit; }' $fbank_config)
echo ${frame_shift:-'0.01'} > $data/frame_shift

#rm $logdir/wav_${name}.*.scp  $logdir/segments.* \
#   $logdir/utt2num_frames.* $logdir/utt2dur.* 2>/dev/null

rm -r $logdir

nf=$(wc -l < $data/feats.scp)
nu=$(wc -l < $data/utt2spk)
if [ $nf -ne $nu ]; then
  echo "$0: It seems not all of the feature files were successfully procesed" \
       "($nf != $nu); consider using utils/fix_data_dir.sh $data"
fi

if (( nf < nu - nu/20 )); then
  echo "$0: Less than 95% the features were successfully generated."\
       "Probably a serious error."
  exit 1
fi

echo "$0: Succeeded creating filterbank features for $name"
