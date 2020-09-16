#!/usr/bin/env python3
# encoding: utf-8

# Copyright 2019 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Training/decoding definition for the speech translation task."""

import copy
import json
import logging
import multiprocessing
import os
import sys
from argparse import Namespace

import six
from chainer import training
from chainer.training import extensions
import numpy as np
from tensorboardX import SummaryWriter
import torch
import torch.nn.functional as F

from espnet.asr.asr_utils import adadelta_eps_decay
from espnet.asr.asr_utils import adam_lr_decay
from espnet.asr.asr_utils import add_results_to_json
from espnet.asr.asr_utils import CompareValueTrigger
from espnet.asr.asr_utils import get_model_conf
from espnet.asr.asr_utils import restore_snapshot
from espnet.asr.asr_utils import snapshot_object
from espnet.asr.asr_utils import torch_load
from espnet.asr.asr_utils import torch_resume
from espnet.asr.asr_utils import torch_snapshot
from espnet.asr.pytorch_backend.asr_init import load_trained_model
from espnet.asr.pytorch_backend.asr_init import load_trained_modules
from espnet.nets.e2e_asr_common import end_detect

from espnet.nets.pytorch_backend.e2e_asr import pad_list
import espnet.nets.pytorch_backend.lm.default as lm_pytorch
from espnet.nets.st_interface import STInterface
from espnet.utils.dataset import ChainerDataLoader
from espnet.utils.dataset import TransformDataset
from espnet.utils.deterministic_utils import set_deterministic_pytorch
from espnet.utils.dynamic_import import dynamic_import
from espnet.utils.io_utils import LoadInputsAndTargets
from espnet.utils.training.batchfy import make_batchset
from espnet.utils.training.iterators import ShufflingEnabler
from espnet.utils.training.tensorboard_logger import TensorboardLogger
from espnet.utils.training.train_utils import check_early_stop
from espnet.utils.training.train_utils import set_early_stop

from espnet.asr.pytorch_backend.asr import CustomConverter as ASRCustomConverter
from espnet.asr.pytorch_backend.asr import CustomEvaluator
from espnet.asr.pytorch_backend.asr import CustomUpdater

import espnet.lm.pytorch_backend.extlm as extlm_pytorch

import matplotlib
matplotlib.use('Agg')

if sys.version_info[0] == 2:
    from itertools import izip_longest as zip_longest
else:
    from itertools import zip_longest as zip_longest


class CustomConverter(ASRCustomConverter):
    """Custom batch converter for Pytorch.

    Args:
        subsampling_factor (int): The subsampling factor.
        dtype (torch.dtype): Data type to convert.
        asr_task (bool): multi-task with ASR task.

    """

    def __init__(self, subsampling_factor=1, dtype=torch.float32, asr_task=False):
        """Construct a CustomConverter object."""
        super().__init__(subsampling_factor=subsampling_factor, dtype=dtype)
        self.asr_task = asr_task

    def __call__(self, batch, device=torch.device('cpu')):
        """Transform a batch and send it to a device.

        Args:
            batch (list): The batch to transform.
            device (torch.device): The device to send to.

        Returns:
            tuple(torch.Tensor, torch.Tensor, torch.Tensor)

        """
        _, ys = batch[0]
        ys_asr = copy.deepcopy(ys)
        xs_pad, ilens, ys_pad = super().__call__(batch, device)
        if self.asr_task:
            ys_pad_asr = pad_list([torch.from_numpy(np.array(y[1])).long()
                                   for y in ys_asr], self.ignore_id).to(device)
        else:
            ys_pad_asr = None

        return xs_pad, ilens, ys_pad, ys_pad_asr


def train(args):
    """Train with the given args.

    Args:
        args (namespace): The program arguments.

    """
    set_deterministic_pytorch(args)

    # check cuda availability
    if not torch.cuda.is_available():
        logging.warning('cuda is not available')

    # get input and output dimension info
    with open(args.valid_json, 'rb') as f:
        valid_json = json.load(f)['utts']
    utts = list(valid_json.keys())
    idim = int(valid_json[utts[0]]['input'][0]['shape'][-1])
    odim = int(valid_json[utts[0]]['output'][0]['shape'][-1])
    logging.info('#input dims : ' + str(idim))
    logging.info('#output dims: ' + str(odim))

    # Initialize with pre-trained ASR encoder and MT decoder
    if args.enc_init is not None or args.dec_init is not None:
        model = load_trained_modules(idim, odim, args, interface=STInterface)
    else:
        model_class = dynamic_import(args.model_module)
        model = model_class(idim, odim, args)
    assert isinstance(model, STInterface)

    subsampling_factor = model.subsample[0]

    if args.rnnlm is not None:
        rnnlm_args = get_model_conf(args.rnnlm, args.rnnlm_conf)
        rnnlm = lm_pytorch.ClassifierWithState(
            lm_pytorch.RNNLM(
                len(args.char_list), rnnlm_args.layer, rnnlm_args.unit,
                getattr(rnnlm_args, "embed_unit", None),  # for backward compatibility
            )
        )
        torch_load(args.rnnlm, rnnlm)
        model.rnnlm = rnnlm

    # write model config
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    model_conf = args.outdir + '/model.json'
    with open(model_conf, 'wb') as f:
        logging.info('writing a model config file to ' + model_conf)
        f.write(json.dumps((idim, odim, vars(args)),
                           indent=4, ensure_ascii=False, sort_keys=True).encode('utf_8'))
    for key in sorted(vars(args).keys()):
        logging.info('ARGS: ' + key + ': ' + str(vars(args)[key]))

    reporter = model.reporter

    # check the use of multi-gpu
    if args.ngpu > 1:
        if args.batch_size != 0:
            logging.warning('batch size is automatically increased (%d -> %d)' % (
                args.batch_size, args.batch_size * args.ngpu))
            args.batch_size *= args.ngpu

    # set torch device
    device = torch.device("cuda" if args.ngpu > 0 else "cpu")
    if args.train_dtype in ("float16", "float32", "float64"):
        dtype = getattr(torch, args.train_dtype)
    else:
        dtype = torch.float32
    model = model.to(device=device, dtype=dtype)

    # Setup an optimizer
    if args.opt == 'adadelta':
        #for name, param in model.named_parameters():
        #    print(name)
        optimizer = torch.optim.Adadelta(
            model.parameters(), rho=0.95, eps=args.eps,
            weight_decay=args.weight_decay)
    elif args.opt == 'adam':
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=args.lr,
                                     weight_decay=args.weight_decay)
    elif args.opt == 'noam':
        from espnet.nets.pytorch_backend.transformer.optimizer import get_std_opt
        optimizer = get_std_opt(model, args.adim, args.transformer_warmup_steps, args.transformer_lr)
    else:
        raise NotImplementedError("unknown optimizer: " + args.opt)

    # setup apex.amp
    if args.train_dtype in ("O0", "O1", "O2", "O3"):
        try:
            from apex import amp
        except ImportError as e:
            logging.error(f"You need to install apex for --train-dtype {args.train_dtype}. "
                          "See https://github.com/NVIDIA/apex#linux")
            raise e
        if args.opt == 'noam':
            model, optimizer.optimizer = amp.initialize(model, optimizer.optimizer, opt_level=args.train_dtype)
        else:
            model, optimizer = amp.initialize(model, optimizer, opt_level=args.train_dtype)
        use_apex = True
    else:
        use_apex = False

    # FIXME: TOO DIRTY HACK
    setattr(optimizer, "target", reporter)
    setattr(optimizer, "serialize", lambda s: reporter.serialize(s))

    # Setup a converter
    converter = CustomConverter(subsampling_factor=subsampling_factor, dtype=dtype,
                                asr_task=args.asr_weight > 0)

    # read json data
    with open(args.train_json, 'rb') as f:
        train_json = json.load(f)['utts']
    with open(args.valid_json, 'rb') as f:
        valid_json = json.load(f)['utts']

    use_sortagrad = args.sortagrad == -1 or args.sortagrad > 0
    # make minibatch list (variable length)
    train = make_batchset(train_json, args.batch_size,
                          args.maxlen_in, args.maxlen_out, args.minibatches,
                          min_batch_size=args.ngpu if args.ngpu > 1 else 1,
                          shortest_first=use_sortagrad,
                          count=args.batch_count,
                          batch_bins=args.batch_bins,
                          batch_frames_in=args.batch_frames_in,
                          batch_frames_out=args.batch_frames_out,
                          batch_frames_inout=args.batch_frames_inout)
    valid = make_batchset(valid_json, 1,
                          args.maxlen_in, args.maxlen_out, args.minibatches,
                          min_batch_size=args.ngpu if args.ngpu > 1 else 1,
                          count=args.batch_count,
                          batch_bins=args.batch_bins,
                          batch_frames_in=args.batch_frames_in,
                          batch_frames_out=args.batch_frames_out,
                          batch_frames_inout=args.batch_frames_inout)

    load_tr = LoadInputsAndTargets(
        mode='asr', load_output=True, preprocess_conf=args.preprocess_conf,
        preprocess_args={'train': True}  # Switch the mode of preprocessing
    )
    load_cv = LoadInputsAndTargets(
        mode='asr', load_output=True, preprocess_conf=args.preprocess_conf,
        preprocess_args={'train': False}  # Switch the mode of preprocessing
    )
    # hack to make batchsize argument as 1
    # actual bathsize is included in a list
    # default collate function converts numpy array to pytorch tensor
    # we used an empty collate function instead which returns list
    train_iter = {'main': ChainerDataLoader(
        dataset=TransformDataset(train, lambda data: converter([load_tr(data)])),
        batch_size=1, num_workers=args.n_iter_processes,
        shuffle=not use_sortagrad, collate_fn=lambda x: x[0])}
    valid_iter = {'main': ChainerDataLoader(
        dataset=TransformDataset(valid, lambda data: converter([load_cv(data)])),
        batch_size=1, shuffle=False, collate_fn=lambda x: x[0],
        num_workers=args.n_iter_processes)}

    # Set up a trainer
    updater = CustomUpdater(
        model, args.grad_clip, train_iter, optimizer,
        device, args.ngpu, args.grad_noise, args.accum_grad, use_apex=use_apex)
    trainer = training.Trainer(
        updater, (args.epochs, 'epoch'), out=args.outdir)

    if use_sortagrad:
        trainer.extend(ShufflingEnabler([train_iter]),
                       trigger=(args.sortagrad if args.sortagrad != -1 else args.epochs, 'epoch'))

    # Resume from a snapshot
    if args.resume:
        logging.info('resumed from %s' % args.resume)
        torch_resume(args.resume, trainer)

    # Evaluate the model with the test dataset for each epoch
    if args.save_interval_iters > 0:
        trainer.extend(CustomEvaluator(model, valid_iter, reporter, device, args.ngpu),
                       trigger=(args.save_interval_iters, 'iteration'))
    else:
        trainer.extend(CustomEvaluator(model, valid_iter, reporter, device, args.ngpu))

    # Save attention weight each epoch
    if args.num_save_attention > 0:
        data = sorted(list(valid_json.items())[:args.num_save_attention],
                      key=lambda x: int(x[1]['input'][0]['shape'][1]), reverse=True)
        if hasattr(model, "module"):
            att_vis_fn = model.module.calculate_all_attentions
            plot_class = model.module.attention_plot_class
        else:
            att_vis_fn = model.calculate_all_attentions
            plot_class = model.attention_plot_class
        att_reporter = plot_class(
            att_vis_fn, data, args.outdir + "/att_ws",
            converter=converter, transform=load_cv, device=device)
        trainer.extend(att_reporter, trigger=(1, 'epoch'))
    else:
        att_reporter = None

    # Make a plot for training and validation values
    trainer.extend(extensions.PlotReport(['main/loss', 'validation/main/loss',
                                          'main/loss_asr', 'validation/main/loss_asr',
                                          'main/loss_st', 'validation/main/loss_st'],
                                         'epoch', file_name='loss.png'))
    trainer.extend(extensions.PlotReport(['main/acc', 'validation/main/acc',
                                          'main/acc_asr', 'validation/main/acc_asr'],
                                         'epoch', file_name='acc.png'))
    trainer.extend(extensions.PlotReport(['main/bleu', 'validation/main/bleu'],
                                         'epoch', file_name='bleu.png'))

    # Save best models
    trainer.extend(snapshot_object(model, 'model.loss.best'),
                   trigger=training.triggers.MinValueTrigger('validation/main/loss'))
    trainer.extend(snapshot_object(model, 'model.acc.best'),
                   trigger=training.triggers.MaxValueTrigger('validation/main/acc'))

    # save snapshot which contains model and optimizer states
    if args.save_interval_iters > 0:
        trainer.extend(torch_snapshot(filename='snapshot.iter.{.updater.iteration}'),
                       trigger=(args.save_interval_iters, 'iteration'))
    else:
        trainer.extend(torch_snapshot(), trigger=(1, 'epoch'))

    # epsilon decay in the optimizer
    if args.opt == 'adadelta':
        if args.criterion == 'acc':
            trainer.extend(restore_snapshot(model, args.outdir + '/model.acc.best', load_fn=torch_load),
                           trigger=CompareValueTrigger(
                               'validation/main/acc',
                               lambda best_value, current_value: best_value > current_value))
            trainer.extend(adadelta_eps_decay(args.eps_decay),
                           trigger=CompareValueTrigger(
                               'validation/main/acc',
                               lambda best_value, current_value: best_value > current_value))
        elif args.criterion == 'loss':
            trainer.extend(restore_snapshot(model, args.outdir + '/model.loss.best', load_fn=torch_load),
                           trigger=CompareValueTrigger(
                               'validation/main/loss',
                               lambda best_value, current_value: best_value < current_value))
            trainer.extend(adadelta_eps_decay(args.eps_decay),
                           trigger=CompareValueTrigger(
                               'validation/main/loss',
                               lambda best_value, current_value: best_value < current_value))
    elif args.opt == 'adam':
        if args.criterion == 'acc':
            trainer.extend(restore_snapshot(model, args.outdir + '/model.acc.best', load_fn=torch_load),
                           trigger=CompareValueTrigger(
                               'validation/main/acc',
                               lambda best_value, current_value: best_value > current_value))
            trainer.extend(adam_lr_decay(args.lr_decay),
                           trigger=CompareValueTrigger(
                               'validation/main/acc',
                               lambda best_value, current_value: best_value > current_value))
        elif args.criterion == 'loss':
            trainer.extend(restore_snapshot(model, args.outdir + '/model.loss.best', load_fn=torch_load),
                           trigger=CompareValueTrigger(
                               'validation/main/loss',
                               lambda best_value, current_value: best_value < current_value))
            trainer.extend(adam_lr_decay(args.lr_decay),
                           trigger=CompareValueTrigger(
                               'validation/main/loss',
                               lambda best_value, current_value: best_value < current_value))

    # Write a log of evaluation statistics for each epoch
    trainer.extend(extensions.LogReport(trigger=(args.report_interval_iters, 'iteration')))
    report_keys = ['epoch', 'iteration', 'main/loss', 'main/loss_st', 'main/loss_asr',
                   'validation/main/loss', 'validation/main/loss_st', 'validation/main/loss_asr',
                   'main/acc', 'validation/main/acc']
    if args.asr_weight > 0:
        report_keys.append('main/acc_asr')
        report_keys.append('validation/main/acc_asr')
    report_keys += ['elapsed_time']
    if args.opt == 'adadelta':
        trainer.extend(extensions.observe_value(
            'eps', lambda trainer: trainer.updater.get_optimizer('main').param_groups[0]["eps"]),
            trigger=(args.report_interval_iters, 'iteration'))
        report_keys.append('eps')
    elif args.opt in ['adam', 'noam']:
        trainer.extend(extensions.observe_value(
            'lr', lambda trainer: trainer.updater.get_optimizer('main').param_groups[0]["lr"]),
            trigger=(args.report_interval_iters, 'iteration'))
        report_keys.append('lr')
    if args.asr_weight > 0:
        if args.mtlalpha > 0:
            report_keys.append('main/cer_ctc')
            report_keys.append('validation/main/cer_ctc')
        if args.mtlalpha < 1:
            if args.report_cer:
                report_keys.append('validation/main/cer')
            if args.report_wer:
                report_keys.append('validation/main/wer')
    if args.report_bleu:
        report_keys.append('validation/main/bleu')
    trainer.extend(extensions.PrintReport(
        report_keys), trigger=(args.report_interval_iters, 'iteration'))

    trainer.extend(extensions.ProgressBar(update_interval=args.report_interval_iters))
    set_early_stop(trainer, args)

    if args.tensorboard_dir is not None and args.tensorboard_dir != "":
        trainer.extend(TensorboardLogger(SummaryWriter(args.tensorboard_dir), att_reporter),
                       trigger=(args.report_interval_iters, "iteration"))
    # Run the training
    trainer.run()
    check_early_stop(trainer, args.epochs)


def trans(args):
    """Decode with the given args.

    Args:
        args (namespace): The program arguments.

    """
    set_deterministic_pytorch(args)
    model, train_args = load_trained_model(args.model[0])
    assert isinstance(model, STInterface)
    # args.ctc_weight = 0.0
    model.trans_args = args

    # read rnnlm
    if args.rnnlm:
        rnnlm_args = get_model_conf(args.rnnlm, args.rnnlm_conf)
        if getattr(rnnlm_args, "model_module", "default") != "default":
            raise ValueError("use '--api v2' option to decode with non-default language model")
        rnnlm = lm_pytorch.ClassifierWithState(
            lm_pytorch.RNNLM(
                len(train_args.char_list), rnnlm_args.layer, rnnlm_args.unit))
        torch_load(args.rnnlm, rnnlm)
        rnnlm.eval()
    else:
        rnnlm = None

    # gpu
    if args.ngpu == 1:
        gpu_id = list(range(args.ngpu))
        logging.info('gpu id: ' + str(gpu_id))
        model.cuda()
        if rnnlm:
            rnnlm.cuda()

    # read json data
    with open(args.trans_json[0], 'rb') as f:
        js = json.load(f)['utts']
    new_js = {}

    load_inputs_and_targets = LoadInputsAndTargets(
        mode='asr', load_output=False, sort_in_input_length=False,
        preprocess_conf=train_args.preprocess_conf
        if args.preprocess_conf is None else args.preprocess_conf,
        preprocess_args={'train': False})

    if args.batchsize == 0:
        with torch.no_grad():
            for idx, name in enumerate(js.keys(), 1):
                logging.info('(%d/%d) decoding ' + name, idx, len(js.keys()))
                batch = [(name, js[name])]
                feat = load_inputs_and_targets(batch)[0][0]
                nbest_hyps = model.translate(feat, args, train_args.char_list, rnnlm)
                new_js[name] = add_results_to_json(js[name], nbest_hyps, train_args.char_list)

    else:
        def grouper(n, iterable, fillvalue=None):
            kargs = [iter(iterable)] * n
            return zip_longest(*kargs, fillvalue=fillvalue)

        # sort data if batchsize > 1
        keys = list(js.keys())
        if args.batchsize > 1:
            feat_lens = [js[key]['input'][0]['shape'][0] for key in keys]
            sorted_index = sorted(range(len(feat_lens)), key=lambda i: -feat_lens[i])
            keys = [keys[i] for i in sorted_index]

        with torch.no_grad():
            for names in grouper(args.batchsize, keys, None):
                names = [name for name in names if name]
                batch = [(name, js[name]) for name in names]
                feats = load_inputs_and_targets(batch)[0]
                nbest_hyps = model.translate_batch(feats, args, train_args.char_list, rnnlm=rnnlm)

                for i, nbest_hyp in enumerate(nbest_hyps):
                    name = names[i]
                    new_js[name] = add_results_to_json(js[name], nbest_hyp, train_args.char_list)

    with open(args.result_label, 'wb') as f:
        f.write(json.dumps({'utts': new_js}, indent=4, ensure_ascii=False, sort_keys=True).encode('utf_8'))

def enc_worker(model, feat):
    return model.encode(feat)

def dec_worker(model, idx, hs, z_list, c_list, train_args, vy, hyp, rnnlm):
    return model.translate_step(hs, vy, hyp, z_list, c_list,
                                 idx, model.trans_args, train_args.char_list, rnnlm)

def trans_step_ensemble_parallelizing(models, feat, rnnlm, train_args, enc_pool):
    manager = multiprocessing.Manager()
    # encoder
    hs = [None] * len(models)
    enc_results = enc_pool.starmap(enc_worker, zip(models, feat))
    for idx, result in enumerate(enc_results):
        hs[idx] = result

    # initialization
    c_list = [None] * len(models)
    z_list = [None] * len(models)
    for i, model in enumerate(models):
        c_list[i] = [model.dec.zero_state(hs[i].unsqueeze(0))]
        z_list[i] = [model.dec.zero_state(hs[i].unsqueeze(0))]
        for _ in six.moves.range(1, model.dec.dlayers):
            c_list[i].append(model.dec.zero_state(hs[i].unsqueeze(0)))
            z_list[i].append(model.dec.zero_state(hs[i].unsqueeze(0)))

    a = [None] * (len(models))
    att_w_list = [None] * (len(models))
    for i, model in enumerate(models):
        model.dec.att[0].reset()

    beam = models[0].trans_args.beam_size
    penalty = models[0].trans_args.penalty


    # preprate sos
    if models[0].dec.replace_sos and models[0].trans_args.tgt_lang:
        y = train_args[0].char_list.index(models[0].trans_args.tgt_lang)
    else:
        y = models[0].dec.sos
    logging.info('<sos> index: ' + str(y))
    logging.info('<sos> mark: ' + train_args[0].char_list[y])
    vy = hs[0].new_zeros(1).long()

    maxlen = np.amin([hs[idx].size(0) for idx in range(len(models))])
    if models[0].trans_args.maxlenratio == 0:
        maxlen = hs[0].shape[0]
    else:
        # maxlen >= 1
        maxlen = max(1, int(models[0].trans_args.maxlenratio * hs[0].size(0)))
    minlen = int(models[0].trans_args.minlenratio * hs[0].size(0))
    logging.info('max output length: ' + str(maxlen))
    logging.info('min output length: ' + str(minlen))

    hyp = {'score': 0.0, 'yseq': [y], 'c_prev': c_list, 'z_prev': z_list, 'a_prev': a}

    hyps = [hyp]
    ended_hyps = []

    model_indices = [x for x in range(len(models))]
    # beam search
    for i in six.moves.range(maxlen):
        logging.debug('position ' + str(i))
        hyps_best_kept = []
        for hyp in hyps:
            vy.unsqueeze(1)
            vy[0] = hyp['yseq'][i]
            logits = [None] * len(models)
            for model_index, model in enumerate(models):
                logits[model_index], z_list[model_index], c_list[model_index], att_w_list[
                    model_index] = model.translate_step(hs[model_index],
                                                        vy, hyp, z_list[model_index], c_list[model_index], model_index,
                                                        model.trans_args, train_args[model_index].char_list, rnnlm)
            logits = torch.mean(torch.stack(logits), dim=0)
            local_att_scores = F.log_softmax(logits, dim=1)

            if rnnlm:
                # rnnlm_state, local_lm_scores = rnnlm.predict(hyp['rnnlm_prev'], vy)
                # local_scores = local_att_scores + trans_args.lm_weight * local_lm_scores
                print('Not yet supported')
            else:
                local_scores = local_att_scores

            local_best_scores, local_best_ids = torch.topk(local_scores, beam, dim=1)

            for j in six.moves.range(beam):
                new_hyp = {}
                # [:] is needed!
                new_hyp['z_prev'] = [z_list[idx][:] for idx in range(len(models))]
                new_hyp['c_prev'] = [c_list[idx][:] for idx in range(len(models))]
                new_hyp['a_prev'] = [att_w_list[idx][:] for idx in range(len(models))]
                new_hyp['score'] = hyp['score'] + local_best_scores[0, j]
                new_hyp['yseq'] = [0] * (1 + len(hyp['yseq']))
                new_hyp['yseq'][:len(hyp['yseq'])] = hyp['yseq']
                new_hyp['yseq'][len(hyp['yseq'])] = int(local_best_ids[0, j])
                #if rnnlm:
                #    new_hyp['rnnlm_prev'] = rnnlm_state
                hyps_best_kept.append(new_hyp)

            hyps_best_kept = sorted(
                hyps_best_kept, key=lambda x: x['score'], reverse=True)[:beam]

        # sort and get nbest
        hyps = hyps_best_kept
        logging.debug('number of pruned hypotheses: ' + str(len(hyps)))
        logging.debug('best hypo: ' + ''.join([train_args[0].char_list[int(x)] for x in hyps[0]['yseq'][1:]]))
        # add eos in the final loop to avoid that there are no ended hyps
        if i == maxlen - 1:
            logging.info('adding <eos> in the last position in the loop')
            for hyp in hyps:
                hyp['yseq'].append(models[0].dec.eos)

        # add ended hypotheses to a final list, and removed them from current hypotheses
        # (this will be a problem, number of hyps < beam)
        remained_hyps = []
        for hyp in hyps:
            if hyp['yseq'][-1] == models[0].dec.eos:
                # only store the sequence that has more than minlen outputs
                # also add penalty
                if len(hyp['yseq']) > minlen:
                    hyp['score'] += (i + 1) * penalty
                    if rnnlm:  # Word LM needs to add final <eos> score
                        hyp['score'] += models[0].trans_args.lm_weight * rnnlm.final(hyp['rnnlm_prev'])
                    ended_hyps.append(hyp)
            else:
                remained_hyps.append(hyp)

        # end detection
        if end_detect(ended_hyps, i) and models[0].trans_args.maxlenratio == 0.0:
            logging.info('end detected at %d', i)
            break

        hyps = remained_hyps
        if len(hyps) > 0:
            logging.debug('remaining hypotheses: ' + str(len(hyps)))
        else:
            logging.info('no hypothesis. Finish decoding.')
            break

        for hyp in hyps:
            logging.debug('hypo: ' + ''.join([train_args[0].char_list[int(x)] for x in hyp['yseq'][1:]]))

        logging.debug('number of ended hypotheses: ' + str(len(ended_hyps)))

    nbest_hyps = sorted(ended_hyps, key=lambda x: x['score'], reverse=True)[
                 :min(len(ended_hyps), models[0].trans_args.nbest)]

    # check number of hypotheses
    if len(nbest_hyps) == 0:
        logging.warning('there is no N-best results, perform translation again with smaller minlenratio.')
        # should copy because Namespace will be overwritten globally
        models[0].trans_args = Namespace(**vars(models[0].trans_args))
        models[0].trans_args.minlenratio = max(0.0, models[0].trans_args.minlenratio - 0.1)
        return trans_step_ensemble(models, feat, rnnlm, train_args)

    logging.info('total log probability: ' + str(nbest_hyps[0]['score']))
    logging.info('normalized log probability: ' + str(nbest_hyps[0]['score'] / len(nbest_hyps[0]['yseq'])))

    return nbest_hyps

def trans_step_ensemble(models, feat, rnnlm, train_args):
    # encoder
    hs = []
    for i, model in enumerate(models):
        #hs.append(model.encode(feat[i]).unsqueeze(0))
        hs.append(model.encode(feat[i]))

    # initialization
    c_list = [None] * len(models)
    z_list = [None] * len(models)
    for i, model in enumerate(models):
        c_list[i] = [model.dec.zero_state(hs[i].unsqueeze(0))]
        z_list[i] = [model.dec.zero_state(hs[i].unsqueeze(0))]
        for _ in six.moves.range(1, model.dec.dlayers):
            c_list[i].append(model.dec.zero_state(hs[i].unsqueeze(0)))
            z_list[i].append(model.dec.zero_state(hs[i].unsqueeze(0)))

    a = [None] * (len(models))
    att_w_list = [None] * (len(models))
    for i, model in enumerate(models):
        model.dec.att[0].reset()

    beam = models[0].trans_args.beam_size
    penalty = models[0].trans_args.penalty


    # preprate sos
    if models[0].dec.replace_sos and models[0].trans_args.tgt_lang:
        y = train_args[0].char_list.index(models[0].trans_args.tgt_lang)
    else:
        y = models[0].dec.sos
    logging.info('<sos> index: ' + str(y))
    logging.info('<sos> mark: ' + train_args[0].char_list[y])
    vy = hs[0].new_zeros(1).long()

    maxlen = np.amin([hs[idx].size(0) for idx in range(len(models))])
    if models[0].trans_args.maxlenratio == 0:
        maxlen = hs[0].shape[0]
    else:
        # maxlen >= 1
        maxlen = max(1, int(models[0].trans_args.maxlenratio * hs[0].size(0)))
    minlen = int(models[0].trans_args.minlenratio * hs[0].size(0))
    logging.info('max output length: ' + str(maxlen))
    logging.info('min output length: ' + str(minlen))

    hyp = {'score': 0.0, 'yseq': [y], 'c_prev': c_list, 'z_prev': z_list, 'a_prev': a}

    hyps = [hyp]
    ended_hyps = []

    # beam search
    for i in six.moves.range(maxlen):
        logging.debug('position ' + str(i))
        hyps_best_kept = []
        for hyp in hyps:
            vy.unsqueeze(1)
            vy[0] = hyp['yseq'][i]
            # local_att_scores = [None] * (len(models))
            logits = [None] * (len(models))
            for model_index, model in enumerate(models):
                logits[model_index], z_list[model_index], c_list[model_index], att_w_list[
                    model_index] = model.translate_step(hs[model_index],
                                                        vy, hyp, z_list[model_index], c_list[model_index], model_index,
                                                        model.trans_args, train_args[model_index].char_list, rnnlm)
            logits = torch.mean(torch.stack(logits), dim=0)
            local_att_scores = F.log_softmax(logits, dim=1)

            if rnnlm:
                # rnnlm_state, local_lm_scores = rnnlm.predict(hyp['rnnlm_prev'], vy)
                # local_scores = local_att_scores + trans_args.lm_weight * local_lm_scores
                print('Not yet supported')
            else:
                local_scores = local_att_scores

            local_best_scores, local_best_ids = torch.topk(local_scores, beam, dim=1)
            for j in six.moves.range(beam):
                new_hyp = {}
                # [:] is needed!
                new_hyp['z_prev'] = [z_list[idx][:] for idx in range(len(models))]
                new_hyp['c_prev'] = [c_list[idx][:] for idx in range(len(models))]
                new_hyp['a_prev'] = [att_w_list[idx][:] for idx in range(len(models))]
                new_hyp['score'] = hyp['score'] + local_best_scores[0, j]
                new_hyp['yseq'] = [0] * (1 + len(hyp['yseq']))
                new_hyp['yseq'][:len(hyp['yseq'])] = hyp['yseq']
                new_hyp['yseq'][len(hyp['yseq'])] = int(local_best_ids[0, j])
                #if rnnlm:
                #    new_hyp['rnnlm_prev'] = rnnlm_state
                hyps_best_kept.append(new_hyp)

            hyps_best_kept = sorted(
                hyps_best_kept, key=lambda x: x['score'], reverse=True)[:beam]

        # sort and get nbest
        hyps = hyps_best_kept
        logging.debug('number of pruned hypotheses: ' + str(len(hyps)))
        logging.debug('best hypo: ' + ''.join([train_args[0].char_list[int(x)] for x in hyps[0]['yseq'][1:]]))
        # add eos in the final loop to avoid that there are no ended hyps
        if i == maxlen - 1:
            logging.info('adding <eos> in the last position in the loop')
            for hyp in hyps:
                hyp['yseq'].append(models[0].dec.eos)

        # add ended hypotheses to a final list, and removed them from current hypotheses
        # (this will be a problem, number of hyps < beam)
        remained_hyps = []
        for hyp in hyps:
            if hyp['yseq'][-1] == models[0].dec.eos:
                # only store the sequence that has more than minlen outputs
                # also add penalty
                if len(hyp['yseq']) > minlen:
                    hyp['score'] += (i + 1) * penalty
                    if rnnlm:  # Word LM needs to add final <eos> score
                        hyp['score'] += models[0].trans_args.lm_weight * rnnlm.final(hyp['rnnlm_prev'])
                    ended_hyps.append(hyp)
            else:
                remained_hyps.append(hyp)

        # end detection
        if end_detect(ended_hyps, i) and models[0].trans_args.maxlenratio == 0.0:
            logging.info('end detected at %d', i)
            break

        hyps = remained_hyps
        if len(hyps) > 0:
            logging.debug('remaining hypotheses: ' + str(len(hyps)))
        else:
            logging.info('no hypothesis. Finish decoding.')
            break

        for hyp in hyps:
            logging.debug('hypo: ' + ''.join([train_args[0].char_list[int(x)] for x in hyp['yseq'][1:]]))

        logging.debug('number of ended hypotheses: ' + str(len(ended_hyps)))

    nbest_hyps = sorted(ended_hyps, key=lambda x: x['score'], reverse=True)[
                 :min(len(ended_hyps), models[0].trans_args.nbest)]

    # check number of hypotheses
    if len(nbest_hyps) == 0:
        logging.warning('there is no N-best results, perform translation again with smaller minlenratio.')
        # should copy because Namespace will be overwritten globally
        models[0].trans_args = Namespace(**vars(models[0].trans_args))
        models[0].trans_args.minlenratio = max(0.0, models[0].trans_args.minlenratio - 0.1)
        return trans_step_ensemble(models, feat, rnnlm, train_args)

    logging.info('total log probability: ' + str(nbest_hyps[0]['score']))
    logging.info('normalized log probability: ' + str(nbest_hyps[0]['score'] / len(nbest_hyps[0]['yseq'])))

    return nbest_hyps

def trans_ensemble(args):
    """Decode with the given args.

    Args:
        args (namespace): The program arguments.
    """
    set_deterministic_pytorch(args)
    models = [None] * len(args.model)
    train_args = [None] * len(args.model)
    for i, model_ in enumerate(args.model):
        models[i], train_args[i] = load_trained_model(model_)
        assert isinstance(models[i], STInterface)
        models[i].trans_args = args

    # read rnnlm
    if args.rnnlm:
        rnnlm_args = get_model_conf(args.rnnlm, args.rnnlm_conf)
        if getattr(rnnlm_args, "model_module", "default") != "default":
            raise ValueError("use '--api v2' option to decode with non-default language model")
        rnnlm = lm_pytorch.ClassifierWithState(
            lm_pytorch.RNNLM(
                len(train_args.char_list), rnnlm_args.layer, rnnlm_args.unit))
        torch_load(args.rnnlm, rnnlm)
        rnnlm.eval()
    else:
        rnnlm = None

    if args.word_rnnlm:
        rnnlm_args = get_model_conf(args.word_rnnlm, args.word_rnnlm_conf)
        word_dict = rnnlm_args.char_list_dict
        char_dict = {x: i for i, x in enumerate(train_args.char_list)}
        word_rnnlm = lm_pytorch.ClassifierWithState(lm_pytorch.RNNLM(
            len(word_dict), rnnlm_args.layer, rnnlm_args.unit))
        torch_load(args.word_rnnlm, word_rnnlm)
        word_rnnlm.eval()

        if rnnlm is not None:
            rnnlm = lm_pytorch.ClassifierWithState(
                extlm_pytorch.MultiLevelLM(word_rnnlm.predictor,
                                           rnnlm.predictor, word_dict, char_dict))
        else:
            rnnlm = lm_pytorch.ClassifierWithState(
                extlm_pytorch.LookAheadWordLM(word_rnnlm.predictor,
                                              word_dict, char_dict))
    # gpu
    if args.ngpu == 1:
        gpu_id = list(range(args.ngpu))
        logging.info('gpu id: ' + str(gpu_id))
        for model in models:
            model.cuda()
        #if rnnlm:
        #    rnnlm.cuda()

    # read json data
    # read two (or more) separate json files
    # each belongs to either fbank or wav2vec
    # cannot do it with cpcaudio yet since the input senquence lengths are not identical
    js = []
    for trans_json_ in args.trans_json:
        with open(trans_json_, 'rb') as f:
            tmp = json.load(f)['utts']
            js.append(tmp)
    new_js = {}
    load_inputs_and_targets = LoadInputsAndTargets(
        mode='asr', load_output=False, sort_in_input_length=False,
        preprocess_conf=train_args[0].preprocess_conf
        if args.preprocess_conf is None else args.preprocess_conf,
        preprocess_args={'train': False})

    if args.batchsize == 0:
        with torch.no_grad():
            enc_pool = multiprocessing.Pool(processes=len(models))
            # dec_pool = multiprocessing.Pool(processes=len(models))
            for idx, name in enumerate(js[0].keys(), 1):
                logging.info('(%d/%d) decoding ' + name, idx, len(js[0].keys()))
                feat = []
                for js_data in js:
                    batch_ = [(name, js_data[name])]
                    feat_ = load_inputs_and_targets(batch_)[0][0]
                    feat.append(feat_)
                if hasattr(args, 'streaming_mode') and args.streaming_mode == 'window':
                    logging.info('Using streaming recognizer with window size %d frames', args.streaming_window)
                elif hasattr(args, 'streaming_mode') and args.streaming_mode == 'segment':
                    logging.info('Using streaming recognizer with threshold value %d', args.streaming_min_blank_dur)
                else:
                    # For now rnnlm = None
                    rnnlm = None
                    # nbest_hyps = trans_step_ensemble(models, feat, rnnlm, train_args)
                    # nbest_hyps = trans_step_ensemble_parallelizing(models, feat, rnnlm, train_args, enc_pool, dec_pool)
                    nbest_hyps = trans_step_ensemble_parallelizing(models, feat, rnnlm, train_args, enc_pool)
                new_js[name] = add_results_to_json(js[0][name], nbest_hyps, train_args[0].char_list)
            enc_pool.close()
            enc_pool.join()
    else:
        """ Don't care about this for now
        """

        def grouper(n, iterable, fillvalue=None):
            kargs = [iter(iterable)] * n
            return zip_longest(*kargs, fillvalue=fillvalue)

        # sort data if batchsize > 1
        keys = list(js.keys())
        if args.batchsize > 1:
            feat_lens = [js[key]['input'][0]['shape'][0] for key in keys]
            sorted_index = sorted(range(len(feat_lens)), key=lambda i: -feat_lens[i])
            keys = [keys[i] for i in sorted_index]

        with torch.no_grad():
            for names in grouper(args.batchsize, keys, None):
                names = [name for name in names if name]
                batch = [(name, js[name]) for name in names]
                feats = load_inputs_and_targets(batch)[0]
                nbest_hyps = model.translate_batch(feats, args, train_args.char_list, rnnlm=rnnlm)

                for i, nbest_hyp in enumerate(nbest_hyps):
                    name = names[i]
                    new_js[name] = add_results_to_json(js[name], nbest_hyp, train_args.char_list)

    with open(args.result_label, 'wb') as f:
        f.write(json.dumps({'utts': new_js}, indent=4, ensure_ascii=False, sort_keys=True).encode('utf_8'))
