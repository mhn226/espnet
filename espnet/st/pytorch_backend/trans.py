"""V2 backend for `st_trans.py` using py:class:`espnet.nets.beam_search.BeamSearch`."""

import json
import logging

import torch

from espnet.asr.asr_utils import add_results_to_json
from espnet.asr.asr_utils import get_model_conf
from espnet.asr.asr_utils import torch_load
from espnet.asr.pytorch_backend.asr_init import load_trained_model
from espnet.nets.pytorch_backend.streaming.simultaneous_trans import SimultaneousSTE2E
from espnet.nets.st_interface import STInterface
from espnet.nets.beam_search import BeamSearch
from espnet.nets.lm_interface import dynamic_import_lm
from espnet.nets.scorers.length_bonus import LengthBonus
from espnet.utils.deterministic_utils import set_deterministic_pytorch
from espnet.utils.io_utils import LoadInputsAndTargets

from espnet.metrics.latency import (
    AverageLagging,
    AverageProportion,
    DifferentiableAverageLagging
)

def trans(args):
    """Decode with custom models that implements ScorerInterface.
    Notes:
        The previous backend espnet.asr.pytorch_backend.st.trans only supports E2E and RNNLM
    Args:
        args (namespace): The program arguments. See py:func:`espnet.bin.st_trans.get_parser` for details
    """
    logging.warning("experimental API for custom LMs is selected by --api v2")
    if args.batchsize > 1:
        raise NotImplementedError("batch decoding is not implemented")
    #if args.streaming_mode is not None:
    #    raise NotImplementedError("streaming mode is not implemented")
    #if args.word_rnnlm:
    #    raise NotImplementedError("word LM is not implemented")

    set_deterministic_pytorch(args)
    models = []
    train_args = []
    for model_arg in args.model:
        model, train_args_ = load_trained_model(model_arg)
        assert isinstance(model, STInterface)
        model.eval()
        models.append(model)
        train_args.append(train_args_)

    prep_conf = train_args[0].preprocess_conf
    char_list = train_args[0].char_list

    load_inputs_and_targets = LoadInputsAndTargets(
        mode='asr', load_output=False, sort_in_input_length=False,
        preprocess_conf=prep_conf
        if args.preprocess_conf is None else args.preprocess_conf,
        preprocess_args={'train': False})

    if args.rnnlm:
        lm_args = get_model_conf(args.rnnlm, args.rnnlm_conf)
        # NOTE: for a compatibility with less than 0.5.0 version models
        lm_model_module = getattr(lm_args, "model_module", "default")
        lm_class = dynamic_import_lm(lm_model_module, lm_args.backend)
        lm = lm_class(len(char_list), lm_args)
        torch_load(args.rnnlm, lm)
        lm.eval()
    else:
        lm = None

    scorers_pool = []
    for model in models:
        scorers_ = model.scorers()
        scorers_["lm"] = lm
        scorers_["length_bonus"] = LengthBonus(len(char_list))
        scorers_pool.append(scorers_)
    weights = dict(
        decoder=1.0,
        lm=args.lm_weight,
        length_bonus=args.penalty)
    beam_search = BeamSearch(
        beam_size=args.beam_size,
        vocab_size=len(char_list),
        weights=weights,
        scorers_pool=scorers_pool,
        sos=model.sos,
        eos=model.eos,
        token_list=char_list,
    )

    if args.ngpu > 1:
        raise NotImplementedError("only single GPU decoding is supported")
    if args.ngpu == 1:
        device = "cuda"
    else:
        device = "cpu"
    dtype = getattr(torch, args.dtype)
    logging.info(f"Decoding device={device}, dtype={dtype}")
    for model in models:
        model.to(device=device, dtype=dtype).eval()
    beam_search.to(device=device, dtype=dtype).eval()

    # read json data
    js = []
    for trans_json in args.trans_json:
        with open(trans_json, 'rb') as f:
            js_ = json.load(f)['utts']
        js.append(js_)
    new_js = {}
    with torch.no_grad():
        for idx, name in enumerate(js[0].keys(), 1):
            logging.info('(%d/%d) decoding ' + name, idx, len(js[0].keys()))
            encs = []
            for model_idx, js_ in enumerate(js):
                batch = [(name, js_[name])]
                feat = load_inputs_and_targets(batch)[0][0]
                enc = models[model_idx].encode(torch.as_tensor(feat).to(device=device, dtype=dtype))
                encs.append(enc)
            nbest_hyps = beam_search(x=encs, maxlenratio=args.maxlenratio, minlenratio=args.minlenratio)
            nbest_hyps = [h.asdict() for h in nbest_hyps[:min(len(nbest_hyps), args.nbest)]]
            new_js[name] = add_results_to_json(js[0][name], nbest_hyps, char_list)

    with open(args.result_label, 'wb') as f:
        f.write(json.dumps({'utts': new_js}, indent=4, ensure_ascii=False, sort_keys=True).encode('utf_8'))

def eval_all_latency(delays, src_len, target_len):
    results = {}
    for name, func in {
        "AL": AverageLagging,
        "AP": AverageProportion,
        "DAL": DifferentiableAverageLagging
    }.items():
        results[name] = func(delays, src_len, target_len).item()
    return results

def word_splitter(yseq, space, eos):
    # A simple word splitter
    sen_ = yseq[yseq != eos]
    space_indices = (sen_ == space).nonzero().squeeze(1)
    space_indices = space_indices[space_indices != 0]
    word_indices = space_indices - torch.ones_like(space_indices)
    word_indices = torch.cat((word_indices, torch.tensor([len(sen_) - 1])))
    return word_indices


def trans_waitk(args):
    """Decode with custom models that implements ScorerInterface.
    Notes:
        The previous backend espnet.asr.pytorch_backend.st.trans only supports E2E and RNNLM
    Args:
        args (namespace): The program arguments. See py:func:`espnet.bin.st_trans.get_parser` for details
    """
    logging.warning("experimental API for custom LMs is selected by --api v2")
    if args.batchsize > 1:
        raise NotImplementedError("batch decoding is not implemented")
    #if args.streaming_mode is not None:
    #    raise NotImplementedError("streaming mode is not implemented")
    #if args.word_rnnlm:
    #    raise NotImplementedError("word LM is not implemented")

    set_deterministic_pytorch(args)
    model, train_args = load_trained_model(args.model[0])
    assert isinstance(model, STInterface)
    model.eval()

    load_inputs_and_targets = LoadInputsAndTargets(
        mode='asr', load_output=False, sort_in_input_length=False,
        preprocess_conf=train_args.preprocess_conf
        if args.preprocess_conf is None else args.preprocess_conf,
        preprocess_args={'train': False})

    if args.rnnlm:
        lm_args = get_model_conf(args.rnnlm, args.rnnlm_conf)
        # NOTE: for a compatibility with less than 0.5.0 version models
        lm_model_module = getattr(lm_args, "model_module", "default")
        lm_class = dynamic_import_lm(lm_model_module, lm_args.backend)
        lm = lm_class(len(train_args.char_list), lm_args)
        torch_load(args.rnnlm, lm)
        lm.eval()
    else:
        lm = None

    #scorers = model.scorers()
    #scorers["lm"] = lm
    #scorers["length_bonus"] = LengthBonus(len(train_args.char_list))
    #weights = dict(
    #    decoder=1.0,
    #    lm=args.lm_weight,
    #    length_bonus=args.penalty)
    #beam_search = BeamSearch(
    #    beam_size=args.beam_size,
    #    vocab_size=len(train_args.char_list),
    #    weights=weights,
    #    scorers=scorers,
    #    sos=model.sos,
    #    eos=model.eos,
    #    token_list=train_args.char_list,
    #)

    if args.ngpu > 1:
        raise NotImplementedError("only single GPU decoding is supported")
    if args.ngpu == 1:
        device = "cuda"
    else:
        device = "cpu"
    dtype = getattr(torch, args.dtype)
    logging.info(f"Decoding device={device}, dtype={dtype}")
    model.to(device=device, dtype=dtype).eval()
    #beam_search.to(device=device, dtype=dtype).eval()

    # read json data
    with open(args.trans_json[0], 'rb') as f:
        js = json.load(f)['utts']
    new_js = {}
    #corpus_latency = {}
    #corpus_AL = []
    #corpus_DAL = []
    with torch.no_grad():
        for idx, name in enumerate(js.keys(), 1):
            logging.info('(%d/%d) decoding ' + name, idx, len(js.keys()))
            batch = [(name, js[name])]

            # HN 09/09: predefine number of toks
            #num_of_toks = js[name]['output'][0]['shape'][0]

            feat = load_inputs_and_targets(batch)[0][0]
            #textgrid_file = '/home/getalp/nguyen35/montreal-forced-aligner/librispeech/data/' + name + '.TextGrid'
            #se2e = SimultaneousSTE2E(e2e=model, recog_args=args, rnnlm=rnnlm)
            se2e = SimultaneousSTE2E(e2e=model, trans_args=args)
            action = {}
            nbest_hyps = []
            for n in range(args.nbest):
                nbest_hyps.append({"yseq": [], "score": 0.0, "latency": {}})

            while action.get('value', None) != model.dec.eos:
                # take an action
                action = se2e.policy(feat)
                #action = se2e.predefined_policy(feat, textgrid_file, num_of_toks)

                #if action['key'] == 'GET':
                #    print('get')
                    #new_states = session.get_src(sent_id, action["value"])
                    #states = self.update_states(states, new_states)

                #elif action['key'] == 'SEND':
                    #print(train_args.char_list[action['value']['dec_hyp']['yseq'][-1]])
                    #text = ''.join(train_args.char_list[int(action['value']['dec_hyp']['yseq'][-1])])
                    #logging.info(text)
                    #for n in range(args.nbest):
                    #    nbest_hyps[n]['yseq'].extend(action['value']['dec_hyp']['yseq'])
                    #    nbest_hyps[n]['score'] += action['value']['dec_hyp']['score']
                if action['key'] == 'SEND':
                    #text = ''.join(train_args.char_list[int(action['value']['dec_hyp']['yseq'][-1])])
                    break
            #nbest_hyps = [h.asdict() for h in nbest_hyps[:min(len(nbest_hyps), args.nbest)]]
            nbest_hyps[0]['yseq'] = action['value']['dec_hyp']['yseq']
            nbest_hyps[0]['scrore'] = action['value']['dec_hyp']['score']
            sen_ = nbest_hyps[0]['yseq'][nbest_hyps[0]['yseq'] != 183]
            space_indices = (sen_ == 179).nonzero().squeeze(1)
            space_indices = space_indices[space_indices != 0]
            word_indices = space_indices - torch.ones_like(space_indices)
            word_indices = torch.cat((word_indices, torch.tensor([len(sen_)-1])))
            print(word_indices)
            word_indices = word_splitter(nbest_hyps[0]['yseq'], 179, model.dec.eos)
            print(word_indices)
            aaaaaaaaaaaaaaaaaaa
            #logging.info('delays: ' + str(action['value']['dec_hyp']['delays']))
            word_delays = [action['value']['dec_hyp']['delays'][i] for i in word_indices]
            logging.info('delays: ' + str(word_delays))
            #latency = eval_all_latency(action['value']['dec_hyp']['delays'], js[name]['input'][0]['shape'][0], js[name]['output'][0]['shape'][0])
            latency = eval_all_latency(word_delays, js[name]['input'][0]['shape'][0], None)
            nbest_hyps[0]['latency'] = latency
            new_js[name] = add_results_to_json(js[name], nbest_hyps, train_args.char_list)
            logging.info('latency: ' + str(latency))
            #corpus_AL.append(latency['AL'])
            #corpus_DAL.append(latency['DAL'])

    with open(args.result_label, 'wb') as f:
        f.write(json.dumps({'utts': new_js}, indent=4, ensure_ascii=False, sort_keys=True).encode('utf_8'))

    #corpus_AL = sum(corpus_AL) / len(corpus_AL)
    #corpus_DAL = sum(corpus_DAL) / len(corpus_DAL)
    #corpus_latency['AL'] = corpus_AL
    #corpus_latency['DAL'] = corpus_DAL
    #print(corpus_latency)