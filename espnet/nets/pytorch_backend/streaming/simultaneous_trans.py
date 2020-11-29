import numpy as np
import torch
import torch.nn.functional as F
import math
import textgrids

import logging
import math
READ=0
WRITE=1

np.random.seed(7)

def len2numframes(len_, sample_rate=16000, frame_len=0.025, frame_shift=0.01):
    # len_ in second
    re = int(round((len_ - frame_len) / frame_shift + 1))
    #re = int(math.floor((len_ - frame_len) / frame_shift + 1))
    if re < 0:
        re = 0
    return re

def read_textgrid(segment_file, k=1):
    # If a TextGrid file is available, read it
    grid = textgrids.TextGrid(segment_file)
    segments = []
    offset = 0.0
    count = 1
    for i, w in enumerate(grid['words']):
        # Convert Praat to Unicode in the label
        label = w.text.transcode()
        #if label == '' and i == 0:  # space
        #    pass
        if label == '':  # space
            continue

        if count < k:
            count += 1
            continue

        if i == len(grid['words']) - 2 and grid['words'][i + 1].text.transcode() == '':
            segments.append([len2numframes(offset), len2numframes(grid['words'][i + 1].xmax)])
        else:
            segments.append([len2numframes(offset), len2numframes(w.xmax)])

        #segments.append([offset, w.xmax])
        #segments.append([len2numframes(offset), len2numframes(w.xmax)])
        offset = w.xmax

    if len(segments) == 0:
        segments.append([0, len2numframes(grid.xmax)])

    return segments

def read_textgrid2(segment_file, k=5, sample_rate = 16000):
    # If a TextGrid file is available, read it
    grid = textgrids.TextGrid(segment_file)
    segments = []
    count = 1
    for i, w in enumerate(grid['words']):
        # Convert Praat to Unicode in the label
        label = w.text.transcode()
        if label == '':  # space
            continue

        if count < k:
            count += 1
            continue

        segments.append([len2numframes(w.xmin), len2numframes(w.xmax)])

    if len(segments) == 0:
        segments.append([0.0, len2numframes(grid.xmax)])

    return segments

def rand_sum(N, M, seed=7):
    # Generate an array of N integers whose some is M
    np.random.seed(seed)
    array = np.random.multinomial(M, np.ones(N) / N)
    return array

def rand_segs(input_segments, k):
    # Generate random senquence of segments whose length is the same as the input segments
    # Number of output segments = number of input segments
    if input_segments[-1][1] <= k:
        return input_segments
    lens = rand_sum(len(input_segments)-1, input_segments[-1][1] - k)
    output_segments = []
    count = k
    output_segments.append([0, k])
    for i, len_ in enumerate(lens):
        output_segments.append([count, count+len_])
        count += len_
    return output_segments

def rand_segs2(input_segments, k):
    # Generate random senquence of segments whose length is the same as the input segments
    # Number of output segments is random
    sequence_len = input_segments[-1][1]
    #num_out_segs = np.random.randint(1, sequence_len)
    num_out_segs = np.random.randint(sequence_len / 200, sequence_len / 5)
    lens = rand_sum(num_out_segs, sequence_len)
    output_segments = []
    count = 0
    for i, len_ in enumerate(lens):
        output_segments.append([count, count + len_])
        count += len_
    return output_segments

def rand_segs3(input_segments, k, low=10, high=50):
    # Generate random senquence of segments whose length is the same as the input segments
    # Each frame is between low and high
    # Default: low = 10ms, high = 50ms
    sequence_len = input_segments[-1][1]
    lens = np.random.randint(10, 50, round(sequence_len / 10))
    output_segments = []
    count = 0
    i = 0
    while (count < sequence_len):
        output_segments.append([count, count + lens[i]])
        count += lens[i]
        i += 1
    output_segments.append([output_segments[-1][1], sequence_len])
    return output_segments

class SimultaneousSTE2E(object):
    """SimultaneousSTE2E constructor.
    :param E2E e2e: E2E ST object
    :param trans_args: arguments for "trans" method of E2E
    """

    def __init__(self, e2e, trans_args, rnnlm=None):
        self._e2e = e2e
        self._trans_args = trans_args
        self._char_list = e2e.char_list
        self._rnnlm = rnnlm

        self._e2e.eval()

        self.previous_encoder_recurrent_state = None

        if self._trans_args.ngpu > 1:
            raise NotImplementedError("only single GPU decoding is supported")
        if self._trans_args.ngpu == 1:
            self.device = "cuda"
        else:
            self.device = "cpu"
        self.dtype = getattr(torch, self._trans_args.dtype)

        self.enc_states = None

        self.hyp = {'score': 0.0, 'yseq': torch.tensor([self._e2e.dec.sos], device=self.device), 'states': None, 'delays': []}
        self.finished = False
        self.finish_read = False
        self.last_action = None
        self.k = 200
        self.g = self.k
        self.N = 1 #2 # maximum number of target tokens generated at one step
        #self.g = 1000000 #offline
        #self.g = math.inf
        #self.s = 5
        self.s = 20
        self.max_len = 400
        self.min_len = 0
        self.offset = 0
        #self.max_len = 1000

        assert self._trans_args.batchsize <= 1, \
            "SegmentStreamingE2E works only with batch size <= 1"

    def decision_from_states(self):

        #if len(self.enc_states) == 0:
        if self.enc_states is None:
            return READ

        # follow every read with a transcription:
        if self.last_action == READ or self.finish_read:
            return WRITE
        else:
            return READ

    #def read_textgrid(self, segment_file, k=5, sample_rate=16000):
    #    # If a TextGrid file is available, read it
    #    grid = textgrids.TextGrid(segment_file)
    #    segments = []
    #    offset = 0.0
    #    count = 0

    #    while offset < self.k:
    #        if count >= len(grid['words']):
    #            break
    #        w = grid['words'][count]
    #        count += 1
    #        # Convert Praat to Unicode in the label
    #        label = w.text.transcode()
    #        if label == '':  # space
    #            continue
    #        offset = w.xmax
    #        if len2numframes(offset) > self.k:
    #            count -= 1
    #            offset = grid['words'][count-1].xmax
    #            break
    #        # count += 1
    #    if count < len(grid['words']):
    #        for i in range(count, len(grid['words'])):
    #            w = grid['words'][i]
    #            # Convert Praat to Unicode in the label
    #            label = w.text.transcode()
    #            if label == '':  # space
    #                continue
    #            if i == len(grid['words']) - 2 and grid['words'][i + 1].text.transcode() == '':
    #                segments.append([len2numframes(offset), len2numframes(grid['words'][i + 1].xmax)])
    #            else:
    #                # segments.append([offset, w.xmax])
    #                segments.append([len2numframes(offset), len2numframes(w.xmax)])
    #            offset = w.xmax
    #    if len(segments) == 0:
    #        segments.append([0, len2numframes(grid.xmax)])

    #    return segments

    def read_textgrid(self, segment_file, k=5, sample_rate=16000):
        # If a TextGrid file is available, read it
        grid = textgrids.TextGrid(segment_file)
        segments = []
        offset = 0.0
        count = 0

        while offset < self.k:
            if count >= len(grid['words']):
                break
            w = grid['words'][count]
            # count += 1
            # Convert Praat to Unicode in the label
            label = w.text.transcode()
            if label == '':  # space
                count += 1
                continue
            offset = w.xmax
            if len2numframes(offset) > self.k:
                count -= 1
                # offset = grid['words'][count].xmax
                if count > 0 and grid['words'][count].text.transcode() == '':
                    count -= 1
                    # offset = grid['words'][count].xmax
                offset = grid['words'][count].xmin
                break
            count += 1
        if count < len(grid['words']):
            for i in range(count, len(grid['words'])):
                w = grid['words'][i]
                # Convert Praat to Unicode in the label
                label = w.text.transcode()
                if label == '':  # space
                    continue
                if i == len(grid['words']) - 2 and grid['words'][i + 1].text.transcode() == '':
                    segments.append([len2numframes(offset), len2numframes(grid['words'][i + 1].xmax)])
                else:
                    # segments.append([offset, w.xmax])
                    segments.append([len2numframes(offset), len2numframes(w.xmax)])
                offset = w.xmax
        if len(segments) == 0:
            segments.append([0, len2numframes(grid.xmax)])

        return segments

    def predefined_policy(self, x, segment_file=None, num_of_toks=0):
        """
        If a forced-aligment file is available, one could use it
        """
        segment_step = 0
        #segments = read_textgrid(segment_file, k=10)
        segments = self.read_textgrid(segment_file)
        segments = rand_segs3(segments, self.k, 10, 50)
        #segments = read_textgrid2(segment_file, k=5)
        #self.min_len = num_of_toks
        self.g = segments[0][1]
        # HN 09/09 - predefined number of tokens
        #num_of_toks =


        #for i, segment in enumerate(segments):
        #    if segment[i] >= self.g:
        #        self.g = segment[i]
        #        segment_step = i
        #        break
        # Read and Write policy
        action = None
        while action is None:
            if self.finished:
                logging.info('finished ' + str(self.finished))
                logging.info('max_len ' + str(self.max_len))
                # Finish the hypo by sending eos to server
                return self.finish_action()

            # Model make decision given current states
            decision = self.decision_from_states()
            logging.info('decision: ' + str(decision))
            if decision == READ and not self.finish_read:
                # READ
                self.last_action = decision
                if "b" in self._e2e.etype:
                    action = self.read_action_blstm(x, segments, segment_step)
                    segment_step += 1
                else:
                    action = self.read_action_ulstm(x, segments, segment_step)
                    segment_step += 1

            else:
                # WRITE
                self.last_action = WRITE
                #action = self.write_action()
                dec_step = len(self.hyp['yseq'])
                logging.info('dec_step: ' + str(dec_step))
                action = self.write_action_until(dec_step=dec_step)

        return action


    def policy(self, x):
        # Read and Write policy
        action = None
        while action is None:
            if self.finished:
                logging.info('finished ' + str(self.finished))
                logging.info('max_len ' + str(self.max_len))
                # Finish the hypo by sending eos to server
                return self.finish_action()

            # Model make decision given current states
            decision = self.decision_from_states()
            logging.info('decision: ' + str(decision))
            if decision == READ and not self.finish_read:
                # READ
                self.last_action = decision
                action = self.read_action_blstm(x)
                #if "b" in self._e2e.etype:
                #    action = self.read_action_blstm(x)
                #else:
                #    action = self.read_action_ulstm(x)

            else:
                # WRITE
                self.last_action = WRITE
                dec_step = len(self.hyp['yseq'])
                logging.info('dec_step: ' + str(dec_step))
                action = self.write_action_until(dec_step=dec_step)
                #action = self.write_action()

        return action

    def read_action_blstm(self, x,  segments=None, segment_step=0):
        # segment_size =  160000  # Wait-until-end
        logging.info('frame_count=' + str(self.g))
        logging.info('len_in=' + str(len(x)))
        logging.info('enc_step: ' + str(segment_step))
        if (self.g >= len(x)) or (segments is not None and segment_step >= len(segments)-1):
            x_ = x
            self.g = len(x)
            self.finish_read = True
        else:
            x_ = x[:self.g]
        logging.info('len_feat=' + str(len(x_)))
        # if states["steps"]["src"] == 0:
        h, ilen = self._e2e.subsample_frames(x_)
        self.enc_states = self._e2e.encode(torch.as_tensor(h).to(device=self.device, dtype=self.dtype))
        #if self.g == math.inf and len(self.hyp['yseq']) == 1:
        if self.finish_read:
            # offline mode
            self.max_len = max(1, int(self._trans_args.maxlenratio * self.enc_states.size(0)))
            #self.min_len = int(self._trans_args.minlenratio * self.enc_states.size(0))
            logging.info('min_len: ' + str(self.min_len))
        if segments == None and not self.finish_read:
            self.g += self.s
        elif segments is not None and segment_step < (len(segments)-1) and not self.finish_read:
            self.g = segments[segment_step + 1][1]
        #self.g += self.s

    def read_action_ulstm(self, x, segments=None, segment_step=0):
        # uni-direction lstm
        logging.info('frame_count=' + str(self.g))
        logging.info('ulstm len_in=' + str(len(x)))
        logging.info('enc_step: ' + str(segment_step))
        if (self.g >= len(x)) or (segments is not None and segment_step>=len(segments)-1):
            self.g = len(x)
            self.finish_read = True

        x_ = x[self.offset:self.g]
        h, ilens = self._e2e.subsample_frames(x_)
        h, _, self.previous_encoder_recurrent_state = self._e2e.enc(h.unsqueeze(0), ilens, self.previous_encoder_recurrent_state)
        if self.enc_states is None:
            #self.enc_states = torch.empty((0, h.size(2)), device=self.device)
            self.enc_states = h.squeeze(0)
        #self.enc_states = torch.cat((self.enc_states, h.squeeze(0)), dim=0)
        else:
            self.enc_states = torch.cat((self.enc_states, h.squeeze(0)))

        self.offset = self.g
        if segments == None and not self.finish_read:
            self.g += self.s
        elif segments is not None and segment_step < (len(segments)-1) and not self.finish_read: 
            self.g = segments[segment_step + 1][1]

        if self.finish_read:
            tmp_h, tmp_ilens = self._e2e.subsample_frames(x)
            tmp_h, _, tmp_prev = self._e2e.enc(tmp_h.unsqueeze(0), tmp_ilens, None)
            self.enc_states = tmp_h.squeeze(0)
            # offline mode
            self.max_len = max(1, int(self._trans_args.maxlenratio * self.enc_states.size(0)))
            self.min_len = int(self._trans_args.minlenratio * self.enc_states.size(0))
            logging.info('min_len: ' + str(self.min_len))

    def write_action_until(self, dec_step=1):
        if ((self.hyp['yseq'][len(self.hyp['yseq'])-1] == self._e2e.dec.eos) and (len(self.hyp['yseq']) > 1)) or (len(self.hyp['yseq']) >= self.max_len - 1):
            # Finish this sentence is predict EOS
            if len(self.hyp['yseq']) == self.max_len - 1:
                self.hyp['yseq'] = torch.cat((self.hyp['yseq'], torch.tensor([self._e2e.dec.eos])))
            else:
                logging.info("############## emit EOS ###############")
            self.finished = True
            return

        if self.hyp['states'] is None:
            self.hyp['states'] = self._e2e.dec.init_state(self.enc_states)

        hyp = {'score': 0.0, 'yseq': torch.tensor([self._e2e.dec.sos], device=self.device), 'states': None}
        hyp['states'] = self._e2e.dec.init_state(self.enc_states)
        m_score = 0.0
        m_id = torch.tensor([self._e2e.dec.sos], device=self.device)
        count = 1
        for i in range(self.max_len):
            score, states = self._e2e.dec.score(hyp['yseq'], hyp['states'], self.enc_states)
            score = F.log_softmax(score, dim=1).squeeze()
            # greedy search, take only the (1) best score
            local_best_score, local_best_id = torch.topk(score, 1)
            if (not self.finish_read and int(local_best_id) == self._e2e.dec.eos):
                # Wait for more source toke -
                logging.info(
                    'EOS emits before reading all of source frames, wait for more to come')
                return
                #local_best_score, local_best_id = torch.topk(score, 2)
                #local_best_score = local_best_score[-1].view(1)
                #local_best_id = local_best_id[-1].view(1)
                #logging.info(local_best_score)
                #logging.info(local_best_id)
                #logging.info(
                #    'EOS emits before reading all of source frames, choose the second best target token instead: '
                #    + str(local_best_id[-1]) + ', ' + self._char_list[local_best_id[-1]])
            m_score = local_best_score[0]
            m_id = local_best_id

            hyp['states']['z_prev'] = states['z_prev']
            hyp['states']['c_prev'] = states['c_prev']
            hyp['states']['a_prev'] = states['a_prev']
            hyp['states']['workspace'] = states['workspace']
            hyp['score'] = hyp['score'] + local_best_score[0]
            hyp['yseq'] = torch.cat((hyp['yseq'], local_best_id))

            if self.finish_read and i >= dec_step-1:
                self.hyp['states']['z_prev'] = hyp['states']['z_prev']
                self.hyp['states']['c_prev'] = hyp['states']['c_prev']
                self.hyp['states']['a_prev'] = hyp['states']['a_prev']
                self.hyp['states']['workspace'] = hyp['states']['workspace']
                self.hyp['score'] = self.hyp['score'] + m_score
                self.hyp['yseq'] = torch.cat((self.hyp['yseq'], m_id))
                self.hyp['delays'].append(self.g)
                if ((self.hyp['yseq'][len(self.hyp['yseq']) - 1] == self._e2e.dec.eos) and (
                        len(self.hyp['yseq']) > 1)) or (len(self.hyp['yseq']) == self.max_len - 1):
                    # Finish this sentence is predict EOS
                    if len(self.hyp['yseq']) == self.max_len - 1:
                        self.hyp['yseq'] = torch.cat((self.hyp['yseq'], torch.tensor([self._e2e.dec.eos])))
                    else:
                        logging.info("############## emit EOS ###############")
                    self.finished = True
                    return
            elif not self.finish_read and i >= dec_step - 1:
                self.hyp['states']['z_prev'] = hyp['states']['z_prev']
                self.hyp['states']['c_prev'] = hyp['states']['c_prev']
                self.hyp['states']['a_prev'] = hyp['states']['a_prev']
                self.hyp['states']['workspace'] = hyp['states']['workspace']
                self.hyp['score'] = self.hyp['score'] + m_score
                self.hyp['yseq'] = torch.cat((self.hyp['yseq'], m_id))
                self.hyp['delays'].append(self.g - self.s)
                if count >= self.N:
                    return
                count += 1
        #if not self.finish_read:
        #    self.hyp['states']['z_prev'] = hyp['states']['z_prev']
        #    self.hyp['states']['c_prev'] = hyp['states']['c_prev']
        #    self.hyp['states']['a_prev'] = hyp['states']['a_prev']
        #    self.hyp['states']['workspace'] = hyp['states']['workspace']
        #    self.hyp['score'] = self.hyp['score'] + m_score
        #    self.hyp['yseq'] = torch.cat((self.hyp['yseq'], m_id))

    def write_action(self):
        model_index = 0
        if self.hyp['states'] is None:
            self.hyp['states'] = self._e2e.dec.init_state(self.enc_states)
        #if ((self.hyp['yseq'][len(self.hyp['yseq'])-1] == self._e2e.dec.eos) and (len(self.hyp['yseq']) > 1)) or (len(self.hyp['yseq']) == self.max_len):
        if ((self.hyp['yseq'][len(self.hyp['yseq'])-1] == self._e2e.dec.eos) and (len(self.hyp['yseq']) > 1)) or (len(self.hyp['yseq']) == self.max_len - 1):
            # Finish this sentence is predict EOS
            if len(self.hyp['yseq']) == self.max_len - 1:
                self.hyp['yseq'] = torch.cat((self.hyp['yseq'], torch.tensor([self._e2e.dec.eos])))
            else:
                logging.info("############## emit EOS ###############")
            self.finished = True
            return

        score, states = self._e2e.dec.score(self.hyp['yseq'], self.hyp['states'], self.enc_states)
        score = F.log_softmax(score, dim=1).squeeze()
        # greedy search, take only the (1) best score
        local_best_score, local_best_id = torch.topk(score, 1)
        logging.info('dec_step: ' + str(len(self.hyp['yseq'])))
        #logging.info(local_best_score)
        logging.info(local_best_id)
        if (not self.finish_read and int(local_best_id) == self._e2e.dec.eos) or \
                (self.finish_read and len(self.hyp['yseq']) < self.min_len and int(local_best_id) == self._e2e.dec.eos):
            local_best_score, local_best_id = torch.topk(score, 2)
            local_best_score = local_best_score[-1].view(1)
            local_best_id = local_best_id[-1].view(1)
            logging.info(local_best_score)
            logging.info(local_best_id)
            if not self.finish_read:
                logging.info('EOS emits before reading all of source frames, choose the second best target token instead: '
                             + str(local_best_id[-1]) + ', ' + self._char_list[local_best_id[-1]])
            else:
                logging.info('EOS emits before reaching minlen, choose the second best target token instead: '
                             + str(local_best_id[-1]) + ', ' + self._char_list[local_best_id[-1]])

        # [:] is needed!
        self.hyp['states']['z_prev'] = states['z_prev']
        self.hyp['states']['c_prev'] = states['c_prev']
        self.hyp['states']['a_prev'] = states['a_prev']
        self.hyp['states']['workspace'] = states['workspace']
        self.hyp['score'] = self.hyp['score'] + local_best_score[0]
        self.hyp['yseq'] = torch.cat((self.hyp['yseq'], local_best_id))

        #if rnnlm:
        #    self.hyp['rnnlm_prev'] = rnnlm_state
        # will be (2 x beam) hyps at most
        #logging.info(self.hyp)
        #return {'key': 'SEND', 'value': {'dec_hyp': self.hyp}}

    #def finish_read(self):
    #    return self.finish_read

    def finish_action(self):
        return {'key': 'SEND', 'value': {'dec_hyp': self.hyp}}

