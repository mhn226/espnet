import numpy as np
import torch
import torch.nn.functional as F

READ=0
WRITE=1

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

        self._blank_idx_in_char_list = -1
        for idx in range(len(self._char_list)):
            if self._char_list[idx] == self._e2e.blank:
                self._blank_idx_in_char_list = idx
                break

        self._subsampling_factor = np.prod(e2e.subsample)
        self._activates = 0
        self._blank_dur = 0

        self._previous_input = []
        self._previous_encoder_recurrent_state = None
        self._encoder_states = []
        self._ctc_posteriors = []

        self.enc_states = []
        #self.c_list = []
        #self.z_list = []
        #self.a = []
        #self.y = self._e2e.dec.sos

        if self._trans_args.ngpu > 1:
            raise NotImplementedError("only single GPU decoding is supported")
        if self._trans_args.ngpu == 1:
            self.device = "cuda"
        else:
            self.device = "cpu"
        self.dtype = getattr(torch, self._trans_args.dtype)

        #yseq = torch.tensor([self.sos], device=x[0].device))
        #self.hyp = {'score': 0.0, 'yseq': torch.tensor([self._e2e.dec.sos], device=self.device), 'states': {'c_prev': [], 'z_prev': [], 'a_prev': []}
        self.hyp = {'score': 0.0, 'yseq': torch.tensor([self._e2e.dec.sos], device=self.device), 'states': None}
        #self.hyp = {'score': 0.0, 'yseq': [self._e2e.dec.sos], 'states': None}
        self.finished = False
        self.finish_read = False
        self.last_action = None
        self.frame_count = 0

        assert self._trans_args.batchsize <= 1, \
            "SegmentStreamingE2E works only with batch size <= 1"
        #assert "b" not in self._e2e.etype, \
        #    "SegmentStreamingE2E works only with uni-directional encoders"

    def decision_from_states(self):
        #print('State\n|| Target:', ''.join(states['tokens']['tgt']), '\n|| ASR:', ''.join(states['tokens']['asr']))

        if len(self.enc_states) == 0:
            return READ

        # follow every read with a transcription:
        if self.last_action == READ:
            return WRITE
        else:
            return READ

        #return READ

    def policy(self, x):
        # Read and Write policy
        action = None

        while action is None:
            if self.finished:
                # Finish the hypo by sending eos to server
                return self.finish_action()

            # Model make decision given current states
            decision = self.decision_from_states()

            if decision == READ and not self.finish_read:
                # READ
                self.last_action = decision
                action = self.read_action(x)

            #elif decision == TRANSCRIBE and not self.finish_transcription(states):
            #    # TRANSCRIBE
            #    states['last_action'] = decision
            #    action = self.transcribe_action(states)

            else:
                # WRITE
                self.last_action = WRITE
                action = self.write_action()

            # None means we make decision again but not sending server anything
            # This happened when read a bufffered token
            # Or predict a subword
        return action

    def read_action(self, x):
        # segment_size =  160000  # Wait-until-end
        self.frame_count += 7000
        # if states["steps"]["src"] == 0:
        x = x[:self.frame_count]
        h, ilen = self._e2e.subsample_frames(x)
        # Run encoder and apply greedy search on CTC softmax output
        self.enc_states = self._e2e.encode(torch.as_tensor(h).to(device=self.device, dtype=self.dtype))
        #h, _, self._previous_encoder_recurrent_state = self._e2e.enc(
        #    h.unsqueeze(0),
        #    ilen,
        #    self._previous_encoder_recurrent_state
        #)
        return {'key': 'GET', 'value': {'enc_states': self.enc_states}}
        # segment_size = 1000 * 3
        #return {'key': GET, 'value': {"segment_size": segment_size}}

    def write_action(self):
        model_index = 0
        if self.hyp['states'] is None:
            self.hyp['states'] = self._e2e.dec.init_state(self.enc_states)
        if self.hyp['yseq'][len(self.hyp['yseq'])-1] == self._e2e.dec.eos or len(self.hyp['yseq']) > self.max_len:
            # Finish this sentence is predict EOS
            self.finished = True

        score, states = self._e2e.dec.score(self.hyp['yseq'], self.hyp['states'], self.enc_states)
        score = F.log_softmax(score, dim=1).squeeze()
        # greedy search, take only the (1) best score
        local_best_score, local_best_id = torch.topk(score, 1)

        # [:] is needed!
        self.hyp['states']['z_prev'] = states['z_prev']
        self.hyp['states']['c_prev'] = states['c_prev']
        self.hyp['states']['a_prev'] = states['a_prev']
        self.hyp['states']['workspace'] = states['workspace']
        self.hyp['score'] = self.hyp['score'] + local_best_score[0]
        #self.hyp['yseq'] = [0] * (1 + len(self.hyp['yseq']))
        #self.hyp['yseq'][:len(self.hyp['yseq'])] = self.hyp['yseq']
        #self.hyp['yseq'][len(self.hyp['yseq'])] = int(local_best_id[0])
        #self.hyp['yseq'].append(int(local_best_id[0]))
        #self.hyp['yseq'] = torch.cat(self.hyp['yseq'], torch.tensor([int(local_best_id[0])], dtype=self.dtype, device=self.device))
        self.hyp['yseq'] = torch.cat(self.hyp['yseq'], [local_best_id])

        #if rnnlm:
        #    self.hyp['rnnlm_prev'] = rnnlm_state
        # will be (2 x beam) hyps at most

        return {'key': 'SEND', 'value': {'dec_hyp': self.hyp}}

    def finish_read(self):
        return self.finish_read

    def finish_action(self):
        return {'key': 'SEND', 'value': {'dec_hyp': self._e2e.dec.eos}}

