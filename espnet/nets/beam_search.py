"""Beam search module."""

from itertools import chain
import logging
from typing import Any
from typing import Dict
from typing import List
from typing import NamedTuple
from typing import Tuple

import torch
import torch.nn.functional as F

from espnet.nets.e2e_asr_common import end_detect
from espnet.nets.scorer_interface import PartialScorerInterface
from espnet.nets.scorer_interface import ScorerInterface


class Hypothesis(NamedTuple):
    """Hypothesis data type."""

    yseq: torch.Tensor
    score: float = 0
    scores: Dict[str, float] = dict()
    states: Dict[str, Dict] = dict()

    def asdict(self) -> dict:
        """Convert data to JSON-friendly dict."""
        return self._replace(
            yseq=self.yseq.tolist(),
            score=float(self.score),
            scores={k: float(v) for k, v in self.scores.items()}
        )._asdict()


class BeamSearch(torch.nn.Module):
    """Beam search implementation."""

    def __init__(self, scorers_pool: List[Dict[str, ScorerInterface]], weights: Dict[str, float],
                 beam_size: int, vocab_size: int,
                 sos: int, eos: int, token_list: List[str] = None,
                 pre_beam_ratio: float = 1.5, pre_beam_score_key: str = "decoder"):
        """Initialize beam search.

        Args:
            scorers (dict[str, ScorerInterface]): Dict of decoder modules e.g., Decoder, CTCPrefixScorer, LM
                The scorer will be ignored if it is `None`
            weights (dict[str, float]): Dict of weights for each scorers
                The scorer will be ignored if its weight is 0
            beam_size (int): The number of hypotheses kept during search
            vocab_size (int): The number of vocabulary
            sos (int): Start of sequence id
            eos (int): End of sequence id
            token_list (list[str]): List of tokens for debug log
            pre_beam_score_key (str): key of scores to perform pre-beam search
            pre_beam_ratio (float): beam size in the pre-beam search will be `int(pre_beam_ratio * beam_size)`

        """
        super().__init__()
        # set scorers
        self.weights = weights
        #self.full_scorers = dict()
        #self.part_scorers = dict()
        self.full_scorers = []
        self.part_scorers = []
        self.nn_dict = torch.nn.ModuleDict()
        for scorers in scorers_pool:
            full_scorers_ = dict()
            part_scorers_ = dict()
            for k, v in scorers.items():
                w = weights.get(k, 0)
                if w == 0 or v is None:
                    continue
                assert isinstance(v, ScorerInterface), f"{k} ({type(v)}) does not implement ScorerInterface"
                if isinstance(v, PartialScorerInterface):
                    part_scorers_[k] = v
                else:
                    full_scorers_[k] = v
                if isinstance(v, torch.nn.Module):
                    self.nn_dict[k] = v
            self.part_scorers.append(part_scorers_)
            self.full_scorers.append(full_scorers_)

        # set configurations
        self.sos = sos
        self.eos = eos
        self.token_list = token_list
        self.pre_beam_size = int(pre_beam_ratio * beam_size)
        self.beam_size = beam_size
        self.n_vocab = vocab_size
        self.pre_beam_score_key = pre_beam_score_key

    def init_hyp(self, x: torch.Tensor) -> Hypothesis:
        """Get an initial hypothesis data.

        Args:
            x (torch.Tensor): The encoder output feature

        Returns:
            Hypothesis: The initial hypothesis.

        """
        #init_states = dict()
        init_scores = dict()
        init_states = []
        #init_scores = []
        for idx, x_ in enumerate(x):
            init_states_ = dict()
            #init_scores_ = dict()
            for k, d in chain(self.full_scorers[idx].items(), self.part_scorers[idx].items()):
                init_states_[k] = d.init_state(x_)
                #init_scores_[k] = 0.0
            init_states.append(init_states_)
            #init_scores.append(init_scores_)
        for k, d in chain(self.full_scorers[idx].items(), self.part_scorers[idx].items()):
            init_scores[k] = 0.0
        return Hypothesis(
            score=0.0, scores=init_scores, states=init_states,
            yseq=torch.tensor([self.sos], device=x[0].device))

    @staticmethod
    def append_token(xs: torch.Tensor, x: int) -> torch.Tensor:
        """Append new token to prefix tokens.

        Args:
            xs (torch.Tensor): The prefix token
            x (int): The new token to append

        Returns:
            torch.Tensor: New tensor contains: xs + [x] with xs.dtype and xs.device

        """
        x = torch.tensor([x], dtype=xs.dtype, device=xs.device)
        return torch.cat((xs, x))

    def score(self, hyp: Hypothesis, x: torch.Tensor, idx) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        """Score new hypothesis by `self.full_scorers`.

        Args:
            hyp (Hypothesis): Hypothesis with prefix tokens to score
            x (torch.Tensor): Corresponding input feature

        Returns:
            Tuple[Dict[str, torch.Tensor], Dict[str, Any]]: Tuple of
                score dict of `hyp` that has string keys of `self.full_scorers`
                and tensor score values of shape: `(self.n_vocab,)`,
                and state dict that has string keys and state values of `self.full_scorers`

        """
        scores = dict()
        states = dict()
        for k, d in self.full_scorers[idx].items():
            scores[k], states[k] = d.score(hyp.yseq, hyp.states[idx][k], x[idx])
        return scores, states

    def score_partial(self, hyp: Hypothesis, ids: torch.Tensor, x: torch.Tensor, idx) \
            -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        """Score new hypothesis by `self.part_scorers`.

        Args:
            hyp (Hypothesis): Hypothesis with prefix tokens to score
            ids (torch.Tensor): 1D tensor of new partial tokens to score
            x (torch.Tensor): Corresponding input feature

        Returns:
            Tuple[Dict[str, torch.Tensor], Dict[str, Any]]: Tuple of
                score dict of `hyp` that has string keys of `self.part_scorers`
                and tensor score values of shape: `(len(ids),)`,
                and state dict that has string keys and state values of `self.part_scorers`

        """
        scores = dict()
        states = dict()
        for k, d in self.part_scorers[idx].items():
            scores[k], states[k] = d.score_partial(hyp.yseq, ids, hyp.states[idx][k], x[idx])
        return scores, states

    def pre_beam(self, scores: Dict[str, torch.Tensor], device: torch.device) -> torch.Tensor:
        """Compute topk token ids for `self.part_scorers`.

        Args:
            scores (Dict[str, torch.Tensor]): The score dict of `hyp` that has string keys of `self.full_scorers`
                and tensor score values; its shape is `(self.n_vocab,)`,
            device (torch.device): The device to compute topk

        Returns:
            torch.Tensor: The partial tokens ids for `self.part_scorers`

        """
        if self.pre_beam_size < self.n_vocab and self.pre_beam_score_key in scores:
            return torch.topk(scores[self.pre_beam_score_key], self.pre_beam_size)[1]
        else:
            return torch.arange(self.n_vocab, device=device)

    def main_beam(self, weighted_scores: torch.Tensor, ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute topk full token ids and partial token ids.

        Args:
            weighted_scores (torch.Tensor): The weighted sum scores for each tokens. Its shape is `(self.n_vocab,)`.
            ids (torch.Tensor): The partial token ids to compute topk

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The topk full token ids and partial token ids.
                Their shapes are `(self.beam_size,)`

        """
        # no pre beam performed
        if weighted_scores.size(0) == ids.size(0):
            top_ids = weighted_scores.topk(self.beam_size)[1]
            return top_ids, top_ids

        # mask pruned in pre-beam not to select in topk
        tmp = weighted_scores[ids]
        weighted_scores[:] = -float("inf")
        weighted_scores[ids] = tmp
        top_ids = weighted_scores.topk(self.beam_size)[1]
        local_ids = weighted_scores[ids].topk(self.beam_size)[1]
        return top_ids, local_ids

    @staticmethod
    def merge_scores(hyp: Hypothesis, scores: Dict[str, torch.Tensor], idx: int,
                     part_scores: Dict[str, torch.Tensor], part_idx: int) -> Dict[str, torch.Tensor]:
        """Merge scores for new hypothesis.

        Args:
            hyp (Hypotheis): The previous hypothesis of prefix tokens
            scores (Dict[str, torch.Tensor]): scores by `self.full_scorers`
            idx (int): The new token id
            part_scores (Dict[str, torch.Tensor]): scores of partial tokens by `self.part_scorers`
            part_idx (int): The new token id for `part_scores`

        Returns:
            Dict[str, torch.Tensor]: The new score dict.
                Its keys are names of `self.full_scorers` and `self.part_scorers`.
                Its values are scalar tensors by the scorers.

        """
        new_scores = dict()
        for k, v in scores.items():
            new_scores[k] = hyp.scores[k] + v[idx]
        for k, v in part_scores.items():
            new_scores[k] = v[part_idx]
        return new_scores

    def merge_states(self, states: Any, part_states: Any, part_idx: int) -> Any:
        """Merge states for new hypothesis.

        Args:
            states: states of `self.full_scorers`
            part_states: states of `self.part_scorers`
            part_idx (int): The new token id for `part_scores`

        Returns:
            Dict[str, torch.Tensor]: The new score dict.
                Its keys are names of `self.full_scorers` and `self.part_scorers`.
                Its values are states of the scorers.

        """
        new_states = []
        for i, states_ in enumerate(states):
            new_states_ = dict()
            for k, v in states_.items():
                new_states_[k] = v
            for k, d in self.part_scorers[i].items():
                new_states_[k] = d.select_state(part_states[k], part_idx)
            new_states.append(states_)
        return new_states

    def top_beam_hyps(self, hyps: List[Hypothesis]) -> List[Hypothesis]:
        """Get top `self.beam_size` hypothesis."""
        return sorted(hyps, key=lambda x: x.score, reverse=True)[:min(len(hyps), self.beam_size)]

    def forward(self, x: torch.Tensor, maxlenratio: float = 0.0, minlenratio: float = 0.0) -> List[Hypothesis]:
        """Perform beam search.

        Args:
            x (torch.Tensor): Encoded speech feature (T, D)
            maxlenratio (float): Input length ratio to obtain max output length.
                If maxlenratio=0.0 (default), it uses a end-detect function
                to automatically find maximum hypothesis lengths
            minlenratio (float): Input length ratio to obtain min output length.

        Returns:
            list[Hypothesis]: N-best decoding results

        """
        # set length bounds
        if maxlenratio == 0:
            maxlen = x[0].shape[0]
        else:
            maxlen = max(1, int(maxlenratio * x[0].size(0)))
        minlen = int(minlenratio * x[0].size(0))
        logging.info('max output length: ' + str(maxlen))
        logging.info('min output length: ' + str(minlen))

        # main loop of prefix search
        running_hyps = [self.init_hyp(x)]
        ended_hyps = []
        for i in range(maxlen):
            logging.debug('position ' + str(i))
            best = []
            for hyp in running_hyps:
                scores = [None] * len(x)
                states = [None] * len(x)
                #part_ids = [None] * len(x)
                part_scores = [None] * len(x)
                part_states = [None] * len(x)
                for idx in range(len(x)):
                    scores[idx], states[idx] = self.score(hyp, x, idx)
                    part_ids_ = self.pre_beam(scores[idx], device=x[idx].device)
                    part_scores[idx], part_states[idx] = self.score_partial(hyp, part_ids_, x, idx)

                full_avg_scores = dict()
                part_avg_scores = dict()
                for k in self.full_scorers[0]:
                    score_k = [score[k] for score in scores]
                    full_avg_scores[k] = torch.mean(torch.stack(score_k), dim=0)
                    if k == "decoder":
                        print(score_k)
                        full_avg_scores[k] = F.log_softmax(full_avg_scores[k], dim=1).squeeze()
                        print(full_avg_scores[k])
                        aaaaaaaaaaaaa
                for k in self.part_scorers[0]:
                    score_k = [score[k] for score in part_scores]
                    part_avg_scores[k] = torch.mean(torch.stack(score_k), dim=0)

                part_ids = self.pre_beam(full_avg_scores, device=x[0].device)

                # weighted sum scores
                weighted_scores = torch.zeros(self.n_vocab, dtype=x[0].dtype, device=x[0].device)
                for k in self.full_scorers[0]:
                    weighted_scores += self.weights[k] * full_avg_scores[k]
                for k in self.part_scorers[0]:
                    weighted_scores[part_ids] += self.weights[k] * part_avg_scores[k]
                weighted_scores += hyp.score

                # update hyps
                for j, part_j in zip(*self.main_beam(weighted_scores, part_ids)):
                    # will be (2 x beam at most)
                    best.append(Hypothesis(
                        score=(weighted_scores[j]),
                        yseq=self.append_token(hyp.yseq, j),
                        scores=self.merge_scores(hyp, full_avg_scores, j, part_avg_scores, part_j),
                        states=self.merge_states(states, part_states, part_j)))

                # sort and prune 2 x beam -> beam
                best = self.top_beam_hyps(best)

            # post process of one iteration
            running_hyps = self.post_process(i, maxlen, maxlenratio, best, ended_hyps)
            if len(running_hyps) == 0:
                logging.info('no hypothesis. Finish decoding.')
                break

        nbest_hyps = sorted(ended_hyps, key=lambda x: x.score, reverse=True)
        # check number of hypotheis
        if len(nbest_hyps) == 0:
            logging.warning('there is no N-best results, perform recognition again with smaller minlenratio.')
            return [] if minlenratio < 0.1 else self.forward(x, maxlenratio, max(0.0, minlenratio - 0.1))

        # report the best result
        best = nbest_hyps[0]
        logging.info(f'total log probability: {best.score}')
        logging.info(f'normalized log probability: {best.score / len(best.yseq)}')
        return nbest_hyps

    def post_process(self, i: int, maxlen: int, maxlenratio: float,
                     running_hyps: List[Hypothesis], ended_hyps: List[Hypothesis]) -> List[Hypothesis]:
        """Perform post-processing of beam search iterations.

        Args:
            i (int): The length of hypothesis tokens.
            maxlen (int): The maximum length of tokens in beam search.
            maxlenratio (int): The maximum length ratio in beam search.
            running_hyps (List[Hypothesis]): The running hypotheses in beam search.
            ended_hyps (List[Hypothesis]): The ended hypotheses in beam search.

        Returns:
            List[Hypothesis]: The new running hypotheses.

        """
        logging.debug(f'the number of running hypothes: {len(running_hyps)}')
        if self.token_list is not None:
            logging.debug("best hypo: " + "".join([self.token_list[x] for x in running_hyps[0].yseq[1:]]))
        # add eos in the final loop to avoid that there are no ended hyps
        if i == maxlen - 1:
            logging.info("adding <eos> in the last position in the loop")
            running_hyps = [h._replace(yseq=self.append_token(h.yseq, self.eos)) for h in running_hyps]

        # add ended hypotheses to a final list, and removed them from current hypotheses
        # (this will be a probmlem, number of hyps < beam)
        remained_hyps = []
        for hyp in running_hyps:
            if hyp.yseq[-1] == self.eos:
                # e.g., Word LM needs to add final <eos> score
                for k, d in chain(self.full_scorers[0].items(), self.part_scorers[0].items()):
                    s = d.final_score(hyp.states[0][k])
                    hyp.scores[k] += s
                    hyp = hyp._replace(score=hyp.score + self.weights[k] * s)
                ended_hyps.append(hyp)
            else:
                remained_hyps.append(hyp)
        # end detection
        if maxlenratio == 0.0 and end_detect([h.asdict() for h in ended_hyps], i):
            logging.info(f'end detected at {i}')
            return []
        if len(remained_hyps) > 0:
            logging.debug(f'remeined hypothes: {len(remained_hyps)}')
        return remained_hyps


def beam_search(x: torch.Tensor, sos: int, eos: int, beam_size: int, vocab_size: int,
                scorers: Dict[str, ScorerInterface], weights: Dict[str, float],
                token_list: List[str] = None, maxlenratio: float = 0.0, minlenratio: float = 0.0,
                pre_beam_ratio: float = 1.5, pre_beam_score_key: str = "decoder") -> list:
    """Perform beam search with scorers.

    Args:
        x (torch.Tensor): Encoded speech feature (T, D)
        sos (int): Start of sequence id
        eos (int): End of sequence id
        beam_size (int): The number of hypotheses kept during search
        vocab_size (int): The number of vocabulary
        scorers (dict[str, ScorerInterface]): Dict of decoder modules e.g., Decoder, CTCPrefixScorer, LM
            The scorer will be ignored if it is `None`
        weights (dict[str, float]): Dict of weights for each scorers
            The scorer will be ignored if its weight is 0
        token_list (list[str]): List of tokens for debug log
        maxlenratio (float): Input length ratio to obtain max output length.
            If maxlenratio=0.0 (default), it uses a end-detect function
            to automatically find maximum hypothesis lengths
        minlenratio (float): Input length ratio to obtain min output length.
        pre_beam_score_key (str): key of scores to perform pre-beam search
        pre_beam_ratio (float): beam size in the pre-beam search will be `int(pre_beam_ratio * beam_size)`

    Returns:
        list: N-best decoding results

    """
    ret = BeamSearch(
        scorers, weights,
        beam_size=beam_size,
        vocab_size=vocab_size,
        pre_beam_ratio=pre_beam_ratio,
        pre_beam_score_key=pre_beam_score_key,
        sos=sos,
        eos=eos,
        token_list=token_list,
    ).forward(
        x=x,
        maxlenratio=maxlenratio,
        minlenratio=minlenratio)
    return [h.asdict() for h in ret]
