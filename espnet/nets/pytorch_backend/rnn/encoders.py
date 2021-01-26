import logging
import six

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence

from espnet.nets.e2e_asr_common import get_vgg2l_odim
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet.nets.pytorch_backend.nets_utils import to_device

import math

class RNNP(torch.nn.Module):
    """RNN with projection layer module

    :param int idim: dimension of inputs
    :param int elayers: number of encoder layers
    :param int cdim: number of rnn units (resulted in cdim * 2 if bidirectional)
    :param int hdim: number of projection units
    :param np.ndarray subsample: list of subsampling numbers
    :param float dropout: dropout rate
    :param str typ: The RNN type
    """

    def __init__(self, idim, elayers, cdim, hdim, subsample, dropout, typ="blstm"):
        super(RNNP, self).__init__()
        bidir = typ[0] == "b"
        for i in six.moves.range(elayers):
            if i == 0:
                inputdim = idim
            else:
                inputdim = hdim
            rnn = torch.nn.LSTM(inputdim, cdim, dropout=dropout, num_layers=1, bidirectional=bidir,
                                batch_first=True) if "lstm" in typ \
                else torch.nn.GRU(inputdim, cdim, dropout=dropout, num_layers=1, bidirectional=bidir, batch_first=True)
            setattr(self, "%s%d" % ("birnn" if bidir else "rnn", i), rnn)
            # bottleneck layer to merge
            if bidir:
                setattr(self, "bt%d" % i, torch.nn.Linear(2 * cdim, hdim))
            else:
                setattr(self, "bt%d" % i, torch.nn.Linear(cdim, hdim))

        self.elayers = elayers
        self.cdim = cdim
        self.subsample = subsample
        self.typ = typ
        self.bidir = bidir

    def forward(self, xs_pad, ilens, prev_state=None):
        """RNNP forward

        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax, idim)
        :param torch.Tensor ilens: batch of lengths of input sequences (B)
        :param torch.Tensor prev_state: batch of previous RNN states
        :return: batch of hidden state sequences (B, Tmax, hdim)
        :rtype: torch.Tensor
        """
        # logging.info(self.__class__.__name__ + ' input lengths: ' + str(ilens))
        elayer_states = []
        for layer in six.moves.range(self.elayers):
            xs_pack = pack_padded_sequence(xs_pad, ilens, batch_first=True)
            rnn = getattr(self, ("birnn" if self.bidir else "rnn") + str(layer))
            rnn.flatten_parameters()
            if prev_state is not None and rnn.bidirectional:
                prev_state = reset_backward_rnn_state(prev_state)
            ys, states = rnn(xs_pack, hx=None if prev_state is None else prev_state[layer])
            elayer_states.append(states)
            # ys: utt list of frame x cdim x 2 (2: means bidirectional)
            ys_pad, ilens = pad_packed_sequence(ys, batch_first=True)
            sub = self.subsample[layer + 1]
            if sub > 1:
                ys_pad = ys_pad[:, ::sub]
                ilens = [int(i + 1) // sub for i in ilens]
            # (sum _utt frame_utt) x dim
            projected = getattr(self, 'bt' + str(layer)
                                )(ys_pad.contiguous().view(-1, ys_pad.size(2)))
            if layer == self.elayers - 1:
                xs_pad = projected.view(ys_pad.size(0), ys_pad.size(1), -1)
            else:
                xs_pad = torch.tanh(projected.view(ys_pad.size(0), ys_pad.size(1), -1))

        return xs_pad, ilens, elayer_states  # x: utt list of frame x dim


class RNN(torch.nn.Module):
    """RNN module

    :param int idim: dimension of inputs
    :param int elayers: number of encoder layers
    :param int cdim: number of rnn units (resulted in cdim * 2 if bidirectional)
    :param int hdim: number of final projection units
    :param float dropout: dropout rate
    :param str typ: The RNN type
    """

    def __init__(self, idim, elayers, cdim, hdim, dropout, typ="blstm"):
        super(RNN, self).__init__()
        bidir = typ[0] == "b"
        self.nbrnn = torch.nn.LSTM(idim, cdim, elayers, batch_first=True,
                                   dropout=dropout, bidirectional=bidir) if "lstm" in typ \
            else torch.nn.GRU(idim, cdim, elayers, batch_first=True, dropout=dropout,
                              bidirectional=bidir)
        if bidir:
            self.l_last = torch.nn.Linear(cdim * 2, hdim)
        else:
            self.l_last = torch.nn.Linear(cdim, hdim)
        self.typ = typ

    def forward(self, xs_pad, ilens, prev_state=None):
        """RNN forward

        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax, D)
        :param torch.Tensor ilens: batch of lengths of input sequences (B)
        :param torch.Tensor prev_state: batch of previous RNN states
        :return: batch of hidden state sequences (B, Tmax, eprojs)
        :rtype: torch.Tensor
        """
        logging.info(self.__class__.__name__ + ' input lengths: ' + str(ilens))
        xs_pack = pack_padded_sequence(xs_pad, ilens, batch_first=True)
        self.nbrnn.flatten_parameters()
        if prev_state is not None and self.nbrnn.bidirectional:
            # We assume that when previous state is passed, it means that we're streaming the input
            # and therefore cannot propagate backward BRNN state (otherwise it goes in the wrong direction)
            prev_state = reset_backward_rnn_state(prev_state)
        ys, states = self.nbrnn(xs_pack, hx=prev_state)
        # ys: utt list of frame x cdim x 2 (2: means bidirectional)
        ys_pad, ilens = pad_packed_sequence(ys, batch_first=True)
        # (sum _utt frame_utt) x dim
        projected = torch.tanh(self.l_last(
            ys_pad.contiguous().view(-1, ys_pad.size(2))))
        xs_pad = projected.view(ys_pad.size(0), ys_pad.size(1), -1)
        return xs_pad, ilens, states  # x: utt list of frame x dim


def reset_backward_rnn_state(states):
    """Sets backward BRNN states to zeroes - useful in processing of sliding windows over the inputs"""
    if isinstance(states, (list, tuple)):
        for state in states:
            state[1::2] = 0.
    else:
        states[1::2] = 0.
    return states


class VGG2L(torch.nn.Module):
    """VGG-like module

    :param int in_channel: number of input channels
    """

    def __init__(self, in_channel=1):
        super(VGG2L, self).__init__()
        # CNN layer (VGG motivated)
        self.conv1_1 = torch.nn.Conv2d(in_channel, 64, 3, stride=1, padding=1)
        self.conv1_2 = torch.nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.conv2_1 = torch.nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.conv2_2 = torch.nn.Conv2d(128, 128, 3, stride=1, padding=1)

        self.in_channel = in_channel

    def forward(self, xs_pad, ilens, **kwargs):
        """VGG2L forward

        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax, D)
        :param torch.Tensor ilens: batch of lengths of input sequences (B)
        :return: batch of padded hidden state sequences (B, Tmax // 4, 128 * D // 4)
        :rtype: torch.Tensor
        """
        logging.info(self.__class__.__name__ + ' input lengths: ' + str(ilens))

        # x: utt x frame x dim
        # xs_pad = F.pad_sequence(xs_pad)

        # x: utt x 1 (input channel num) x frame x dim
        xs_pad = xs_pad.view(xs_pad.size(0), xs_pad.size(1), self.in_channel,
                             xs_pad.size(2) // self.in_channel).transpose(1, 2)

        # NOTE: max_pool1d ?
        xs_pad = F.relu(self.conv1_1(xs_pad))
        xs_pad = F.relu(self.conv1_2(xs_pad))
        xs_pad = F.max_pool2d(xs_pad, 2, stride=2, ceil_mode=True)

        xs_pad = F.relu(self.conv2_1(xs_pad))
        xs_pad = F.relu(self.conv2_2(xs_pad))
        xs_pad = F.max_pool2d(xs_pad, 2, stride=2, ceil_mode=True)
        if torch.is_tensor(ilens):
            ilens = ilens.cpu().numpy()
        else:
            ilens = np.array(ilens, dtype=np.float32)
        ilens = np.array(np.ceil(ilens / 2), dtype=np.int64)
        ilens = np.array(
            np.ceil(np.array(ilens, dtype=np.float32) / 2), dtype=np.int64).tolist()

        # x: utt_list of frame (remove zeropaded frames) x (input channel num x dim)
        xs_pad = xs_pad.transpose(1, 2)
        xs_pad = xs_pad.contiguous().view(
            xs_pad.size(0), xs_pad.size(1), xs_pad.size(2) * xs_pad.size(3))
        return xs_pad, ilens, None  # no state in this layer


class Encoder(torch.nn.Module):
    """Encoder module

    :param str etype: type of encoder network
    :param int idim: number of dimensions of encoder network
    :param int elayers: number of layers of encoder network
    :param int eunits: number of lstm units of encoder network
    :param int eprojs: number of projection units of encoder network
    :param np.ndarray subsample: list of subsampling numbers
    :param float dropout: dropout rate
    :param int in_channel: number of input channels
    """

    def __init__(self, etype, idim, elayers, eunits, eprojs, subsample, dropout, in_channel=1):
        super(Encoder, self).__init__()
        typ = etype.lstrip("vgg").rstrip("p")
        self.etype = etype
        self.eunits = eunits
        if typ not in ['lstm', 'gru', 'blstm', 'bgru']:
            logging.error("Error: need to specify an appropriate encoder architecture")

        if etype.startswith("vgg"):
            if etype[-1] == "p":
                self.enc = torch.nn.ModuleList([VGG2L(in_channel),
                                                RNNP(get_vgg2l_odim(idim, in_channel=in_channel), elayers, eunits,
                                                     eprojs,
                                                     subsample, dropout, typ=typ)])
                logging.info('Use CNN-VGG + ' + typ.upper() + 'P for encoder')
            else:
                self.enc = torch.nn.ModuleList([VGG2L(in_channel),
                                                RNN(get_vgg2l_odim(idim, in_channel=in_channel), elayers, eunits,
                                                    eprojs,
                                                    dropout, typ=typ)])
                logging.info('Use CNN-VGG + ' + typ.upper() + ' for encoder')
        else:
            if etype[-1] == "p":
                self.enc = torch.nn.ModuleList(
                    [RNNP(idim, elayers, eunits, eprojs, subsample, dropout, typ=typ)])
                logging.info(typ.upper() + ' with every-layer projection for encoder')
            else:
                self.enc = torch.nn.ModuleList([RNN(idim, elayers, eunits, eprojs, dropout, typ=typ)])
                logging.info(typ.upper() + ' without projection for encoder')

    def forward_def(self, xs_pad, ilens, prev_states=None):
        """Encoder forward

    #    :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax, D)
    #    :param torch.Tensor ilens: batch of lengths of input sequences (B)
    #    :param torch.Tensor prev_state: batch of previous encoder hidden states (?, ...)
    #    :return: batch of hidden state sequences (B, Tmax, eprojs)
    #    :rtype: torch.Tensor
    #    """
        if prev_states is None:
            prev_states = [None] * len(self.enc)
        assert len(prev_states) == len(self.enc)

        current_states = []
        for module, prev_state in zip(self.enc, prev_states):
            xs_pad, ilens, states = module(xs_pad, ilens, prev_state=prev_state)
            print('x_pad_ encoded', xs_pad.size(), ilens[0])
            current_states.append(states)

        # make mask to remove bias value in padded part
        mask = to_device(self, make_pad_mask(ilens).unsqueeze(-1))

        return xs_pad.masked_fill(mask, 0.0), ilens, current_states

    def forward(self, xs_pad, ilens, k, s):
        """Encoder forward

        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax, D)
        :param torch.Tensor ilens: batch of lengths of input sequences (B)
        :param torch.Tensor prev_state: batch of previous encoder hidden states (?, ...)
        :return: batch of hidden state sequences (B, Tmax, eprojs)
        :rtype: torch.Tensor
        """
        #if prev_states is None:
        #    prev_states = [None] * len(self.enc)
        #assert len(prev_states) == len(self.enc)
        """
        encoder_output = []
        current_states = []
        ilens_out = []
        g = k
        Tmax = xs_pad.size(1)
        prev_states = [None] * len(self.enc)
        offset = 0
        u_xs_pad_buff = torch.empty((xs_pad.size(0), 0, self.eunits), device=xs_pad.device)
        ilen_buff = [0] * xs_pad.size(0)
        #u_xs_pad_buff = None
        print(self.etype, self.eunits, len(self.enc), xs_pad.size())
        # B, Tmax, D -> Tmax, B, D
        #xs_pad = xs_pad.transpose(0, 1)
        ilens_ = torch.zeros(ilens.size(), dtype=ilens.dtype, device=ilens.device)
        while (g < Tmax):
            if "b" in self.etype:
                #prev_states = [None] * len(self.enc)
                xs_pad_ = xs_pad.transpose(1, 2)[:, :, :g].transpose(1, 2)
                ilens_ = torch.zeros(ilens.size(), dtype=ilens.dtype, device=ilens.device)
                ilens_ = ilens_.new_full(ilens.size(), fill_value=g)
                print(xs_pad_.size(), ilens_.size())
            else:
                #xs_pad_ = xs_pad.transpose(1, 2)[:, :, offset:g].transpose(1, 2)
                xs_pad_ = xs_pad.transpose(0, 1).clone()[offset:g].transpose(0, 1)
                ilens_ = ilens_.new_full(ilens.size(), fill_value=(g - offset))
                print(xs_pad_.size(), ilens_.size())

            assert len(prev_states) == len(self.enc)

            current_states_ = []
            for module, prev_state in zip(self.enc, prev_states):
                xs_pad_, ilens_, states = module(xs_pad_, ilens_, prev_state=prev_state)
                print('x_pad_ encoded', xs_pad_.size(), ilens_[0])
                current_states_.append(states)

            # make mask to remove bias value in padded part
            mask = to_device(self, make_pad_mask(ilens_).unsqueeze(-1))
            if "b" in self.etype:
                encoder_output.append(xs_pad_.masked_fill(mask, 0.0))
                ilens_out.append(ilens_)
            else:
                prev_states = current_states_
                #print("enc", xs_pad_.size(), u_xs_pad_buff.size(), xs_pad.size())
                u_xs_pad_buff = torch.cat((u_xs_pad_buff, xs_pad_), dim=1)
                encoder_output.append(u_xs_pad_buff)
                ilen_buff = [x + y for x, y in zip(ilen_buff, ilens_)]
                ilens_out.append(ilen_buff)
                offset = g
            current_states.append(current_states_)
            g += s

        # g = Tmax
        current_states_ = []
        assert len(prev_states) == len(self.enc)
        if "b" in self.etype:
            xs_pad_ = xs_pad
            ilens_ = ilens
        elif offset < Tmax:
            #xs_pad_ = xs_pad.transpose(1, 2)[:, :, offset:Tmax].transpose(1, 2)
            xs_pad_ = xs_pad.transpose(0, 1).clone()[offset:Tmax].transpose(0, 1)
            #ilens_ = torch.zeros(ilens.size(), dtype=ilens.dtype, device=ilens.device)
            ilens_ = ilens_.new_full(ilens.size(), fill_value=(Tmax - offset))
        for module, prev_state in zip(self.enc, prev_states):
            xs_pad_, ilens_, states = module(xs_pad_, ilens_, prev_state=prev_state)
            current_states_.append(states)

        if "b" in self.etype:
            mask = to_device(self, make_pad_mask(ilens_).unsqueeze(-1))
            encoder_output.append(xs_pad_.masked_fill(mask, 0.0))
            ilens_out.append(ilens_)
            #print('enc: ', prev_states, ilens_, xs_pad_.size())
        elif offset < Tmax:
            u_xs_pad_buff = torch.cat((u_xs_pad_buff, xs_pad_), dim=1)
            encoder_output.append(u_xs_pad_buff)
            ilen_buff = [x + y for x, y in zip(ilen_buff, ilens_)]
            ilens_out.append(ilen_buff)
            #print('enc: ', prev_states, ilens_, xs_pad_.size(), u_xs_pad_buff.size())
        """

        encoder_output = None
        current_states = []
        ilens_out = []
        g = k
        Tmax = xs_pad.size(1)
        prev_states = [None] * len(self.enc)
        offset = 0
        u_xs_pad_buff = torch.empty((xs_pad.size(0), 0, self.eunits), device=xs_pad.device)
        ilen_buff = [0] * xs_pad.size(0)
        # u_xs_pad_buff = None
        print(self.etype, self.eunits, len(self.enc), xs_pad.size())
        # B, Tmax, D -> Tmax, B, D
        # xs_pad = xs_pad.transpose(0, 1)
        ilens_ = torch.zeros(ilens.size(), dtype=ilens.dtype, device=ilens.device)
        while (g < Tmax):
            xs_pad_ = xs_pad.transpose(0, 1).detach().clone()[:g].transpose(0, 1)
            ilens_ = ilens_.new_full(ilens.size(), fill_value=g).detach()
            print(xs_pad_.size(), ilens_.size())
            current_states_ = []
            for module, prev_state in zip(self.enc, prev_states):
                xs_pad_, ilens_, states = module(xs_pad_, ilens_, prev_state=prev_state)
                print('x_pad_ encoded', xs_pad_.size(), ilens_[0])
                if states is not None:
                    for state in states:
                        print('state: ', state.size())
                #current_states_.append(states)
            mask = to_device(self, make_pad_mask(ilens_).unsqueeze(-1))
            #encoder_output.append(xs_pad_.masked_fill(mask, 0.0))
            if encoder_output is None:
                encoder_output = xs_pad_.masked_fill(mask, 0.0)
            else:
                encoder_output = torch.cat((encoder_output, xs_pad_.masked_fill(mask, 0.0)), dim=1)
            #ilens_out.append(ilens_)
            #current_states.append(current_states_)
            g += s

        """
        encoder_output = []
        current_states = []
        ilens_out = []
        g = k
        Tmax = xs_pad.size(1)
        prev_states = [None] * len(self.enc)
        assert len(prev_states) == len(self.enc)
        #ilens_ = torch.zeros(ilens.size(), dtype=ilens.dtype, device=ilens.device)
        current_states = []
        for module, prev_state in zip(self.enc, prev_states):
            xs_pad, ilens, states = module(xs_pad, ilens, prev_state=prev_state)
            current_states.append(states)

        # make mask to remove bias value in padded part
        mask = to_device(self, make_pad_mask(ilens).unsqueeze(-1))
        xs_pad.masked_fill(mask, 0.0)

        while (g < Tmax):
            lens_ = math.ceil(g / 2)
            lens_ = math.ceil(lens_ / 2)
            print(type(lens_), lens_)
            xs_pad_ = xs_pad.transpose(0, 1).clone()[:lens_].transpose(0, 1)
            print('x_pad_ encoded', xs_pad_.size(), ilens.new_full(ilens.size(), fill_value=lens_))
            encoder_output.append(xs_pad_)
            #ilens_.new_full(ilens_.size(), fill_value=lens_).detach()
            ilens_out.append(ilens.new_full(ilens.size(), fill_value=lens_))
            g += s
        encoder_output.append(xs_pad)
        print('x_pad_ encoded', xs_pad.size(), ilens)
        ilens_out.append(ilens)
        """

        #current_states.append(current_states_)
        print("enc ends")
        return {
            "encoder_output": encoder_output,
            "ilens": ilens_out,
            "current_states": current_states
        }


def encoder_for(args, idim, subsample):
    """Instantiates an encoder module given the program arguments

    :param Namespace args: The arguments
    :param int or List of integer idim: dimension of input, e.g. 83, or
                                        List of dimensions of inputs, e.g. [83,83]
    :param List or List of List subsample: subsample factors, e.g. [1,2,2,1,1], or
                                        List of subsample factors of each encoder. e.g. [[1,2,2,1,1], [1,2,2,1,1]]
    :rtype torch.nn.Module
    :return: The encoder module
    """
    num_encs = getattr(args, "num_encs", 1)  # use getattr to keep compatibility
    if num_encs == 1:
        # compatible with single encoder asr mode
        return Encoder(args.etype, idim, args.elayers, args.eunits, args.eprojs, subsample, args.dropout_rate)
    elif num_encs >= 1:
        enc_list = torch.nn.ModuleList()
        for idx in range(num_encs):
            enc = Encoder(args.etype[idx], idim[idx], args.elayers[idx], args.eunits[idx], args.eprojs, subsample[idx],
                          args.dropout_rate[idx])
            enc_list.append(enc)
        return enc_list
    else:
        raise ValueError("Number of encoders needs to be more than one. {}".format(num_encs))
