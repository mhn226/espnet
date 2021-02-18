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
        print('#################')
        print(xs_pad.size())
        # NOTE: max_pool1d ?
        xs_pad = F.relu(self.conv1_1(xs_pad))
        print(xs_pad.size())
        xs_pad = F.relu(self.conv1_2(xs_pad))
        print(xs_pad.size())
        xs_pad = F.max_pool2d(xs_pad, 2, stride=2, ceil_mode=True)
        print(xs_pad.size())
        xs_pad = F.relu(self.conv2_1(xs_pad))
        print(xs_pad.size())
        xs_pad = F.relu(self.conv2_2(xs_pad))
        print(xs_pad.size())
        xs_pad = F.max_pool2d(xs_pad, 2, stride=2, ceil_mode=True)
        print(xs_pad.size())
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
        print(xs_pad.size())
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

    def forward(self, xs_pad, ilens, prev_states=None):
        """Encoder forward

        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax, D)
        :param torch.Tensor ilens: batch of lengths of input sequences (B)
        :param torch.Tensor prev_state: batch of previous encoder hidden states (?, ...)
        :return: batch of hidden state sequences (B, Tmax, eprojs)
        :rtype: torch.Tensor
        """
        if prev_states is None:
            prev_states = [None] * len(self.enc)
        assert len(prev_states) == len(self.enc)

        """
        current_states = []
        for module, prev_state in zip(self.enc, prev_states):
           xs_pad, ilens, states = module(xs_pad, ilens, prev_state=prev_state)
           current_states.append(states)

        print(xs_pad.size(), ilens)
        # make mask to remove bias value in padded part
        mask = to_device(self, make_pad_mask(ilens).unsqueeze(-1))
        """

        """
        # Test
        prev_state = None
        g = 200
        s = 20
        offset = 0
        xs_pad_, ilens_, states = self.enc[0](xs_pad, ilens, prev_state=prev_states[0])
        print(xs_pad.size(), xs_pad_.size())
        output = None
        while (g < xs_pad.size(1)):
            g_ = np.array(np.ceil(torch.tensor([g]) / 2), dtype=np.int64)
            g_ = np.array(
                np.ceil(np.array(g_, dtype=np.float32) / 2), dtype=np.int64).tolist()
            g_ = g_[0]
            xs_pad2, ilens2, prev_state = self.enc[1](xs_pad_.transpose(0, 1)[offset:g_].transpose(0, 1),
                                                      [g_-offset], prev_state=prev_state)
            if output is None:
                output = xs_pad2.squeeze(0)
            else:
                print(output.size(), xs_pad2.size())
                output = torch.cat((output, xs_pad2.squeeze(0)))
            offset = g_
            g += s
        if (g >= xs_pad.size(1)):
            #g = xs_pad.size(1)
            #g_ = np.array(np.ceil(torch.tensor([g]) / 2), dtype=np.int64)
            #g_ = np.array(
            #    np.ceil(np.array(g_, dtype=np.float32) / 2), dtype=np.int64).tolist()
            #g_ = g_[0]
            g_ = ilens_[0]
            xs_pad2, ilens2, prev_state = self.enc[1](xs_pad_.transpose(0, 1)[offset:g_].transpose(0, 1),
                                                      [g_ - offset], prev_state=prev_state)
            if output is None:

                output = xs_pad2.squeeze(0)
            else:
                output = torch.cat((output, xs_pad2.squeeze(0)))
        mask = to_device(self, make_pad_mask(torch.tensor(ilens_)).unsqueeze(-1))
        # End test

        current_states = []


        #return xs_pad.masked_fill(mask, 0.0), ilens, current_states
        print(output.size())
        print(output.unsqueeze(0).size())
        print(ilens_, ilens2, torch.tensor(ilens_))
        return output.unsqueeze(0).masked_fill(mask, 0.0), torch.tensor(ilens_), current_states
        """

        """
        # Test
        prev_state = None
        g = 200
        s = 20
        offset = 0
        output = None
        current_states = []
        o_ilens = None
        while (g < xs_pad.size(1)):
            current_states = []
            xs_pad_ = xs_pad.transpose(0, 1)[offset:g].transpose(0, 1)
            ilens_ = torch.tensor([g-offset])
            for module, prev_state in zip(self.enc, prev_states):
                xs_pad_, ilens_, states = module(xs_pad_, ilens_, prev_state=prev_state)
                current_states.append(states)
            prev_states[0] = None
            prev_states[1] = states
            if output is None:
                output = xs_pad_.squeeze(0)
                o_ilens = ilens_
            else:
                output = torch.cat((output, xs_pad_.squeeze(0)))
                o_ilens += ilens_
            offset = g
            g += s
        if (g >= xs_pad.size(1)):
            g = xs_pad.size(1)
            xs_pad_ = xs_pad.transpose(0, 1)[offset:g].transpose(0, 1)
            ilens_ = torch.tensor([g - offset])
            for module, prev_state in zip(self.enc, prev_states):
                xs_pad_, ilens_, states = module(xs_pad_, ilens_, prev_state=prev_state)
                current_states.append(states)
            prev_states[0] = None
            prev_states[1] = states
            if output is None:
                output = xs_pad_.squeeze(0)
                o_ilens = ilens_
            else:
                output = torch.cat((output, xs_pad_.squeeze(0)))
                o_ilens += ilens_
        mask = to_device(self, make_pad_mask(torch.tensor(o_ilens)).unsqueeze(-1))
        # End test

        current_states = []

        # return xs_pad.masked_fill(mask, 0.0), ilens, current_states
        print(output.size())
        print(output.unsqueeze(0).size())
        print(ilens_, torch.tensor(ilens_), o_ilens)
        """

        prev_state = None
        g = 200
        s = 20
        offset = 0
        output = None
        out_vgg = None
        o_ilens = None
        current_states = []
        while (g < xs_pad.size(1)):
            xs_pad_, ilens_, prev_state = self.enc[0](xs_pad.transpose(0, 1)[offset:g].transpose(0, 1),
                                                      torch.tensor([g - offset]), prev_state=prev_state)
            if out_vgg is None:
                out_vgg = xs_pad_.squeeze(0)
                o_ilens = ilens_
            else:
                out_vgg[-1] = xs_pad_.squeeze(0)[0]
                out_vgg = torch.cat((out_vgg, xs_pad_.squeeze(0)[1:]))
                print(out_vgg.size())
                o_ilens += [ilens_[0] - 1]
            offset = g - 4
            g += s
        if (g >= xs_pad.size(1)):
            g = xs_pad.size(1)
            xs_pad_, ilens_, prev_state = self.enc[0](xs_pad.transpose(0, 1)[offset:g].transpose(0, 1),
                                                      torch.tensor([g - offset]), prev_state=prev_state)
            print(xs_pad_.size(), ilens_)
            if out_vgg is None:
                out_vgg = xs_pad_.squeeze(0)
                o_ilens = ilens_
            else:
                #out_vgg = torch.cat((out_vgg, xs_pad_.squeeze(0)))
                #o_ilens += ilens_
                out_vgg[-1] = xs_pad_.squeeze(0)[0]
                out_vgg = torch.cat((out_vgg, xs_pad_.squeeze(0)[1:]))
                o_ilens += [ilens_[0] - 1]
        o_ilens = [sum(o_ilens)]
        print(out_vgg.unsqueeze(0).size(), o_ilens)
        xs_pad, ilens, _ = self.enc[1](out_vgg.unsqueeze(0), o_ilens, prev_state=None)
        mask = to_device(self, make_pad_mask(torch.tensor(ilens)).unsqueeze(-1))
        return xs_pad.masked_fill(mask, 0.0), o_ilens, current_states

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
