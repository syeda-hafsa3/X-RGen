from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import modules.utils as utils

class CaptionModel(nn.Module):
    def __init__(self):
        super(CaptionModel, self).__init__()

    # Implements beam search
    # Calls beam_step and returns the final set of beams
    # Augments log-probabilities with diversity terms when number of groups > 1
    def forward(self, *args, **kwargs):
        mode = kwargs.get('mode', 'forward')
        if 'mode' in kwargs:
            del kwargs['mode']
        return getattr(self, '_' + mode)(*args, **kwargs)

    def beam_search(self, init_state, init_logprobs, *args, **kwargs):
        
        # Function to compute the similarity score to be augmented
        def add_diversity(beam_seq_table, logprobs, t, divm, diversity_lambda, bdash):
            local_time = t - divm
            unaug_logprobs = logprobs.clone()
            batch_size = beam_seq_table[0].shape[0]

            if divm > 0:
                change = logprobs.new_zeros(batch_size, logprobs.shape[-1])
                for prev_choice in range(divm):
                    prev_decisions = beam_seq_table[prev_choice][:, :, local_time]  # Nxb
                    for prev_labels in range(bdash):
                        change.scatter_add_(1, prev_decisions[:, prev_labels].unsqueeze(-1),
                                            change.new_ones(batch_size, 1))

                if local_time == 0:
                    logprobs = logprobs - change * diversity_lambda
                else:
                    logprobs = logprobs - self.repeat_tensor(bdash, change) * diversity_lambda

            return logprobs, unaug_logprobs

        # Function for one step of classical beam search
        def beam_step(logprobs, unaug_logprobs, beam_size, t, beam_seq, beam_seq_logprobs, beam_logprobs_sum, state):
            batch_size = beam_logprobs_sum.shape[0]
            vocab_size = logprobs.shape[-1]
            logprobs = logprobs.reshape(batch_size, -1, vocab_size)  # NxbxV

            if t == 0:
                assert logprobs.shape[1] == 1
                beam_logprobs_sum = beam_logprobs_sum[:, :1]

            candidate_logprobs = beam_logprobs_sum.unsqueeze(-1) + logprobs  # beam_logprobs_sum Nxb logprobs NxbxV
            ys, ix = torch.sort(candidate_logprobs.reshape(candidate_logprobs.shape[0], -1), -1, True)
            ys, ix = ys[:, :beam_size], ix[:, :beam_size]

            beam_ix = ix // vocab_size  # Nxb which beam
            selected_ix = ix % vocab_size  # Nxb # which word
            state_ix = (beam_ix + torch.arange(batch_size).type_as(beam_ix).unsqueeze(-1) * logprobs.shape[1]).reshape(-1)

            if t > 0:
                assert (beam_seq.gather(1, beam_ix.unsqueeze(-1).expand_as(beam_seq)) == beam_seq.reshape(-1, beam_seq.shape[-1])[state_ix].view_as(beam_seq)).all()
                beam_seq = beam_seq.gather(1, beam_ix.unsqueeze(-1).expand_as(beam_seq))

                beam_seq_logprobs = beam_seq_logprobs.gather(1, beam_ix.unsqueeze(-1).unsqueeze(-1).expand_as(beam_seq_logprobs))

            beam_seq = torch.cat([beam_seq, selected_ix.unsqueeze(-1)], -1)  # beam_seq Nxbxl
            beam_logprobs_sum = beam_logprobs_sum.gather(1, beam_ix) + logprobs.reshape(batch_size, -1).gather(1, ix)

            _tmp_beam_logprobs = unaug_logprobs[state_ix].reshape(batch_size, -1, vocab_size)
            beam_logprobs = unaug_logprobs.reshape(batch_size, -1, vocab_size).gather(1,
                                                                                      beam_ix.unsqueeze(-1).expand(-1, -1, vocab_size))  # NxbxV
            assert (_tmp_beam_logprobs == beam_logprobs).all()

            beam_seq_logprobs = torch.cat([beam_seq_logprobs, beam_logprobs.reshape(batch_size, -1, 1, vocab_size)], 2)

            new_state = [None for _ in state]
            for _ix in range(len(new_state)):
                new_state[_ix] = state[_ix][:, state_ix]
            state = new_state

            return beam_seq, beam_seq_logprobs, beam_logprobs_sum, state

        # Start of the diverse beam search
        opt = kwargs['opt']
        temperature = opt.get('temperature', 1)
        beam_size = opt.get('beam_size', 10)
        group_size = opt.get('group_size', 1)
        diversity_lambda = opt.get('diversity_lambda', 0.5)
        decoding_constraint = opt.get('decoding_constraint', 0)
        suppress_UNK = opt.get('suppress_UNK', 0)
        length_penalty = utils.penalty_builder(opt.get('length_penalty', ''))
        bdash = beam_size // group_size  # beam per group

        batch_size = init_logprobs.shape[0]
        device = init_logprobs.device

        # Initializations
        beam_seq_table = [torch.LongTensor(batch_size, bdash, 0).to(device) for _ in range(group_size)]
        beam_seq_logprobs_table = [torch.FloatTensor(batch_size, bdash, 0, self.vocab_size + 1).to(device) for _ in range(group_size)]
        beam_logprobs_sum_table = [torch.zeros(batch_size, bdash).to(device) for _ in range(group_size)]

        done_beams_table = [[[] for __ in range(group_size)] for _ in range(batch_size)]
        state_table = [[_.clone() for _ in init_state] for _ in range(group_size)]
        logprobs_table = [init_logprobs.clone() for _ in range(group_size)]

        args = list(args)
        args = utils.split_tensors(group_size, args)

        for t in range(self.max_seq_length + group_size - 1):
            for divm in range(group_size):
                if t >= divm and t <= self.max_seq_length + divm - 1:
                    logprobs = logprobs_table[divm]

                    # Suppress previous word
                    if decoding_constraint and t - divm > 0:
                        logprobs.scatter_(1, beam_seq_table[divm][:, :, t - divm - 1].reshape(-1, 1).to(device), float('-inf'))

                    # Suppress UNK tokens
                    if suppress_UNK and hasattr(self, 'vocab') and self.vocab[str(logprobs.size(1) - 1)] == 'UNK':
                        logprobs[:, logprobs.size(1) - 1] = logprobs[:, logprobs.size(1) - 1] - 1000
                    
                    # Add diversity
                    logprobs, unaug_logprobs = add_diversity(beam_seq_table, logprobs, t, divm, diversity_lambda, bdash)

                    # Infer new beams
                    beam_seq_table[divm], beam_seq_logprobs_table[divm], beam_logprobs_sum_table[divm], state_table[divm] = beam_step(
                        logprobs, unaug_logprobs, bdash, t - divm, beam_seq_table[divm], beam_seq_logprobs_table[divm],
                        beam_logprobs_sum_table[divm], state_table[divm])

                    for b in range(batch_size):
                        is_end = beam_seq_table[divm][b, :, t - divm] == self.eos_idx
                        if t == self.max_seq_length + divm - 1:
                            is_end.fill_(1)
                        for vix in range(bdash):
                            if is_end[vix]:
                                final_beam = {
                                    'seq': beam_seq_table[divm][b, vix].clone(),
                                    'logps': beam_seq_logprobs_table[divm][b, vix].clone(),
                                    'unaug_p': beam_seq_logprobs_table[divm][b, vix].sum().item(),
                                    'p': beam_logprobs_sum_table[divm][b, vix].item()
                                }
                                final_beam['p'] = length_penalty(t - divm + 1, final_beam['p'])
                                done_beams_table[b][divm].append(final_beam)
                        beam_logprobs_sum_table[divm][b, is_end] -= 1000

                    it = beam_seq_table[divm][:, :, t - divm].reshape(-1)
                    logprobs_table[divm], state_table[divm] = self.get_logprobs_state(it.cuda(), *(args[divm] + [state_table[divm]]))
                    logprobs_table[divm] /= temperature

        final_output = []
        for b in range(batch_size):
            done_beams = []
            for divm in range(group_size):
                done_beams.append(sorted(done_beams_table[b][divm], key=lambda x: x['p'], reverse=True))

            done_beams = done_beams[:group_size]
            done_beams = [x[0] for x in done_beams]
            final_output.append(done_beams)

        return final_output
