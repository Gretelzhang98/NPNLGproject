from collections import defaultdict

import random
import torch
import torch.nn.functional as F
from os.path import join as pjoin
from tqdm import tqdm

from torch.distributions import Categorical

from rsa_utils import logsumexp
import numpy as np
import pickle

from scipy.stats import entropy






class RSA(object):
    """
    RSA through matrix normalization

    We can compute RSA through the following steps:
    Step 1: add image prior: + log P(i) to the row
    Step 2: Column normalize
    - Pragmatic Listener L1: L1(i|c) \propto S0(c|i) P(i)
    Step 3: Multiply the full matrix by rationality parameter (0, infty), when rationality=1, no changes (similar to temperature)
    Step 4: add speaker prior: + log P(c_t|i, c_<t) (basically add the original literal matrix) (very easy)
            OR add a unconditioned speaker prior: + log P(c) (through a language model, like KenLM)
    Step 5: Row normalization
    - Pragmatic Speaker S1: S1(c|i) \propto L1(i|c) p(c), where p(c) can be S0

    The reason for additions is e^{\alpha log L1(i|c) + log p(i)}, where \alpha is rationality parameter
    """

    def __init__(self):
        # can be used to add KenLM language model
        # The "gigaword" one takes too long to load
        pass

    def build_literal_matrix(self, orig_logprob, distractor_logprob):
        """
        :param orig_logprob: [n_sample]
        :param distractor_logprob: [num_distractors, n_sample]
        :return: We put orig_logprob as the FIRST row
                [num_distractors+1 , n_sample]
        """
        return torch.cat([orig_logprob.unsqueeze(0), distractor_logprob], dim=0)

    def compute_pragmatic_speaker(self, literal_matrix,
                                  rationality=1.0, speaker_prior=False, lm_logprobsf=None,
                                  return_diagnostics=False):
        """
        Do the normalization over logprob matrix

        literal_matrix: [num_distractor_images+1, captions]

        :param literal_matrix: should be [I, C]  (num_images, num_captions)
                               Or [I, Vocab] (num_images, vocab_size)
        :param speaker_prior: turn on, we default to adding literal matrix
        :param speaker_prior_lm_mat: [I, Vocab] (a grammar weighting for previous tokens)

        :return:
               A re-weighted matrix [I, C/Vocab]
        """
        # step 1
        pass
        # step 2
        s0 = literal_matrix.clone()
        norm_const = logsumexp(literal_matrix, dim=0, keepdim=True)
        l1 = literal_matrix.clone() - norm_const
        # step 3
        l1 *= rationality
        # step 4
        if speaker_prior:
            # we add speaker prior
            # this needs to be a LM with shared vocabulary
            if lm_logprobsf is not None:
                s1 = l1 + lm_logprobsf[0]
            else:
                s1 = l1 + s0
        # step 5
        norm_const = logsumexp(s1, dim=1, keepdim=True)  # row normalization
        s1 = s1 - norm_const

        if return_diagnostics:
            return s1, l1, s0

        return s1

    def compute_entropy(self, prob_mat, dim, keepdim=True):
        return -torch.sum(prob_mat * torch.exp(prob_mat), dim=dim, keepdim=keepdim)

    def compute_pragmatic_speaker_w_similarity(self, literal_matrix, num_similar_images,
                                               rationality=1.0, speaker_prior=False, lm_logprobsf=None,
                                               entropy_penalty_alpha=0.0, return_diagnostics=False):

        s0_mat = literal_matrix
        prior = s0_mat.clone()[0]

        l1_mat = s0_mat - logsumexp(s0_mat, dim=0, keepdim=True)

        same_cell_prob_mat = l1_mat[:num_similar_images + 1] - logsumexp(l1_mat[:num_similar_images + 1], dim=0)
        l1_qud_mat = same_cell_prob_mat.clone()

        entropy = self.compute_entropy(same_cell_prob_mat, 0, keepdim=True)  # (1, |V|)

        utility_2 = entropy

        utility_1 = logsumexp(l1_mat[:num_similar_images + 1], dim=0, keepdim=True)  # [1, |V|]

        utility = (1 - entropy_penalty_alpha) * utility_1 + entropy_penalty_alpha * utility_2

        s1 = utility * rationality

        # apply rationality
        if speaker_prior:
            if lm_logprobsf is None:
                s1 += prior
            else:
                s1 += lm_logprobsf[0]  # lm rows are all the same  # here is two rows summation

        if return_diagnostics:
            # We return RSA-terms only; on the oustide (Debugger), we re-assemble for snapshots of computational process
            # s0, L1, u1, L1*, u2, u1+u2, s1
            # mat, vec, vec, mat, vec, vec, vec
            return s0_mat, l1_mat, utility_1, l1_qud_mat, entropy, utility_2, utility, s1 - logsumexp(s1, dim=1,
                                                                                                      keepdim=True)

        return s1 - logsumexp(s1, dim=1, keepdim=True)


class IncRSA(RSA):
    def __init__(self, model, rsa_dataset, lm_model=None):
        super().__init__()
        self.model = model
        self.rsa_dataset = rsa_dataset

        args = self.rsa_dataset.args

        trainer_creator = getattr(model, args.model)
        evaluator = trainer_creator(args, model, rsa_dataset.split_to_data['val'], [],
                                    None, rsa_dataset.device)
        evaluator.train = False

        self.evaluator = evaluator
        self.device = self.evaluator.device

    def sentence_decode(self, sampled_ids):
        outputs = sampled_ids
        vocab = self.evaluator.dataset.vocab

        generated_captions = []
        for out_idx in range(len(outputs)):
            sentence = []
            for w in outputs[out_idx]:
                word = vocab.get_word_from_idx(w.data.item())
                if word != vocab.end_token:
                    sentence.append(word)
                else:
                    break
            generated_captions.append(' '.join(sentence))

        return generated_captions

    def semantic_speaker(self, image_id_list, decode_strategy="greedy"):
        # image_id here is a string!
        image_input, labels = self.rsa_dataset.get_batch(image_id_list)
        if decode_strategy == 'greedy':
            image_input = image_input.to(self.device)
            sample_ids = self.model.generate_sentence(image_input, self.evaluator.start_word,
                                                      self.evaluator.end_word, labels, labels_onehot=None,
                                                      max_sampling_length=50, sample=False)
        else:
            raise Exception("not implemented")

        if len(sample_ids.shape) == 1:
            sample_ids = sample_ids.unsqueeze(0)

        return self.sentence_decode(sample_ids)

    def greedy_pragmatic_speaker(self, img_id, question_id, rationality,
                                 speaker_prior, entropy_penalty_alpha,
                                 max_cap_per_cell=5, cell_select_strategy=None,
                                 no_similar=False, verbose=True, return_diagnostic=False, segment=False,
                                 subset_similarity=False):
        # collect_stats: debug mode (or eval mode); collect RSA statistics to understand internal workings

        if max_cap_per_cell == 0:
            return self.semantic_speaker([img_id], decode_strategy="greedy")

        dis_cell, sim_cell, quality = self.rsa_dataset.get_cells_by_partition(img_id, question_id, max_cap_per_cell,
                                                                              cell_select_strategy,
                                                                              no_similar=no_similar,
                                                                              segment=segment,
                                                                              subset_similarity=subset_similarity)

        image_id_list = [img_id] + sim_cell + dis_cell
        with torch.no_grad():
            if not return_diagnostic:
                captions = self.greedy_pragmatic_speaker_free(image_id_list, len(sim_cell),
                                                              rationality, speaker_prior, entropy_penalty_alpha)
            else:
                captions, diags = self.greedy_pragmatic_speaker_free(image_id_list, len(sim_cell),
                                                                     rationality, speaker_prior, entropy_penalty_alpha,
                                                                     return_diagnostic=True)

        if return_diagnostic:
            return captions[0], quality, diags

        return captions[0], quality

    def fill_list(self, items, lists):
        # this is a pass-by-reference update
        for item, ls in zip(items, lists):
            ls.append(item)

    def greedy_pragmatic_speaker_free(self, image_id_list, num_sim, rationality,
                                      speaker_prior, entropy_penalty_alpha, lm_logprobsf=None,
                                      max_sampling_length=50, sample=False, return_diagnostic=False):
        """
        We always assume image_id_list[0] is the target image
        image_id_list[:num_sim] are the within cell
        image_id_list[num_sim:] are the distractors

        Will only return 1 caption for the target image
        :param image_id_list:
        :param num_sim:
        :param num_distractor:
        :param max_sampling_length:
        :return:
        """

        image_input, labels = self.rsa_dataset.get_batch(image_id_list)
        image_inputs = image_input.to(self.device)

        start_word = self.evaluator.start_word
        end_word = self.evaluator.end_word

        feat_func = self.model.get_labels_append_func(labels, None)
        image_features = image_inputs

        image_features = self.model.linear1(image_features)
        image_features = F.relu(image_features)
        image_features = feat_func(image_features)
        image_features = image_features.unsqueeze(1)  # (11, 1, 1200)

        embedded_word = self.model.word_embed(start_word)
        embedded_word = embedded_word.expand(image_features.size(0), -1, -1)

        init_states = (None, None)
        lstm1_states, lstm2_states = init_states

        end_word = end_word.squeeze().expand(image_features.size(0))
        reached_end = torch.zeros_like(end_word.data).byte()

        sampled_ids = []

        if return_diagnostic:
            # their length is the time step length
            s0_list, l1_list, u1_list, l1_qud_list, entropy_list, u2_list, u_list, s1_list = [], [], [], [], [], [], [], []

        # greedy loop, over time step
        i = 0
        while not reached_end.all() and i < max_sampling_length:
            lstm1_input = embedded_word

            # LSTM 1
            lstm1_output, lstm1_states = self.model.lstm1(lstm1_input, lstm1_states)

            lstm1_output = torch.cat((image_features, lstm1_output), 2)

            # LSTM 2
            lstm2_output, lstm2_states = self.model.lstm2(lstm1_output, lstm2_states)

            outputs = self.model.linear2(lstm2_output.squeeze(1))
            # outputs: torch.Size([11, 3012])

            # all our RSA computation is in log-prob space
            log_probs = F.log_softmax(outputs, dim=-1)  # log(softmax(x))

            # rsa
            literal_matrix = log_probs

            # diagnostics
            if not return_diagnostic:
                pragmatic_array = self.compute_pragmatic_speaker_w_similarity(literal_matrix, num_sim,
                                                                              rationality=rationality,
                                                                              speaker_prior=speaker_prior,
                                                                              entropy_penalty_alpha=entropy_penalty_alpha,
                                                                              lm_logprobsf=lm_logprobsf)
            else:
                s0_mat, l1_mat, utility_1, l1_qud_mat, entropy, utility_2, combined_u, pragmatic_array = self.compute_pragmatic_speaker_w_similarity(
                    literal_matrix, num_sim,
                    rationality=rationality,
                    speaker_prior=speaker_prior,
                    entropy_penalty_alpha=entropy_penalty_alpha,
                    lm_logprobsf=lm_logprobsf,
                    return_diagnostics=True)
                self.fill_list([s0_mat, l1_mat, utility_1, l1_qud_mat, entropy, utility_2, combined_u, pragmatic_array],
                               [s0_list, l1_list, u1_list, l1_qud_list, entropy_list, u2_list, u_list, s1_list])

            # pragmatic_array:
            # torch.Size([1, 3012])

            # pragmatic array becomes the computational output
            # but we need to repeat it for all
            # beam search / diverse beam search this part is easier to handle.
            outputs = pragmatic_array.expand(len(image_id_list), -1)  # expand along batch dimension
            # rsa augmentation ends

            if sample:
                predicted, log_p = self.sample(outputs)
                active_batches = (~reached_end)
                log_p *= active_batches.float().to(log_p.device)
                # log_probabilities.append(log_p.unsqueeze(1))
                # lengths += active_batches.long()
            else:
                predicted = outputs.max(1)[1]

            reached_end = reached_end | predicted.eq(end_word).data
            sampled_ids.append(predicted.unsqueeze(1))
            embedded_word = self.model.word_embed(predicted)
            embedded_word = embedded_word.unsqueeze(1)

            i += 1

        sampled_ids = torch.cat(sampled_ids, 1).squeeze()

        if return_diagnostic:
            return self.sentence_decode(sampled_ids), [s0_list, l1_list, u1_list, l1_qud_list, entropy_list, u2_list,
                                                       u_list, s1_list]

        return self.sentence_decode(sampled_ids)

    def sample(self, logits):
        dist = Categorical(logits=logits)
        sample = dist.sample()
        return sample, dist.log_prob(sample)

    def diverse_beam_search(self):
        pass

    def nucleus_sampling(self):
        pass