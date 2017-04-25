import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init
import torch.utils.data
from torch.autograd import Variable


class DependencyEncoder(nn.Module):
    """A dependency parser variation of the stack encoder in the SPINN
    architecture.
    """

    def __init__(self, encoder_size, tracking_lstm=True, tracking_lstm_dim=64):
        super(DependencyEncoder, self).__init__()
        self.encoder_size = encoder_size
        self.tlstm_dim = tracking_lstm_dim
        if tracking_lstm:
            self.tracking_lstm = nn.LSTMCell(3 * encoder_size,
                                             tracking_lstm_dim)
        else:
            self.tracking_lstm = None
        self.composition = DependencyTreeLSTMCell(tracking_lstm_dim,
                                                  self.encoder_size)

    def _batch_states(self, states):
        hidden, cell = zip(*states)
        return torch.cat(hidden), torch.cat(cell)

    def _compose(self, hc_head, hc_child, tracking, direction):
        head = self._batch_states(hc_head)
        child = self._batch_states(hc_child)
        if tracking:
            tracking = torch.stack(tracking)
        else:
            tracking = None
        red_h, red_c = self.composition(tracking, head, child, direction)
        return red_h, red_c

    def forward(self, sequence, transitions):
        """Encode sentence by recursively computing representations of
        head-child pairs in a dependency tree.

        Args:
            sequence (autograd.Variable): An autograd.Variable containing the
            sentences to encode of size (B x L x D), where B is batch size, L
            is the length of the sentences and D is the dimensionality of the
            data.

            transitions (torch.Tensor): A torch.Tensor containing the
            transitions that lead to the given dependency tree for each
            sentence. Size (B x T) where B is the batch size and T is the
            number of transitions.
        """
        batch_size = transitions.size(0)
        timesteps = transitions.size(1)
        tokens_h, tokens_c = sequence.chunk(2, 2)
        buffers_h = [list(torch.split(x.squeeze(), 1, 0))
                     for x in tokens_h.split(1, 0)]
        buffers_c = [list(torch.split(x.squeeze(), 1, 0))
                     for x in tokens_c.split(1, 0)]
        stacks = [[(bh[0], bc[0]), (bh[0], bc[0])] for bh, bc in
                  zip(buffers_h, buffers_c)]
        if self.tracking_lstm:
            tlstm_hidden = Variable(torch.randn(batch_size, self.tlstm_dim))
            tlstm_cell = Variable(torch.randn(batch_size, self.tlstm_dim))

        # if we transpose the transitions matrix we can index it like
        # transitions[timestep] to get all the data for the timestep
        transitions = transitions.t()
        for timestep in range(timesteps):
            mask = transitions[timestep].int()
            if self.tracking_lstm:
                h_stack1 = torch.cat([stack[-1][0] for stack in stacks])
                h_stack2 = torch.cat([stack[-2][0] for stack in stacks])
                buffer_top = torch.cat([bh[0] for bh in buffers_h])
                tlstm_in = torch.cat((buffer_top, h_stack1, h_stack2), 1)
                tlstm_hidden, tlstm_cell = self.tracking_lstm(tlstm_in,
                                                              (tlstm_hidden,
                                                               tlstm_cell))

            left_head, left_child, left_tracking = [], [], []
            right_head, right_child, right_tracking = [], [], []
            tstep_data = zip(mask, stacks, buffers_h, buffers_c)

            for i, (trans, stack, buf_h, buf_c) in enumerate(tstep_data):
                if trans == 1:  # SHIFT
                    stack.append((buf_h.pop(0), buf_c.pop(0)))
                elif trans == 2:  # LEFT-ARC
                    left_head.append(stack.pop())
                    left_child.append(stack.pop())
                    if self.tracking_lstm:
                        left_tracking.append(tlstm_hidden[i])
                elif trans == 3:  # RIGHT-ARC
                    right_child.append(stack.pop())
                    right_head.append(stack.pop())
                    if self.tracking_lstm:
                        right_tracking.append(tlstm_hidden[i])

            if left_head:
                red_h, red_c = self._compose(left_head, left_child,
                                             left_tracking, 'left')
                left_reduced_h = iter(red_h)
                left_reduced_c = iter(red_c)

            if right_head:
                red_h, red_c = self._compose(right_head, right_child,
                                             right_tracking, 'right')
                right_reduced_h = iter(red_h)
                right_reduced_c = iter(red_c)

            if left_head or right_head:
                for trans, stack in zip(mask, stacks):
                    if trans == 2:  # IF WE LEFT-REDUCE
                        stack.append((next(left_reduced_h).unsqueeze(0),
                                      next(left_reduced_c).unsqueeze(0)))
                    elif trans == 3:  # IF WE RIGHT-REDUCE
                        stack.append((next(right_reduced_h).unsqueeze(0),
                                      next(right_reduced_c).unsqueeze(0)))

        out = torch.cat([example[-1][0] for example in stacks], 0)
        return out


class StackEncoder(nn.Module):
    """Implementation of the SPINN stack-based encoder.
    """

    def __init__(self, encoder_size, tracking_lstm=False, tracking_lstm_dim=64):
        super(StackEncoder, self).__init__()
        self.encoder_size = encoder_size
        self.tlstm_dim = tracking_lstm_dim
        if tracking_lstm:
            self.tracking_lstm = nn.LSTMCell(3 * encoder_size,
                                             tracking_lstm_dim)
        else:
            self.tracking_lstm = None
        self.composition = TreeLSTMCell(self.tlstm_dim, self.encoder_size)

    def _batch_states(self, states):
        hidden, cell = zip(*states)
        return torch.cat(hidden), torch.cat(cell)

    def forward(self, sequence, transitions):
        """Encode the sentences by recursively combining left and right
        children into new nodes in a binary constituency tree.

        Args:
            sequence (torch.Tensor): Tensor containing the sentences to encode,
            of size (B x L x D), where B is batch size, L is length of the
            sequences and D is the dimensionality of the data.

            transitions (torch.Tensor): Tensor containing the transitions for
            each sequence of size (B x T), where B is batch size and T is the
            number of transitions.

        Returns:
            torch.Tensor: Tensor of size (B x D) containing the final hidden
            states of the given sequences.
        """
        batch_size = transitions.size(0)
        timesteps = transitions.size(1)
        tokens_h, tokens_c = sequence.chunk(2, 2)
        buffers_h = [list(torch.split(x.squeeze(), 1, 0))
                     for x in tokens_h.split(1, 0)]
        buffers_c = [list(torch.split(x.squeeze(), 1, 0))
                     for x in tokens_c.split(1, 0)]
        stacks = [[(bh[0], bc[0]), (bh[0], bc[0])] for bh, bc in
                  zip(buffers_h, buffers_c)]
        if self.tracking_lstm:
            tlstm_hidden = Variable(torch.randn(batch_size, self.tlstm_dim))
            tlstm_cell = Variable(torch.randn(batch_size, self.tlstm_dim))

        # if we transpose the transitions matrix we can index it like
        # transitions[timestep] to get all the data for the timestep
        transitions = transitions.t()
        for timestep in range(timesteps):
            mask = transitions[timestep].int()
            if self.tracking_lstm:
                h_stack1 = torch.cat([stack[-1][0] for stack in stacks])
                h_stack2 = torch.cat([stack[-2][0] for stack in stacks])
                buffer_top = torch.cat([bh[0] for bh in buffers_h])
                tlstm_in = torch.cat((buffer_top, h_stack1, h_stack2), 1)
                tlstm_hidden, tlstm_cell = self.tracking_lstm(tlstm_in,
                                                              (tlstm_hidden,
                                                               tlstm_cell))

            right, left, tracking = [], [], []
            tstep_data = zip(mask, stacks, buffers_h, buffers_c)

            for i, (transition, stack, buf_h, buf_c) in enumerate(tstep_data):
                if transition == 1:  # SHIFT
                    stack.append((buf_h.pop(0), buf_c.pop(0)))
                elif transition == 2:  # REDUCE
                    right.append(stack.pop())
                    left.append(stack.pop())
                    if self.tracking_lstm:
                        tracking.append(tlstm_hidden[i])

            if right:
                hc_right = self._batch_states(right)
                hc_left = self._batch_states(left)
                if self.tracking_lstm:
                    tracking = torch.stack(tracking)
                else:
                    tracking = None
                reduced_h, reduced_c = self.composition(tracking, hc_right,
                                                        hc_left)
                reduced_h = iter(reduced_h)
                reduced_c = iter(reduced_c)

                for trans, stack in zip(mask, stacks):
                    if trans == 2:  # IF WE REDUCE
                        stack.append((next(reduced_h).unsqueeze(0),
                                      next(reduced_c).unsqueeze(0)))

        out = torch.cat([example[-1][0] for example in stacks], 0)
        return out


class BOWEncoder(nn.Module):
    """A simple bag-of-words encoder that performs composition by summing the
    vectors in the given sentence.
    """
    def __init__(self, embeddings_size):
        super(BOWEncoder, self).__init__()
        self.encoder_size = embeddings_size

    def forward(self, sequence, transitions):
        return sequence.sum(1).squeeze(1)


class SPINNetwork(nn.Module):
    """The complete SPINN module that takes as input sequences of words and
    (optionally) sequences of transitions (if the encoder that is used is
    either the StackEncoder or the DependencyEncoder).

    Args:
        embedding_dim (int): Desired dimensionality of the embeddings.
        vocab_size (int): The size of the vocabulary.
        encoder (nn.Module): A nn.Module that takes as input a sequence
        and transitions (which can be None) and returns a vector representation
        of said sequence.
    """

    def __init__(self, embedding_dim, vocab_size, encoder):
        super(SPINNetwork, self).__init__()
        self.word_embedding = nn.Embedding(vocab_size, embedding_dim,
                                           padding_idx=0)
        self.embedding_dim = embedding_dim
        self.encoder_dim = encoder.encoder_size
        if type(encoder) == BOWEncoder:
            self.projection_dim = self.encoder_dim
        else:
            self.projection_dim = self.encoder_dim * 2
        self.projection = nn.Linear(self.embedding_dim,
                                    self.projection_dim)
        self.batch_norm = nn.BatchNorm1d(self.projection_dim)
        self.encoder = encoder
        self.classifier = MLPClassifier(self.encoder_dim * 4, 1024)
        torch.nn.init.kaiming_normal(self.projection.weight)

    def forward(self, prem_sequence, hypo_sequence, prem_transitions,
                hypo_transitions):
        """Perform classification of the sentence pair. Encods the sentences
        into vector representations and use these as input to a multi-layer
        perceptron that performs the classification.

        Args:
            prem_sequence (autograd.Variable): An autograd.Variable, of size
            (B x L) where B is batch size and L is the length of the sequence,
            containing the premise.

            hypo_sequence (autograd.Variable): An autograd.Variable, of size
            (B x L) where B is batch size and L is the length of the sequence,
            containg the hypothesis.

            prem_transitions (torch.Tensor): A tensor of size (B x T) where B
            is the batch size and T is the number of transitions containing the
            transitions for the premise sequence, or None if not to be used.

            hypo_transitions (torch.Tensor): A tensor of size (B x T) where B
            is the batch size and T is the number of transitions containing the
            transitions for the hypothesis sequence, or None if not to be used.
        """
        seq_len = prem_sequence.size(1)
        prem_emb = self.word_embedding(prem_sequence)
        hypo_emb = self.word_embedding(hypo_sequence)
        # Repackage in new Variable to avoid updates to the embedding layer
        prem_emb = Variable(prem_emb.data)
        hypo_emb = Variable(hypo_emb.data)
        prem_proj = self.projection(prem_emb.view(-1, self.embedding_dim))
        hypo_proj = self.projection(hypo_emb.view(-1, self.embedding_dim))
        prem_bnorm = self.batch_norm(prem_proj)
        hypo_bnorm = self.batch_norm(hypo_proj)
        prem = prem_bnorm.view(-1, seq_len, self.projection_dim)
        hypo = hypo_bnorm.view(-1, seq_len, self.projection_dim)
        prem_encoded = self.encoder(prem, prem_transitions)
        hypo_encoded = self.encoder(hypo, hypo_transitions)
        x_classifier = torch.cat((hypo_encoded, prem_encoded,
                                  hypo_encoded - prem_encoded,
                                  hypo_encoded * prem_encoded), 1)
        return self.classifier(x_classifier)


class TreeLSTMCell(nn.Module):

    """A binary tree LSTM cell, as defined by Tai et al. (2015).

    Args:
        input_size (int): Size of the input to the LSTM.
        hidden_size (int): Size of the hidden state.
    """

    def __init__(self, input_size, hidden_size):
        super(TreeLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.W = nn.Linear(input_size, hidden_size * 5, bias=False)
        self.U_r = nn.Linear(hidden_size, hidden_size * 5, bias=False)
        self.U_l = nn.Linear(hidden_size, hidden_size * 5)
        self.init_parameters()

    def init_parameters(self):
        for weight in self.parameters():
            if weight.dim() > 1:
                torch.nn.init.kaiming_normal(weight.data)

    def forward(self, x, hc_right, hc_left):
        """Perform forward propagation.

        Args:
            x (torch.Tensor): Input to the LSTM cell. Needs to be provided
            but can be given as None.
            hc_right (tuple): A tuple containing the hidden state and memory
            cell of the right child of this new node.
            hc_left (tuple): A tuple containing the hidden state and memory
            cell of the left child of this new node.
        """
        # Inputs = (B X D)
        h_right, c_right = hc_right
        h_left, c_left = hc_left
        gates = self.U_l(h_left) + self.U_r(h_right)
        if x is not None:
            gates += self.W(x)

        i_j, o_j, f_jl, f_jr, u_j = gates.chunk(5, 1)

        c_j = i_j.sigmoid() * u_j.tanh() + (f_jl.sigmoid() * c_left +
                                            f_jr.sigmoid() * c_right)
        h_j = o_j.sigmoid() * c_j.tanh()
        return h_j, c_j


class DependencyTreeLSTMCell(nn.Module):
    """A dependency tree variation of the LSTM.

    Args:
        input_size (int): Size of the input to the LSTM.
        hidden_size (int): Size of the hidden state.
    """

    def __init__(self, input_size, hidden_size):
        super(DependencyTreeLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.W = nn.Linear(input_size, hidden_size * 5, bias=False)
        self.U_head = nn.Linear(hidden_size, hidden_size * 5)
        self.U_child_left = nn.Linear(hidden_size, hidden_size * 5, bias=False)
        self.U_child_right = nn.Linear(hidden_size, hidden_size * 5, bias=False)
        self.init_parameters()

    def init_parameters(self):
        for weight in self.parameters():
            if weight.dim() > 1:
                torch.nn.init.kaiming_normal(weight.data)

    def forward(self, x, hc_head, hc_child, direction):
        """Perform forward propagation.

        Args:
            x (torch.Tensor): Input to the LSTM cell. Needs to be provided
            but can be given as None.
            hc_head (tuple): A tuple containing the hidden state and memory
            cell of the head.
            hc_child (tuple): A tuple containing the hidden state and memory
            cell of the child.
            direction (str): A string in {'left', 'right'} indicating the
            direction of attachment for these pairs.
        """
        # Inputs = (B X D)
        assert direction in ['left', 'right'], "Illegal attachment direction."
        h_head, c_head = hc_head
        h_child, c_child = hc_child
        if direction == 'left':
            gates = self.U_head(h_head) + self.U_child_left(h_child)
        elif direction == 'right':
            gates = self.U_head(h_head) + self.U_child_right(h_child)
        if x is not None:
            gates += self.W(x)

        i_j, o_j, f_jh, f_jc, u_j = gates.chunk(5, 1)

        c_j = i_j.sigmoid() * u_j.tanh() + (f_jh.sigmoid() * c_head +
                                            f_jc.sigmoid() * c_child)
        h_j = o_j.sigmoid() * c_j.tanh()
        return h_j, c_j


class MLPClassifier(nn.Module):

    """An ordinary feed-forward multi-layer perceptron consisting of two fully
    connected layers, of which the first has a ReLU non-linearity, and the
    output of the last layer has a softmax classifier that performs three way
    classification. Applies batch normalization to the input as well as the
    output of the hidden layer.

    Args:
        input_size (int): Size of the input to the network.
        hidden_size (int): Size of the hidden layer of the network.
    """

    def __init__(self, input_size, hidden_size):
        super(MLPClassifier, self).__init__()
        self.batch_norm_in = nn.BatchNorm1d(input_size)
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.batch_norm_out = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, 3)
        self._init_parameters()

    def _init_parameters(self):
        torch.nn.init.kaiming_normal(self.fc1.weight)
        torch.nn.init.uniform(self.fc2.weight, -0.005, 0.005)

    def forward(self, x):
        x = self.batch_norm_in(x)
        x = F.dropout(x, p=0.2)
        x = F.relu(self.fc1(x))
        x = self.batch_norm_out(x)
        x = F.dropout(x, p=0.2)
        x = self.fc2(x)
        return F.log_softmax(x)


def test(model, data):
    model.eval()
    test_loss = 0
    correct = 0
    for batch, (premise, hypothesis, premise_transitions,
                hypothesis_transitions, target) in enumerate(data):
        prem = Variable(premise, volatile=True)
        hypo = Variable(hypothesis, volatile=True)
        target = Variable(target.squeeze())
        output = model(prem, hypo,
                       premise_transitions,
                       hypothesis_transitions)
        test_loss += F.nll_loss(output, target).data[0]
        _, pred = output.data.max(1)
        correct += pred.eq(target.data).sum()

    average_test_loss = test_loss / len(data)
    return average_test_loss, correct
