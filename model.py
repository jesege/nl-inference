import argparse
import time
import random
import json
import math
import utils
import torch
import torch.nn as nn
import torch.nn.init
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable


class DependencyEncoder(nn.Module):
    def __init__(self, hidden_size, tracking_lstm=True, tracking_lstm_dim=64):
        super(DependencyEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.tlstm_dim = tracking_lstm_dim
        if tracking_lstm_dim:
            self.tracking_lstm = nn.LSTMCell(3 * hidden_size,
                                             tracking_lstm_dim)
        self.encoder = BinaryTreeLSTMCell(50, self.hidden_size)

    def forward(self, tokens_h, tokens_c, transitions):
        batch_size = transitions.size(0)
        timesteps = transitions.size(1)
        stack = [[(torch.zeros(self.hidden_size), torch.zeros(self.hidden_size))
                 for t in range(timesteps + 1)] for b in range(batch_size)]
        buffer_h = tokens_h
        buffer_c = tokens_c
        backpointerq = [[] for _ in range(batch_size)]
        buffer_ptr = torch.IntTensor(batch_size).zero_()

        if self.tracking_lstm:
            tlstm_hidden = Variable(torch.randn(batch_size, self.tlstm_dim))
            tlstm_cell = Variable(torch.randn(batch_size, self.tlstm_dim))
        else:
            # dummy X input to the tree lstm
            tlstm_hidden = Variable(torch.zeros(batch_size, self.x_size))

        transitions = transitions.t()

        for timestep in range(1, timesteps + 1):
            transitions_timestep = transitions[timestep - 1]
            mask = torch.ge(transitions_timestep, 1)

            stk_ptr_1 = torch.LongTensor([utils.get_queue_item(
                backpointerq[i], -1) for i in range(len(backpointerq))])
            stk_ptr_2 = torch.LongTensor([utils.get_queue_item(
                backpointerq[i], -2) for i in range(len(backpointerq))])

            h_buffer_top = Variable(torch.Tensor(batch_size, self.hidden_size))
            c_buffer_top = Variable(torch.Tensor(batch_size, self.hidden_size))
            h_stack1 = Variable(torch.Tensor(batch_size, self.hidden_size))
            c_stack1 = Variable(torch.Tensor(batch_size, self.hidden_size))
            h_stack2 = Variable(torch.Tensor(batch_size, self.hidden_size))
            c_stack2 = Variable(torch.Tensor(batch_size, self.hidden_size))

            for i, sp1, sp2, bp in zip(range(batch_size), stk_ptr_1,
                                       stk_ptr_2, buffer_ptr):
                h_stack1[i], c_stack1[i] = stack[i][sp1]
                h_stack2[i], c_stack2[i] = stack[i][sp2]
                h_buffer_top[i] = buffer_h[i, bp]
                c_buffer_top[i] = buffer_c[i, bp]

            if self.tracking_lstm:
                tlstm_in = torch.cat((h_buffer_top, h_stack1, h_stack2), 1)
                tlstm_hidden, tlstm_cell = self.tracking_lstm(tlstm_in,
                                                              (tlstm_hidden,
                                                               tlstm_cell))

            h_left_arc, c_left_arc = self.encoder(tlstm_hidden,
                                                  h_stack1, h_stack2,
                                                  c_stack1, c_stack2)
            h_right_arc, c_right_arc = self.encoder(tlstm_hidden,
                                                    h_stack2, h_stack1,
                                                    c_stack2, c_stack2)
            for i in range(batch_size):
                if transitions_timestep[i] == 0:
                    stack[i][timestep] = (h_buffer_top[i], c_buffer_top[i])
                elif transitions_timestep[i] == 1:
                    stack[i][timestep] = (h_left_arc[i], c_left_arc[i])
                elif transitions_timestep[i] == 2:
                    stack[i][timestep] = (h_right_arc[i], c_right_arc[i])

            # Move the buffer and stack pointers
            buffer_ptr = buffer_ptr + (1 - mask)
            for i in range(len(backpointerq)):
                if mask[i] == 1:
                    backpointerq[i] = backpointerq[i][:-2]
                backpointerq[i].append(timestep)

        out = torch.stack([example[-1][0] for example in stack], 0)
        return out


class StackEncoder(nn.Module):
    """Implementation of the SPINN stack-based encoder.
    """
    def __init__(self, hidden_size, tracking_lstm=False, tracking_lstm_dim=64):
        super(StackEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.tlstm_dim = tracking_lstm_dim
        if tracking_lstm:
            self.tracking_lstm = nn.LSTMCell(3 * hidden_size,
                                             tracking_lstm_dim)
        self.encoder = BinaryTreeLSTMCell(self.tlstm_dim, self.hidden_size)

    def forward(self, tokens_h, tokens_c, transitions):
        """Encode sentence in the sequence given the transitions.

        Args:
            tokens_h (torch.Tensor): Tensor containing sentences to encode, of
            size (B x L x D), where B is batch size, L is length of the
            sequences and D is the dimensionality of the data.

            tokens_c (torch.Tensor): Tensor containing sentences to encode, of
            size (B x L x D), where B is batch size, L is length of the
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
        stack = [[(torch.zeros(self.hidden_size), torch.zeros(self.hidden_size))
                 for t in range(timesteps + 1)] for b in range(batch_size)]
        buffer_h = tokens_h
        buffer_c = tokens_c
        backpointerq = [[] for _ in range(batch_size)]
        buffer_ptr = torch.IntTensor(batch_size).zero_()
        if self.tracking_lstm:
            tlstm_hidden = Variable(torch.randn(batch_size, self.tlstm_dim))
            tlstm_cell = Variable(torch.randn(batch_size, self.tlstm_dim))
        else:
            # dummy X input to the tree lstm
            tlstm_hidden = Variable(torch.zeros(batch_size, self.tlstm_dim))

        # if we transpose the transitions matrix we can index it like
        # transitions[timestep] to get all the data for the timestep
        transitions = transitions.t()
        for timestep in range(1, timesteps + 1):
            mask = transitions[timestep - 1].int()
            stk_ptr_1 = torch.LongTensor([utils.get_queue_item(
                    backpointerq[i], -1) for i in range(len(backpointerq))])
            stk_ptr_2 = torch.LongTensor([utils.get_queue_item(
                    backpointerq[i], -2) for i in range(len(backpointerq))])

            h_buffer_top = Variable(torch.Tensor(batch_size, self.hidden_size))
            c_buffer_top = Variable(torch.Tensor(batch_size, self.hidden_size))
            h_right = Variable(torch.Tensor(batch_size, self.hidden_size))
            c_right = Variable(torch.Tensor(batch_size, self.hidden_size))
            h_left = Variable(torch.Tensor(batch_size, self.hidden_size))
            c_left = Variable(torch.Tensor(batch_size, self.hidden_size))

            for i, sp1, sp2, bp in zip(range(batch_size), stk_ptr_1,
                                       stk_ptr_2, buffer_ptr):
                h_right[i], c_right[i] = stack[i][sp1]
                h_left[i], c_left[i] = stack[i][sp2]
                h_buffer_top[i] = buffer_h[i, bp]
                c_buffer_top[i] = buffer_c[i, bp]

            if self.tracking_lstm:
                tlstm_in = torch.cat((h_buffer_top, h_right, h_left), 1)
                tlstm_hidden, tlstm_cell = self.tracking_lstm(tlstm_in,
                                                              (tlstm_hidden,
                                                               tlstm_cell))
            h_stack_top, c_stack_top = self.encoder(tlstm_hidden, h_right,
                                                    c_right, h_left, c_left)
            # Update the stack with new values.
            for i in range(batch_size):
                if mask[i] == 1:
                    stack[i][timestep] = (h_stack_top[i], c_stack_top[i])
                else:
                    stack[i][timestep] = (h_buffer_top[i], c_buffer_top[i])

            # Move the buffer and stack pointers
            buffer_ptr = buffer_ptr + (1 - mask)
            for i in range(len(backpointerq)):
                if mask[i] == 1:
                    backpointerq[i] = backpointerq[i][:-2]
                backpointerq[i].append(timestep)

        # return the hidden states from the last timestep
        out = torch.stack([example[-1][0] for example in stack], 0)
        return out


class SPINNetwork(nn.Module):
    def __init__(self, embeddings, encoder_dim, tracking=False):
        super(SPINNetwork, self).__init__()
        self.wemb = nn.Embedding(embeddings.size(0), embeddings.size(1),
                                 padding_idx=0)
        self.wemb.weight = nn.Parameter(embeddings)
        # self.wemb.weight.requires_grad = False
        self.projection = nn.Linear(embeddings.size(1),
                                    encoder_dim * 2)
        self.encoder = StackEncoder(encoder_dim, tracking_lstm=tracking)
        self.encoder_dim = encoder_dim
        self.classifier = MLPClassifier(encoder_dim * 4, 1024)
        torch.nn.init.kaiming_normal(self.projection.weight)

    def forward(self, premise_sequence, hypothesis_sequence,
                premise_transitions, hypotheis_transitions):
        prem_emb = self.wemb(premise_sequence)
        hypo_emb = self.wemb(hypothesis_sequence)
        h_prem, c_prem = torch.stack([self.projection(prem_emb[:, i, :])
                                      for i in range(prem_emb.size(1))],
                                     dim=1).split(self.encoder_dim, dim=2)
        h_hypo, c_hypo = torch.stack([self.projection(hypo_emb[:, i, :])
                                     for i in range(hypo_emb.size(1))],
                                     dim=1).split(self.encoder_dim, dim=2)
        premise_encoded = self.encoder(h_prem, c_prem, premise_transitions)
        hypothesis_encoded = self.encoder(h_hypo, c_hypo, hypotheis_transitions)
        x_classifier = torch.cat((hypothesis_encoded, premise_encoded,
                                  hypothesis_encoded - premise_encoded,
                                  hypothesis_encoded * premise_encoded), 1)
        return self.classifier(x_classifier)


class BinaryTreeLSTMCell(nn.Module):
    """ An Binary Tree-Long-Short Term Memory cell as defined by
    Tai et al. (2015).
    """
    def __init__(self, input_size, hidden_size):
        super(BinaryTreeLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.W_i = nn.Linear(input_size, hidden_size, bias=False)
        self.W_f = nn.Linear(input_size, hidden_size, bias=False)
        self.W_o = nn.Linear(input_size, hidden_size, bias=False)
        self.W_u = nn.Linear(input_size, hidden_size, bias=False)
        self.U_il = nn.Linear(hidden_size, hidden_size, bias=False)
        self.U_ir = nn.Linear(hidden_size, hidden_size, bias=False)
        self.U_fl = nn.Linear(hidden_size, hidden_size, bias=False)
        self.U_fr = nn.Linear(hidden_size, hidden_size, bias=False)
        self.U_ol = nn.Linear(hidden_size, hidden_size, bias=False)
        self.U_or = nn.Linear(hidden_size, hidden_size, bias=False)
        self.U_ur = nn.Linear(hidden_size, hidden_size, bias=False)
        self.U_ul = nn.Linear(hidden_size, hidden_size, bias=False)
        self.b_i = nn.Parameter(torch.Tensor(1, hidden_size))
        self.b_f = nn.Parameter(torch.Tensor(1, hidden_size))
        self.b_o = nn.Parameter(torch.Tensor(1, hidden_size))
        self.b_u = nn.Parameter(torch.Tensor(1, hidden_size))
        self.init_parameters()

    def init_parameters(self):
        for weight in self.parameters():
            torch.nn.init.kaiming_normal(weight.data)

    def forward(self, x, h_right, c_right, h_left, c_left):
        # Inputs = (B X D)
        # The inputting is a bit unituitive and tailor made to the SPINN
        # encoder. Might be bad. But it works and we are probably not gonna
        # use it for anything else.
        i_j = torch.sigmoid(self.W_i(x) + torch.sum(torch.stack(
            (self.U_il(h_left), self.U_ir(h_right))), 0).squeeze() +
             self.b_i.expand(x.size(0), self.hidden_size))
        o_j = torch.sigmoid(self.W_o(x) + torch.sum(torch.stack(
            (self.U_ol(h_left), self.U_or(h_right))), 0).squeeze() +
             self.b_o.expand(x.size(0), self.hidden_size))
        f_jl = torch.sigmoid(self.W_o(x) + torch.sum(torch.stack(
            (self.U_fl(h_left), self.U_fr(h_right))), 0).squeeze() +
             self.b_f.expand(x.size(0), self.hidden_size))
        f_jr = torch.sigmoid(self.W_o(x) + torch.sum(torch.stack(
            (self.U_fl(h_left), self.U_fr(h_right))), 0).squeeze() +
             self.b_f.expand(x.size(0), self.hidden_size))
        u_j = torch.tanh(self.W_u(x) + torch.sum(torch.stack(
            (self.U_ul(h_left), self.U_ur(h_right))), 0).squeeze() +
             self.b_u.expand(x.size(0), self.hidden_size))
        c_j = i_j * u_j + torch.sum(torch.stack((
            f_jl * c_left, f_jr * c_right)), 0).squeeze()
        h_j = o_j * torch.tanh(c_j)
        return h_j, c_j


class CSTreeLSTMCell(nn.Module):
    """A Child-Sum Tree-Long Short Term Memory cell as defined in
    Tai et al. (2015).

    """
    def __init__(self, input_size, hidden_size):
        super(CSTreeLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.W_i = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.W_f = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.W_o = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.W_u = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.U_i = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.U_f = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.U_o = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.U_u = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.reset_parameters()
        # Initialize biases after resetting of parameters; they can be zeroes.
        self.b_i = nn.Parameter(torch.Tensor(1, hidden_size))
        self.b_f = nn.Parameter(torch.Tensor(1, hidden_size))
        self.b_o = nn.Parameter(torch.Tensor(1, hidden_size))
        self.b_u = nn.Parameter(torch.Tensor(1, hidden_size))

    def reset_parameters(self):
        stdev = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdev, stdev)

    def forward(self, x, h, C):
        # Currently, batching (probably) does not work. Probably should be
        # fixed, at least for the sake of being consistent with the pytorch
        # recurrent models. x should be (N, D) where N is batch size and D
        # input size, h and C would be (N, C, H) where N is batch size, C is
        # the number of children and H is hidden size. Calculation of f_j
        # should be changed. Probabluy can be done with a nifty matrix op.
        h_tilde = torch.sum(h, 0).squeeze()
        i_j = torch.sigmoid(x @ self.W_i + h_tilde @ self.U_i + self.b_i)
        o_j = torch.sigmoid(x @ self.W_o + h_tilde @ self.U_o + self.b_o)
        f_x = x @ self.W_f
        f_j = torch.stack([torch.sigmoid(f_x + f @ self.U_f + self.b_f)
                           for f in h])
        u_j = torch.tanh(x @ self.W_u + h_tilde @ self.U_u + self.b_u)
        c_j = i_j * u_j + torch.sum(f_j * C, 0).squeeze()
        h_j = o_j * torch.tanh(c_j)
        return h_j, c_j


class MLPClassifier(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 3)
        self._init_parameters()

    def _init_parameters(self):
        torch.nn.init.kaiming_normal(self.fc1.weight)
        torch.nn.init.uniform(self.fc2.weight, -0.005, 0.005)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.log_softmax(self.fc2(x))
        return x


class BaselineNetwork(nn.Module):
    """A baseline network architecture for natural language inference. Each
    sentence is represented as a bag-of-words -- the sum of the word
    embeddings. This model is based on the bag-of-words baseline of Bowman et
    al. (2015).

    Args:
        embeddings (torch.LongTensor): A torch.LongTensor containing
        pre-trained word embeddings of dimensions (N, D) where N is the number
        of embeddings and D is the dimensionality of the embeddings.
    """
    def __init__(self, embeddings):
        super(BaselineNetwork, self).__init__()
        self.wemb = nn.Embedding(embeddings.size(0), embeddings.size(1))
        self.wemb.weight = nn.Parameter(embeddings)
        self.emb_transform = nn.Linear(embeddings.size(1), 100)
        self.fc1 = nn.Linear(200, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, 3)

    def forward(self, premise, hypothesis):
        x_premise = self.wemb(premise).sum(1).squeeze(1)
        x_hypothesis = self.wemb(hypothesis).sum(1).squeeze(1)
        x_premise = F.tanh(self.emb_transform(x_premise))
        x_hypothesis = F.tanh(self.emb_transform(x_hypothesis))
        x = torch.cat((x_premise, x_hypothesis), 1)
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = F.tanh(self.fc3(x))
        return F.log_softmax(x)


class SNLICorpus(torch.utils.data.Dataset):
    """Class for representing a specific split of the SNLI corpus.

    Args:
        path (string): Path to the corpus.
        vocab (dict): Dictionary mapping words to integers.
        pad (bool, optional): Whether to pad sentences. Needed if input to the
        network is supplied in batches.
        seq_length (int, optional): Max length of sentences. Sentences longer
        than this value will be cropped, and sentences that are shorter will be
        padded if pad is set to True.
    """
    def __init__(self, path, vocab, pad=True, seq_length=30):
        self.premises = []
        self.hypotheses = []
        self.premise_transitions = []
        self.hypothesis_transitions = []
        self.labels = []
        self.vocab = vocab
        self.pad = pad
        self.seq_length = seq_length
        self.label_map = {'neutral': [0],
                          'entailment': [1],
                          'contradiction': [2]}
        self._load_data(path)

    def _load_data(self, path):
        with open(path, 'r') as f:
            for i, line in enumerate(f):
                instance = json.loads(line)
                label = instance.get('gold_label')
                if label == '-':
                    continue
                gold_label = self.label_map.get(label)
                premise, premise_transitions = utils.convert_binary_bracketing(
                    instance.get('sentence1_binary_parse'))
                hypothesis, hypothesis_transitions = utils.convert_binary_bracketing(
                    instance.get('sentence2_binary_parse'))
                premise = [self.vocab.get(x, 1) for x in premise]
                hypothesis = [self.vocab.get(x, 1) for x in hypothesis]

                if self.pad:
                    premise, premise_transitions = self._pad_examples(
                        premise, premise_transitions)
                    hypothesis, hypothesis_transitions = self._pad_examples(
                        hypothesis, hypothesis_transitions)

                self.premises.append(torch.LongTensor(premise))
                self.hypotheses.append(torch.LongTensor(hypothesis))
                self.labels.append(torch.LongTensor(gold_label))
                self.premise_transitions.append(torch.LongTensor(
                    premise_transitions))
                self.hypothesis_transitions.append(torch.LongTensor(
                    hypothesis_transitions))

    def _pad_examples(self, tokens, transitions):
        transitions_left_pad = self.seq_length - len(transitions)
        shifts_before = transitions.count(0)
        transitions_padded = self._pad_and_crop(transitions,
                                                transitions_left_pad)
        shifts_after = transitions_padded.count(0)
        tokens_left_pad = shifts_after - shifts_before
        tokens_padded = self._pad_and_crop(tokens, tokens_left_pad)
        return tokens_padded, transitions_padded

    def _pad_and_crop(self, sequence, left_pad):
        if left_pad < 0:
            sequence = sequence[-left_pad:]
            left_pad = 0
        right_pad = self.seq_length - (left_pad + len(sequence))
        sequence = ([0] * left_pad) + sequence + ([0] * right_pad)
        return sequence

    def __getitem__(self, idx):
        return (self.premises[idx],
                self.hypotheses[idx],
                self.premise_transitions[idx],
                self.hypothesis_transitions[idx],
                self.labels[idx])

    def __len__(self):
        return len(self.premises)


def load_embeddings(path, dim):
    """Load pretrained word embeddings into format appropriate for the
    PyTorch network. Assumes word vectors are kept in a file with one vector
    per line, where the first column is the corresponding word.

    Args:
        path (str): Path to the word embeddings.
        dim (int): Dimensionality of the embeddings.
    Returns:
        dict: A dictionary mapping words to their vector index.
        torch.LongTensor: A torch.LongTensor of dimension (N, D) where N is
        the size of the vocabulary and D is the dimension of the embeddings.
    """
    embeddings = [[0 for _ in range(dim)],
                  [random.uniform(-0.1, 0.1) for _ in range(dim)]]
    word2id = {"<PAD>": 0}
    word2id = {"<UNKNOWN>": 1}
    with open(path, 'r') as f:
        for line in f:
            entry = line.strip().split()
            word = entry[0]
            word2id.update({word: len(word2id)})
            embeddings.append([float(x) for x in entry[1:]])

    return word2id, torch.FloatTensor(embeddings)


def train(model, data_loader, optimizer, epoch, log_interval=50):
    model.train()
    exec_time = 0
    correct = 0
    for batch, (premise, hypothesis, premise_transitions,
                hypothesis_transitions, target) in enumerate(data_loader):
        start_time = time.time()
        premise = Variable(premise)
        hypothesis = Variable(hypothesis)
        target = Variable(target.squeeze())
        optimizer.zero_grad()
        output = model(premise, hypothesis,
                       premise_transitions,
                       hypothesis_transitions)
        loss = F.nll_loss(output, target)
        _, pred = output.data.max(1)
        correct += pred.eq(target.data).sum()
        loss.backward()
        optimizer.step()
        end_time = time.time()
        exec_time += end_time - start_time
        if batch % log_interval == 0:
            print('Epoch {}, batch: {}/{} \tLoss: {:.6f}'.format(
                epoch, batch, len(data_loader.dataset) //
                data_loader.batch_size, loss.data[0]))
            print('Average time per batch: {:.3f} seconds.'.format(
               exec_time / (batch + 1)))

    print('Training accuracy epoch {}: {:d}/{:d} ({:.2%})'.format(epoch,
          correct, len(data_loader.dataset), correct / len(data_loader.dataset)
    ))


def test(model, data):
    model.eval()
    test_loss = 0
    correct = 0
    for batch, (premise, hypothesis, premise_transitions,
                hypothesis_transitions, target) in enumerate(data):
        premise = Variable(premise, volatile=True)
        hypothesis = Variable(hypothesis, volatile=True)
        target = Variable(target.squeeze())
        output = model(premise, hypothesis,
                       premise_transitions,
                       hypothesis_transitions)
        test_loss += F.cross_entropy(output, target).data[0]
        _, pred = output.data.max(1)
        correct += pred.eq(target.data).sum()

    average_test_loss = test_loss / len(data)
    return average_test_loss, correct

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--embeddings', type=str,
                        default='embeddings/glove.6B.100d.txt',
                        help='Path to pretrained word embeddings.')
    parser.add_argument('--wdim', type=int, default=100,
                        help='Dimensionality of the provided embeddings.')
    parser.add_argument('--edim', type=int, default=100,
                        help='Dimensionality of the sentence encoder.')
    parser.add_argument('--train', type=str,
                        default='data/snli_1.0_train.jsonl',
                        help='Path to training data.')
    parser.add_argument('--dev', type=str,
                        default='data/snli_1.0_dev.jsonl',
                        help='Path to development data.')
    parser.add_argument('--lr', type=float, default=2e-3,
                        help='Learning rate')
    parser.add_argument('--l2', type=float, default=3e-5,
                        help='L2 regularization term.')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Size of mini-batches.')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs to train for.')
    parser.add_argument('--log-interval', type=int, default=50,
                        help='Logs every n batches.')
    parser.add_argument('--save', type=str, default='model.pt',
                        help='Path to save the trained model.')
    parser.add_argument('--model', type=str, help='Path to a saved model.')
    parser.add_argument('--test', type=str, help="""Path to testing portion of
                        the corpus. If provided, --model argument has to be
                        given too.""")
    args = parser.parse_args()

    print("Loading data.")
    vocabulary, embeddings = load_embeddings(args.embeddings, args.wdim)
    train_loader = torch.utils.data.DataLoader(SNLICorpus(
        args.train, vocabulary, pad=True), batch_size=args.batch_size,
        shuffle=True, num_workers=1)
    dev_loader = torch.utils.data.DataLoader(SNLICorpus(
        args.dev, vocabulary), batch_size=args.batch_size)
    if args.model:
        model = torch.load(args.model)
    else:
        model = SPINNetwork(embeddings, args.edim, tracking=True)
    print('Model architecture:')
    print(model)

    learning_rate = args.lr
    best_dev_acc = 0
    for epoch in range(1, args.epochs + 1):
        try:
            parameters = filter(lambda p: p.requires_grad, model.parameters())
            optimizer = optim.RMSprop(parameters, lr=args.lr,
                                      weight_decay=args.l2)

            train(model, train_loader, optimizer, epoch,
                  log_interval=args.log_interval)

            test_loss, correct_dev = test(model, dev_loader)
            dev_accuracy = correct_dev / len(dev_loader)
            print("""\nDev:
                     \nAverage loss: {:.4f},
                      Accuracy: {:d}/{:d} ({:.2%})\n""".format(
                      test_loss, correct_dev, len(dev_loader), dev_accuracy))
            if dev_accuracy > best_dev_acc:
                with open(args.save, 'wb') as f:
                    torch.save(model, f)
                    print("Saved model to {}".format(args.save))

            # decay learning rate every other epoch
            # In Bowman et al. (2016) they do it every 10k steps.
            if epoch % 2 == 0:
                learning_rate *= 0.75
        except KeyboardInterrupt:
            print("Training aborted early.")
            break

