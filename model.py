import argparse
import time
import os.path
import sys
import pickle
import json
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
        if tracking_lstm:
            self.tracking_lstm = nn.LSTMCell(3 * hidden_size,
                                             tracking_lstm_dim)
        else:
            self.tracking_lstm = None
        self.encoder = TreeLSTMCell(tracking_lstm_dim, self.hidden_size)

    def forward(self, tokens_h, tokens_c, transitions):
        batch_size = transitions.size(0)
        timesteps = transitions.size(1)
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
        for timestep in range(1, timesteps + 1):
            mask = transitions[timestep - 1].int()
            if self.tracking_lstm:
                h_stack1 = torch.cat([stack[-1][0] for stack in stacks])
                h_stack2 = torch.cat([stack[-2][0] for stack in stacks])
                buffer_top = torch.cat([bh[0] for bh in buffers_h])
                tlstm_in = torch.cat((buffer_top, h_stack1, h_stack2), 1)
                tlstm_hidden, tlstm_cell = self.tracking_lstm(tlstm_in,
                                                              (tlstm_hidden,
                                                               tlstm_cell))

            h_head, c_head, h_child, c_child, tracking = [], [], [], [], []

            for i, (transition, stack, buf_h, buf_c) in enumerate(zip(mask,
                                                                  stacks,
                                                                  buffers_h,
                                                                  buffers_c)):
                if transition == 1:  # SHIFT
                    stack.append((buf_h.pop(0), buf_c.pop(0)))
                elif transition == 2:  # LEFT-ARC
                    h_he, c_he = stack.pop()
                    h_ch, c_ch = stack.pop()
                    h_head.append(h_he)
                    c_head.append(c_he)
                    h_child.append(h_ch)
                    c_child.append(c_ch)
                    if self.tracking_lstm:
                        tracking.append(tlstm_hidden[i])
                elif transition == 3:  # RIGHT-ARC
                    h_ch, c_ch = stack.pop()
                    h_he, c_he = stack.pop()
                    h_head.append(h_he)
                    c_head.append(c_he)
                    h_child.append(h_ch)
                    c_child.append(c_ch)

            if h_head:
                h_head = torch.cat(h_head)
                c_head = torch.cat(c_head)
                h_child = torch.cat(h_child)
                c_child = torch.cat(c_child)
                if self.tracking_lstm:
                    tracking = torch.stack(tracking)
                else:
                    tracking = None
                reduced_h, reduced_c = self.encoder(tracking,
                                                    (h_head, c_head),
                                                    (h_child, c_child))
                reduced_h = iter(reduced_h)
                reduced_c = iter(reduced_c)

                for trans, stack in zip(mask, stacks):
                    if trans == 2 or trans == 3:  # IF WE REDUCE
                        stack.append((next(reduced_h).unsqueeze(0),
                                      next(reduced_c).unsqueeze(0)))

        out = torch.cat([example[-1][0] for example in stacks], 0)
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
        else:
            self.tracking_lstm = None
        self.encoder = TreeLSTMCell(self.tlstm_dim, self.hidden_size)

    def forward(self, tokens_h, tokens_c, transitions):
        """Encode sentence in the sequence given the transitions.

        Args:
            tokens_h (torch.Tensor): Tensor containing the h-representations of
            the sentences to encode, of size (B x L x D), where B is batch
            size, L is length of the sequences and D is the dimensionality of
            the data.

            tokens_c (torch.Tensor): Tensor containing the c-representations of
            the sentences to encode, of size (B x L x D), where B is batch
            size, L is length of the sequences and D is the dimensionality of
            the data.

            transitions (torch.Tensor): Tensor containing the transitions for
            each sequence of size (B x T), where B is batch size and T is the
            number of transitions.

        Returns:
            torch.Tensor: Tensor of size (B x D) containing the final hidden
            states of the given sequences.
        """
        batch_size = transitions.size(0)
        timesteps = transitions.size(1)
        buffers_h = [list(torch.split(x.squeeze(), 1, 0))
                     for x in tokens_h.split(1, 0)]
        buffers_c = [list(torch.split(x.squeeze(), 1, 0))
                     for x in tokens_c.split(1, 0)]
        # TODO: Can we initialize the stacks as just empty lists?
        stacks = [[(bh[0], bc[0]), (bh[0], bc[0])] for bh, bc in
                  zip(buffers_h, buffers_c)]
        if self.tracking_lstm:
            tlstm_hidden = Variable(torch.randn(batch_size, self.tlstm_dim))
            tlstm_cell = Variable(torch.randn(batch_size, self.tlstm_dim))

        # if we transpose the transitions matrix we can index it like
        # transitions[timestep] to get all the data for the timestep
        transitions = transitions.t()
        for timestep in range(1, timesteps + 1):
            mask = transitions[timestep - 1].int()
            if self.tracking_lstm:
                h_stack1 = torch.cat([stack[-1][0] for stack in stacks])
                h_stack2 = torch.cat([stack[-2][0] for stack in stacks])
                buffer_top = torch.cat([bh[0] for bh in buffers_h])
                tlstm_in = torch.cat((buffer_top, h_stack1, h_stack2), 1)
                tlstm_hidden, tlstm_cell = self.tracking_lstm(tlstm_in,
                                                              (tlstm_hidden,
                                                               tlstm_cell))

            h_right, c_right, h_left, c_left, tracking = [], [], [], [], []

            for i, (transition, stack, buf_h, buf_c) in enumerate(zip(mask,
                                                                  stacks,
                                                                  buffers_h,
                                                                  buffers_c)):
                if transition == 1:  # SHIFT
                    stack.append((buf_h.pop(0), buf_c.pop(0)))
                elif transition == 2:  # REDUCE
                    h_r, c_r = stack.pop()
                    h_l, c_l = stack.pop()
                    h_right.append(h_r), c_right.append(c_r)
                    h_left.append(h_l), c_left.append(c_l)
                    if self.tracking_lstm:
                        tracking.append(tlstm_hidden[i])

            if h_right:
                h_right = torch.cat(h_right)
                c_right = torch.cat(c_right)
                h_left = torch.cat(h_left)
                c_left = torch.cat(c_left)
                if self.tracking_lstm:
                    tracking = torch.stack(tracking)
                else:
                    tracking = None
                reduced_h, reduced_c = self.encoder(tracking,
                                                    (h_right, c_right),
                                                    (h_left, c_left))
                reduced_h = iter(reduced_h)
                reduced_c = iter(reduced_c)

                for trans, stack in zip(mask, stacks):
                    if trans == 2:  # IF WE REDUCE
                        stack.append((next(reduced_h).unsqueeze(0),
                                      next(reduced_c).unsqueeze(0)))

        out = torch.cat([example[-1][0] for example in stacks], 0)
        return out


class BOWEncoder(nn.Module):
    def __init__(self):
        super(StackEncoder, self).__init__()

    def forward(self, tokens_h, tokens_c, transitions):
        sequence = torch.cat((tokens_h, tokens_c), 2)
        return sequence.sum(1).squeeze(1)


class SPINNetwork(nn.Module):
    def __init__(self, embeddings, encoder):
        super(SPINNetwork, self).__init__()
        self.wemb = nn.Embedding(embeddings.size(0), embeddings.size(1),
                                 padding_idx=0)
        self.wemb.weight = nn.Parameter(embeddings)
        self.wemb.weight.requires_grad = False
        self.encoder_dim = encoder.hidden_size
        self.projection_dim = self.encoder_dim * 2
        self.projection = nn.Linear(embeddings.size(1),
                                    self.projection_dim)
        self.batch_norm = nn.BatchNorm1d(self.projection_dim)
        self.encoder = encoder
        self.classifier = MLPClassifier(self.encoder_dim * 4, 1024)
        torch.nn.init.kaiming_normal(self.projection.weight)

    def forward(self, prem_sequence, hypo_sequence, prem_transitions,
                hypo_transitions):
        seq_len = prem_sequence.size(1)
        prem_emb = self.wemb(prem_sequence)
        hypo_emb = self.wemb(hypo_sequence)
        # Repack the embeddings in new Variables since
        # we don't  want to update the embedding layer
        # prem_emb = Variable(prem_emb.data)
        # hypo_emb = Variable(prem_emb.data)
        prem_proj = torch.stack([self.projection(prem_emb[:, i, :])
                                 for i in range(prem_emb.size(1))],
                                dim=1)
        hypo_proj = torch.stack([self.projection(hypo_emb[:, i, :])
                                 for i in range(hypo_emb.size(1))],
                                dim=1)
        prem_bnorm = self.batch_norm(prem_proj.view(-1, self.projection_dim))
        hypo_bnorm = self.batch_norm(hypo_proj.view(-1, self.projection_dim))
        h_prem, c_prem = prem_bnorm.view(-1, seq_len,
                                         self.projection_dim).chunk(2, 2)
        h_hypo, c_hypo = hypo_bnorm.view(-1, seq_len,
                                         self.projection_dim).chunk(2, 2)
        prem_encoded = self.encoder(h_prem, c_prem, prem_transitions)
        hypo_encoded = self.encoder(h_hypo, c_hypo, hypo_transitions)
        x_classifier = torch.cat((hypo_encoded, prem_encoded,
                                  hypo_encoded - prem_encoded,
                                  hypo_encoded * prem_encoded), 1)
        return self.classifier(x_classifier)


class TreeLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(TreeLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.W = nn.Linear(input_size, hidden_size * 5, bias=False)
        self.U_r = nn.Linear(hidden_size, hidden_size * 5, bias=False)
        self.U_l = nn.Linear(hidden_size, hidden_size * 5)
        # self.init_parameters()

    def init_parameters(self):
        for weight in self.parameters():
            torch.nn.init.kaiming_normal(weight.data)

    def forward(self, x, hc_right, hc_left):
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

    def forward(self, x, hc_right, hc_left):
        # Inputs = (B X D)
        h_right, c_right = hc_right
        h_left, c_left = hc_left

        i_j = (self.U_il(h_left) + self.U_il(h_right)) + self.b_i.expand(
                h_right.size(0), self.hidden_size)
        o_j = (self.U_ol(h_left) + self.U_or(h_right)) + self.b_o.expand(
                h_right.size(0), self.hidden_size)
        f_jl = (self.U_fl(h_left) + self.U_fr(h_right)) + self.b_f.expand(
                h_right.size(0), self.hidden_size)
        f_jr = (self.U_fl(h_left) + self.U_fr(h_right)) + self.b_f.expand(
                h_right.size(0), self.hidden_size)
        u_j = (self.U_ul(h_left) + self.U_ur(h_right)) + self.b_u.expand(
                h_right.size(0), self.hidden_size)

        if x is not None:
            o_j = self.W_o(x) + o_j
            f_jl = self.W_f(x) + f_jl
            f_jr = self.W_f(x) + f_jr
            u_j = self.W_u(x) + u_j

        c_j = i_j.sigmoid() * u_j.tanh() + ((f_jl.sigmoid() * c_left) +
                                            (f_jr.sigmoid() * c_right))
        h_j = o_j.sigmoid() * torch.tanh(c_j)
        return h_j, c_j


class MLPClassifier(nn.Module):
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
        network is to be supplied in batches.
        seq_length (int, optional): Max length of transition sequences.
        Sentences with a larger number of transitions will be removed from the
        data.
        dependency (bool, optional): If set to True, the transitions will be
        according to the dependency tree of the sentence. Defaults to False.
    """
    def __init__(self, path, vocab, pad=True, seq_length=50, dependency=False):
        self.examples = []
        self.vocab = vocab
        self.pad = pad
        self.seq_length = seq_length
        self.token_length = seq_length // 2
        self.dependency = dependency
        self.label_map = {'neutral': [0],
                          'entailment': [1],
                          'contradiction': [2]}
        self._load_data(path)

    def _load_data(self, path):
        sentences_removed = 0
        with open(path, 'r') as f:
            for i, line in enumerate(f):
                instance = json.loads(line)
                label = instance.get('gold_label')
                if label == '-':
                    continue
                gold_label = self.label_map.get(label)
                if self.dependency:
                    premise, premise_transitions = utils.get_dependency_transitions(
                        instance.get('sentence1_dependency_parse'))
                    hypothesis, hypothesis_transitions = utils.get_dependency_transitions(
                        instance.get('sentence2_dependency_parse'))
                else:
                    premise, premise_transitions = utils.convert_binary_bracketing(
                        instance.get('sentence1_binary_parse'))
                    hypothesis, hypothesis_transitions = utils.convert_binary_bracketing(
                        instance.get('sentence2_binary_parse'))
                premise = [self.vocab.get(x, 1) for x in premise]
                hypothesis = [self.vocab.get(x, 1) for x in hypothesis]
                example = {}
                if (len(premise_transitions) > self.seq_length or
                   len(hypothesis_transitions) > self.seq_length):
                    continue

                if premise and hypothesis:
                    example["premise"] = premise
                    example["hypothesis"] = hypothesis
                    example["premise_transition"] = premise_transitions
                    example["hypothesis_transition"] = hypothesis_transitions
                    example["label"] = gold_label
                    example["prem_len"] = len(premise)
                    example["hypo_len"] = len(hypothesis)
                    self.examples.append(example)
                else:
                    sentences_removed += 1
        print("Removed {} sentences.".format(sentences_removed))

    def __getitem__(self, idx):
        return self.examples[idx]

    def __len__(self):
        return len(self.examples)


def train(model, data_loader, optimizer, epoch, log_interval=50):
    model.train()
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
        exec_time = end_time - start_time
        if batch % log_interval == 0:
            print('Epoch {}, batch: {}/{} \tLoss: {:.6f}'.format(
                epoch, batch, len(data_loader.dataset) //
                data_loader.batch_size, loss.data[0]))
            print('Execution time last batch: {:.3f} seconds.'.format(
               exec_time))

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
    parser.add_argument('--wv-cache', type=str,
                        default='data/glove100d-cache.pt',
                        help='Path from where to load or save vector cache')
    parser.add_argument('--vocab', type=str, default='data/vocabulary.pkl')
    parser.add_argument('--wdim', type=int, default=100,
                        help='Dimensionality of the provided embeddings.')
    parser.add_argument('--edim', type=int, default=100,
                        help='Dimensionality of the sentence encoder.')
    parser.add_argument('--tdim', type=int, default=64,
                        help='Dimensionality of the tracking LSTM.')
    parser.add_argument('--tracking', action='store_true',
                        help='Use a tracking LSTM.')
    parser.add_argument('--dependency', action='store_true',
                        help='Use the dependency based encoder.')
    parser.add_argument('--train', type=str,
                        default='data/snli_1.0_train.jsonl',
                        help='Path to training data.')
    parser.add_argument('--dev', type=str,
                        default='data/snli_1.0_dev.jsonl',
                        help='Path to development data.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--l2', type=float, default=3e-5,
                        help='L2 regularization term.')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Size of mini-batches.')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs to train for.')
    parser.add_argument('--log-interval', type=int, default=50,
                        help='Logs every n batches.')
    parser.add_argument('--lr-decay-every', type=int, default=10000,
                        help='Decay lr every n steps.')
    parser.add_argument('--lr-decay', type=float, default=0.75,
                        help='Learning rate decay factor.')
    parser.add_argument('--save', type=str, default='models/model',
                        help='Path to save the trained model.')
    parser.add_argument('--model', type=str, help='Path to a saved model.')
    parser.add_argument('--test', type=str, help="""Path to testing portion of
                        the corpus. If provided, --model argument has to be
                        given too.""")
    args = parser.parse_args()

    print("Loading data.")

    if os.path.isfile(args.vocab) and os.path.isfile(args.wv_cache):
        with open(args.vocab, 'rb') as f:
            vocabulary = pickle.load(f)
        embeddings = torch.load(args.wv_cache)
    else:
        vocabulary, embeddings = utils.load_embeddings(args.embeddings,
                                                       args.wdim)
        with open(args.vocab, 'wb') as f:
            pickle.dump(vocabulary, f)
        torch.save(embeddings, args.wv_cache)

    train_loader = torch.utils.data.DataLoader(SNLICorpus(
        args.train, vocabulary, pad=True, dependency=args.dependency),
        batch_size=args.batch_size, shuffle=True, num_workers=1,
        collate_fn=utils.collate_transitions)
    dev_loader = torch.utils.data.DataLoader(SNLICorpus(
        args.dev, vocabulary), batch_size=args.batch_size,
        collate_fn=utils.collate_transitions)

    if args.model:
        model = torch.load(args.model)
    else:
        if args.dependency:
            encoder = DependencyEncoder(args.edim, tracking_lstm=args.tracking,
                                        tracking_lstm_dim=args.tdim)
        else:
            encoder = StackEncoder(args.edim, tracking_lstm=args.tracking,
                                   tracking_lstm_dim=args.tdim)
        model = SPINNetwork(embeddings, encoder)

    print('Model architecture:')
    print(model)
    if args.test:
        assert args.model, "You need to provide a model to test."
        test_loader = torch.utils.data.DataLoader(SNLICorpus(
            args.test, vocabulary), batch_size=args.batch_size,
            collate_fn=utils.collate_transitions)
        test_loss, correct = test(model, test_loader)
        test_accuracy = correct / len(test_loader.dataset)
        print("""\nTest:
            Average loss: {:.4f},
            Accuracy: {:d}/{:d} ({:.2%})\n""".format(
                      test_loss, correct, len(test_loader.dataset),
                      test_accuracy))
        sys.exit()

    learning_rate = args.lr
    best_dev_acc = 0
    saving_prefix = args.save
    iteration = 0
    # TODO: Add learning rate decay.
    for epoch in range(1, args.epochs + 1):
        try:
            parameters = filter(lambda p: p.requires_grad, model.parameters())
            optimizer = optim.RMSprop(parameters, lr=args.lr,
                                      weight_decay=args.l2)
            correct_train = 0
            for batch, (premise, hypothesis, premise_transitions,
                        hypothesis_transitions, target) in enumerate(train_loader):
                model.train()
                iteration += 1
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
                correct_train += pred.eq(target.data).sum()
                loss.backward()
                optimizer.step()
                end_time = time.time()
                exec_time = end_time - start_time
                if iteration % args.lr_decay_every == 0:
                    for pg in optimizer.param_groups:
                        pg['lr'] = args.lr * (args.lr_decay **
                                              (iteration / args.lr_decay_every))
                if batch % args.log_interval == 0:
                    print('Epoch {}, batch: {}/{} \tLoss: {:.6f}'.format(
                        epoch, batch, len(train_loader.dataset) //
                        train_loader.batch_size, loss.data[0]))
                    print('Exec time last batch: {:.3f} seconds.'.format(
                       exec_time))

            print('Training acc. epoch {}: {:d}/{:d} ({:.2%})'.format(epoch,
                  correct_train, len(train_loader.dataset),
                  correct_train / len(train_loader.dataset)))

            test_loss, correct_dev = test(model, dev_loader)
            dev_accuracy = correct_dev / len(dev_loader.dataset)
            print("""\nDev:
            Average loss: {:.4f},
            Accuracy: {:d}/{:d} ({:.2%})\n""".format(
                      test_loss, correct_dev, len(dev_loader.dataset),
                      dev_accuracy))
            if dev_accuracy > best_dev_acc:
                best_dev_acc = dev_accuracy
                saving_suffix = "-devacc{:.2f}-iters{}.pt".format(dev_accuracy,
                                                                  iteration)
                save_path = saving_prefix + saving_suffix
                with open(save_path, 'wb') as f:
                    torch.save(model, f)
                print("New best dev accuracy: {:.2f}".format(dev_accuracy))
                print("Saved model to {}".format(save_path))

        except KeyboardInterrupt:
            print("Training aborted early.")
            break
