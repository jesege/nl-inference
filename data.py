import torch
import logging
import json
import utils
import random


class SNLICorpus(torch.utils.data.Dataset):
    """Class for representing a specific split of the SNLI corpus.

    Args:
        path (string): Path to the corpus.
        vocab (dict): Dictionary mapping words to integers.
        seq_length (int, optional): Max length of transition sequences.
        Sentences with a larger number of transitions will be removed from the
        data.
        dependency (bool, optional): If set to True, the transitions will be
        according to the dependency tree of the sentence. Defaults to False.
    """
    def __init__(self, path, vocab, seq_length=50, dependency=False):
        self.examples = []
        self.vocab = vocab
        self.seq_length = seq_length
        self.token_length = seq_length // 2
        self.dependency = dependency
        self.label_map = {'neutral': [0],
                          'entailment': [1],
                          'contradiction': [2]}
        self._load_data(path)

    def _load_data(self, path):
        logger = logging.getLogger("model.data_loading")
        sentences_removed = 0
        long_sentences = 0
        with open(path, 'r') as f:
            for i, line in enumerate(f):
                instance = json.loads(line)
                label = instance.get('gold_label')
                if label == '-':
                    continue
                gold_label = self.label_map.get(label)
                if self.dependency:
                    premise, prem_transitions = get_dependency_transitions(
                        instance.get('sentence1_dependency_parse'))
                    hypothesis, hypo_transitions = get_dependency_transitions(
                        instance.get('sentence2_dependency_parse'))
                else:
                    premise, prem_transitions = convert_binary_bracketing(
                        instance.get('sentence1_binary_parse'))
                    hypothesis, hypo_transitions = convert_binary_bracketing(
                        instance.get('sentence2_binary_parse'))
                premise = [self.vocab.get(x, 1) for x in premise]
                hypothesis = [self.vocab.get(x, 1) for x in hypothesis]
                example = {}
                if (len(prem_transitions) > self.seq_length or
                   len(hypo_transitions) > self.seq_length):
                    long_sentences += 1
                    continue

                if premise and hypothesis:
                    example["premise"] = premise
                    example["hypothesis"] = hypothesis
                    example["premise_transition"] = prem_transitions
                    example["hypothesis_transition"] = hypo_transitions
                    example["label"] = gold_label
                    example["prem_len"] = len(premise)
                    example["hypo_len"] = len(hypothesis)
                    self.examples.append(example)
                else:
                    sentences_removed += 1

        logger.info("Skipped %d long sentences from file %s." %
                    (long_sentences, path))
        if sentences_removed > 0:
            logger.info(
                "Could not derive transition sequences for %d sentences from file %s." %
                (sentences_removed, path))

    def __getitem__(self, idx):
        return self.examples[idx]

    def __len__(self):
        return len(self.examples)


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
        the size of the vocabulary and D is the dimensionality of the
        embeddings.
    """
    embeddings = [[0 for _ in range(dim)],
                  [random.uniform(-0.1, 0.1) for _ in range(dim)],
                  [random.uniform(-0.1, 0.1) for _ in range(dim)]]
    word2id = {"<PAD>": 0, "<UNK>": 1, "ROOT": 2}
    with open(path, 'r') as f:
        for line in f:
            entry = line.strip().split()
            word = entry[0]
            word2id.update({word: len(word2id)})
            embeddings.append([float(x) for x in entry[1:]])

    return word2id, torch.FloatTensor(embeddings)


def convert_binary_bracketing(sentence):
    """Convert the sentence, represented as an s-expression, to a sequence
    of tokens and transitions.

    Args:
        sentence (str): The sentece represented as a binary parse in the form
        of an s-expression.
    Returns:
        tuple: A tuple containing a list of the tokens in the sentence and a
        list of transitions in {0, 1} that leads to this parse tree, where 0
        is shift and 1 is reduce.
    """
    tokens = []
    transitions = []
    for token in sentence.split(' '):
        if token == "(":
            continue
        elif token == ")":  # REDUCE
            transitions.append(2)
        else:
            transitions.append(1)  # SHIFT
            tokens.append(token.lower())

    return tokens, transitions


def get_dependency_transitions(sentence):
    """Return a list of transitions that leads to the given dependency tree.

    Args:
        sentence (list): A list of tokens in the sentence where each element
        is structured as "idx(int):word(str):head(int)".
    Returns:
        list: A list of transitions in {1, 2, 3} that leads to this parse tree,
        where 1 is shift, 2 is reduce-left and 3 is reduce-right.
    """

    gold_tree = utils.get_gold_tree(sentence)
    parse_tree = utils.get_parse_tree(gold_tree)
    tokens = [x.word.lower() for x in gold_tree]
    buffer = [x.idx for x in parse_tree]
    # print(buffer)
    stack = []
    transitions = []

    def state_is_terminal():
        return len(stack) == 1 and stack[-1] == 0 and len(buffer) == 0

    def left_is_valid():
        return len(stack) >= 2 and stack[-2] != 0 and transition_is_correct(
                                                      stack[-1], stack[-2])

    def right_is_valid():
        return len(stack) >= 2 and transition_is_correct(stack[-2], stack[-1])

    def shift_is_valid():
        return len(buffer) > 0

    def transition_is_correct(head, child):
        gold_head = gold_tree[head]
        gold_child = gold_tree[child]
        potential_child = parse_tree[child]
        # Check if the assumed child is in fact a child of the given head
        if gold_child.idx not in gold_head.children:
            return False
        # If we have not found all children for this node we should not do an
        # arc transition
        if potential_child.children != gold_child.children:
            return False
        return True

    while not state_is_terminal():
        if left_is_valid():
            transitions.append(2)
            parse_tree.update(stack[-1], stack.pop(-2))
        elif right_is_valid():
            transitions.append(3)
            parse_tree.update(stack[-2], stack.pop(-1))
        elif shift_is_valid():
            transitions.append(1)
            stack.append(buffer.pop(0))
        else:
            return [], []

    return tokens[1:], transitions[1:len(transitions)-1]


def collate_transitions(batch):
    premises = []
    hypotheses = []
    prem_trans = []
    hypo_trans = []
    labels = []
    prem_len = max([example["prem_len"] for example in batch])
    hypo_len = max([example["hypo_len"] for example in batch])
    # We add an extra padding token in order to avoid problems with indexing
    # for the tracking LSTM. Only add an extra to the tokens, not for the
    # transitions!
    token_len = max(prem_len, hypo_len) + 1
    trans_len = token_len * 2 - 2
    for example in batch:
        premise = pad_example(example["premise"], token_len)
        prem_tran = pad_example(example["premise_transition"], trans_len)
        hypothesis = pad_example(example["hypothesis"], token_len)
        hypo_tran = pad_example(example["hypothesis_transition"], trans_len)
        premises.append(torch.LongTensor(premise))
        hypotheses.append(torch.LongTensor(hypothesis))
        prem_trans.append(torch.LongTensor(prem_tran))
        hypo_trans.append(torch.LongTensor(hypo_tran))
        labels.append(torch.LongTensor(example["label"]))
    return (torch.stack(premises), torch.stack(hypotheses),
            torch.stack(prem_trans), torch.stack(hypo_trans),
            torch.stack(labels))


def pad_example(ex, trgt_len):
    padding = trgt_len - len(ex)
    return ex + [0] * padding
