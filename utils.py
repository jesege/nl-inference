import StanfordDependencies
import json
import torch
import copy
import random


class Node(object):
    def __init__(self, idx, head, word=None):
        self.idx = idx
        self.head = head
        self.word = word
        self.children = set()


class Tree(object):
    def __init__(self):
        self.nodes = []
        self.node_idxs = []

    def update(self, head, child):
        self.nodes[child].head = head
        self.nodes[head].children.add(child)

    def add_node(self, node):
        self.nodes.append(node)

    def __getitem__(self, idx):
        return self.nodes[idx]

    def __iter__(self):
        for node in self.nodes:
            yield node


def get_parse_tree(gold_tree):
    parse_tree = copy.deepcopy(gold_tree)
    for node in parse_tree.nodes:
        node.head = None
        node.children = set()
    return parse_tree


def get_queue_item(lst, idx):
    try:
        return lst[idx]
    except IndexError:
        return 0


def get_gold_tree(sentence):
    dep_tree = Tree()
    root = Node(0, None, word='ROOT')
    dep_tree.nodes.append(root)
    for word in sentence.split(' '):
        idx, token, head, deprel = word.split("(")
        node = Node(int(idx), int(head), word=token)
        dep_tree.add_node(node)
    for node in dep_tree:
        # The root node is not the child of ANYONE
        if node.idx == 0:
            continue
        head = dep_tree[node.head]
        dep_tree[node.head].children.add(node.idx)
    # for node in dep_tree.nodes:
    #     print("node idx: {}, head: {}, word: {}, children: {}".format(
    #         node.idx, node.head, node.word, node.children))
    return dep_tree


def convert_trees(input_file, output_file):
    sd = StanfordDependencies.get_instance(backend='jpype')
    with open(input_file, 'r') as inpf, open(output_file, 'w') as outf:
        for line_no, line in enumerate(inpf):
            if line_no % 1000 == 0:
                print("Processing sentence pair {}.".format(line_no))
            sentence = json.loads(line)
            prem = sentence.get('sentence1_parse')
            hypo = sentence.get('sentence2_parse')
            prem_dep = ' '.join(['{}({}({}({}'.format(x.index, x.form, x.head, x.deprel)
                                 for x in sd.convert_tree(prem)])
            hypo_dep = ' '.join(['{}({}({}({}'.format(x.index, x.form, x.head, x.deprel)
                                 for x in sd.convert_tree(hypo)])
            sentence.update({'sentence1_dependency_parse': prem_dep,
                             'sentence2_dependency_parse': hypo_dep})
            outf.write(json.dumps(sentence) + "\n")

    print("Wrote file with dependency parse annotation to {}".format(
           output_file))


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

    # TODO: Currently, we have an actual ROOT node. Should we have it?
    # If not we can slice. tokens[1:], transitions[1:len(transitions)-1]
    gold_tree = get_gold_tree(sentence)
    parse_tree = get_parse_tree(gold_tree)
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


def pad_examples(tokens, transitions, seq_len):
    transitions_left_pad = seq_len - len(transitions)
    shifts_before = transitions.count(0)
    transitions_padded = pad_and_crop(transitions,
                                      transitions_left_pad, seq_len)
    shifts_after = transitions_padded.count(0)
    tokens_left_pad = shifts_after - shifts_before
    tokens_padded = pad_and_crop(tokens, tokens_left_pad, seq_len)
    return tokens_padded, transitions_padded


def pad_and_crop(sequence, left_pad, seq_len, transitions):
    if left_pad < 0:
        sequence = sequence[-left_pad:]
        left_pad = 0
    right_pad = seq_len - (left_pad + len(sequence))
    sequence = ([0] * left_pad) + sequence + ([0] * right_pad)
    return sequence

