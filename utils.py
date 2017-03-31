import StanfordDependencies
import json
import torch
import copy


class Node(object):
    def __init__(self, idx, head, word=None):
        self.idx = idx
        self.head = head
        self.word = word
        self.children = set()


class Tree(object):
    def __init__(self):
        self.nodes = []

    def update(self, head, child):
        self.nodes[child].head = None
        self.nodes[head].children.add(child)


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
        idx, token, head = word.split(":")
        node = Node(int(idx), int(head), word=token)
        dep_tree.nodes.append(node)
    for node in dep_tree.nodes:
        # The root node is not the child of ANYONE
        if node.idx == 0:
            continue
        dep_tree.nodes[node.head].children.add(node.idx)
    for node in dep_tree.nodes:
        print("node idx: {}, head: {}, word: {}, children: {}".format(
            node.idx, node.head, node.word, node.children))
    return dep_tree


def convert_trees(input_file, output_file):
    sd = StanfordDependencies.get_instance(backend='jpype')
    with open(input_file, 'r') as inpf, open(output_file, 'w') as outf:
        for line in inpf:
            sentence = json.loads(line)
            prem = sentence.get('sentence1_parse')
            hypo = sentence.get('sentence2_parse')
            prem_dep = ' '.join(['{}:{}:{}'.format(x.index, x.form, x.head)
                                 for x in sd.convert_tree(prem)])
            hypo_dep = ' '.join(['{}:{}:{}'.format(x.index, x.form, x.head)
                                 for x in sd.convert_tree(hypo)])
            sentence.update({'sentence1_dependency_parse': prem_dep,
                             'sentence2_dependency_parse': hypo_dep})
            outf.write(json.dumps(sentence) + "\n")

    print("Wrote file with dependency parse annotation to {}".format(
           output_file))


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
        elif token == ")":
            transitions.append(1)
        else:
            transitions.append(0)
            tokens.append(token.lower())

    return tokens, transitions


def get_dependency_transitions(sentence):
    """Return a list of transitions that leads to the given dependency tree.

    Args:
        sentence (list): A list of tokens in the sentence where each element
        is structured as "idx(int):word(str):head(int)".
    Returns:
        list: A list of transitions in {0, 1, 2} that leads to this parse tree,
        where 0 is shift, 1 is reduce-left and 2 is reduce-right.
    """

    gold_tree = get_gold_tree(sentence)
    parse_tree = get_parse_tree(gold_tree)
    buffer = [x.idx for x in parse_tree.nodes]
    stack = [buffer.pop(0)]
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
        gold_head = gold_tree.nodes[head]
        gold_child = gold_tree.nodes[child]
        potential_child = parse_tree.nodes[child]
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
            transitions.append(1)
            parse_tree.update(stack[-1], stack.pop(-2))
        elif right_is_valid():
            transitions.append(2)
            parse_tree.update(stack[-2], stack.pop(-1))
        elif shift_is_valid():
            transitions.append(0)
            stack.append(buffer.pop(0))

    return transitions
