import StanfordDependencies
import json


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
            sentence.update({'sentence1_dep_parse': prem_dep,
                             'sentence2_dep_parse': hypo_dep})
            outf.write(json.dumps(sentence) + "\n")

    print("Wrote file with dependency parse annotation to {}".format(
           output_file))
