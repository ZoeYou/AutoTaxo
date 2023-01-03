# script used to seperate entities with descriptive contents
from pathlib import Path
import multiprocessing, pickle, re, copy
from anytree import RenderTree, PostOrderIter, PreOrderIter

global preposition_list
preposition_list = ['about', 'above', 'across', 'after', 'against', 'along', 'among', 'around', 'at', 'before', 'behind', 'between', 'beyond', 'but', 'by', 'despite', 'down', 'during', 'except', 'for', 'from', 'in', 'into', 'like', 'near', 'of', 'off', 'on', 'onto', 'out', 'over', 'past', 'plus', 'since', 'throughout', 'to', 'towards', 'under', 'until', 'up', 'upon', 'with', 'within', 'without', 'having', 'comprising', 'or']

input_path = Path('./trees_CPC8-3')
tree_files = sorted([str(f) for f in input_path.glob("*.pickle")])

def load_tree(file_name):
    with open(file_name, "rb") as inf:
        root_node = pickle.load(inf)
    return root_node

descriptive_patterns = r'accessories|arrangements?|applications?|apparatus?|appliances?|^methods?|methods or|methods for| methods|details|means for|devices? | devices?|^tools?|or methods?|or implements?|^machines for|machines or|instruments for|implements for|equipments? for|specially adapted|characterised by|^types? of|^special|systems using|instruments? employing|^measurement? of|therefor|thereof|therewith|thereby| designed for |^treatment |aspects of|particular use of|general design| them | their|^preparation'

for file in tree_files:
    tree_entities = copy.deepcopy(load_tree(file))
    tree_descriptions = copy.deepcopy(load_tree(file))

    nodes_entity_todo = [node for node in PostOrderIter(tree_entities)]
    nodes_desc_todo = [node for node in PostOrderIter(tree_descriptions)]
    
    for node in nodes_entity_todo:
        if '@@@' in node.name or re.search(descriptive_patterns, node.name.lower(), re.IGNORECASE) or len([word for word in node.name.lower().split() if word in preposition_list])>2 or len(node.name.split())>5:
            node.name = ''

    for node in nodes_desc_todo:
        if not ('@@@' in node.name or re.search(descriptive_patterns, node.name.lower(), re.IGNORECASE) or len([word for word in node.name.lower().split() if word in preposition_list])>2 or len(node.name.split())>5):
            node.name = ''

    # save updated tree in a new file
    output_file_ents = input_path / (tree_entities.name + '_entities.txt')
    output_file_desc = input_path / (tree_entities.name + '_desc.txt')

    with output_file_ents.open('w') as out_f:
        for pre, _, node in RenderTree(tree_entities):
            out_f.write("%s%s" % (pre, node.name))
            out_f.write("\n")

    with output_file_desc.open('w') as out_f:
        for pre, _, node in RenderTree(tree_descriptions):
            out_f.write("%s%s" % (pre, node.name))
            out_f.write("\n")

