# script used to seperate entities with descriptive contents
from pathlib import Path
import pickle, re, copy, sys
from anytree import RenderTree, PostOrderIter
from count_freqs import * 


def save_tree(root_node, file_name):
    with open(file_name, "wb") as outf:
        pickle.dump(root_node, outf)

def load_tree(file_name):
    with open(file_name, "rb") as inf:
        root_node = pickle.load(inf)
    return root_node

def is_leaf_node(node) -> bool:
    if not node.name:   return False
    if node.children:
        for child in node.children:
            return is_leaf_node(child)
    else:
        return True

def split_leaf_node(node) -> list:
    """
    split leaf nodes if they have a parallel structure (e.g. Klystrons, travalling-wave tubes, magnetronts)

    """
    #TODO 




if __name__ == '__main__':
    preposition_list = open('./prep_en.txt','w').read().splitlines()
    descriptive_patterns = r'accessories|arrangements?|applications?|apparatus?|appliances?|^methods?|methods or|methods for| methods|details|means for|devices? | devices?|^tools?|or methods?|or implements?|^machines for|machines or|instruments for|implements for|equipments? for|specially adapted|characterised by|^types? of|^special|systems using|instruments? employing|^measurement? of|therefor|thereof|therewith|thereby| designed for |^treatment |aspects of|particular use of|general design| them | their|^preparation|mechanisms?|^Processes for'

    input_path = Path(sys.argv[1])
    tree_files = sorted([str(f) for f in input_path.glob("*.pickle")])

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
        output_tree_ents = input_path / (tree_entities.name + '_entities.pickle')
        output_tree_desc = input_path / (tree_entities.name + '_desc.pickle')

        # print separated taxonomies in text files
        with output_file_ents.open('w') as out_f:
            list_entities = [node.name for _,_,node in RenderTree(tree_entities) if node.name]
            freq_entities = freqs_solr(list_entities)
            for pre, _, node in RenderTree(tree_entities):
                out_f.write("%s%s\t%s" % (pre, node.name, freq_entities[node.name]))
                out_f.write("\n")

        with output_file_desc.open('w') as out_f:
            list_desc = [node.name for _,_,node in RenderTree(tree_descriptions) if node.name]
            freq_desc = freqs_solr(list_desc)
            for pre, _, node in RenderTree(tree_descriptions):
                out_f.write("%s%s\t%s" % (pre, node.name, freq_desc[node.name]))
                out_f.write("\n")

        # save separated taxonomies as tree object
        save_tree(tree_entities, output_tree_ents)
        save_tree(tree_descriptions, output_tree_desc)

