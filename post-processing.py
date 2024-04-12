# script used to seperate entities with descriptive contents
from pathlib import Path
import pickle, re, copy, sys, requests
from anytree import Node, RenderTree, PostOrderIter
from tqdm import tqdm
# from count_freqs import * 
from spacy.tokenizer import Tokenizer

import csv

import spacy
nlp = spacy.load("en_core_web_sm", disable=['ner', 'lemmatizer', 'textcat'])


CREATE_CSV = True   # create term-hypernym pairs based on taxonomy => save in csv file



def custom_tokenizer(nlp):
    inf = list(nlp.Defaults.infixes)               # Default infixes
    inf.remove(r"(?<=[0-9])[+\-\*^](?=[0-9-])")    # Remove the generic op between numbers or between a number and a -
    inf = tuple(inf)                               # Convert inf to tuple
    infixes = inf + tuple([r"(?<=[0-9])[+*^](?=[0-9-])", r"(?<=[0-9])-(?=-)"])  # Add the removed rule after subtracting (?<=[0-9])-(?=[0-9]) pattern
    infixes = [x for x in infixes if '-|–|—|--|---|——|~' not in x] # Remove - between letters rule
    infix_re = spacy.util.compile_infix_regex(infixes)

    return Tokenizer(nlp.vocab, prefix_search=nlp.tokenizer.prefix_search,
                                suffix_search=nlp.tokenizer.suffix_search,
                                infix_finditer=infix_re.finditer,
                                token_match=nlp.tokenizer.token_match,
                                rules=nlp.Defaults.tokenizer_exceptions)

nlp.tokenizer = custom_tokenizer(nlp)


def save_tree(root_node, file_name):
    with open(file_name, "wb") as outf:
        pickle.dump(root_node, outf)

def load_tree(file_name):
    with open(file_name, "rb") as inf:
        root_node = pickle.load(inf)
    return root_node

def is_leaf_node(node) -> bool:
    if not node.name:
        return False

    if node.children:
        if any([is_leaf_node(child) for child in node.children]):
            return False
        else:
            return True
    else:
        return True

def split_leaf_node(leaf: Node) -> Node:
    """
    split leaf nodes if they have a parallel structure (e.g. Klystrons, travalling-wave tubes, magnetronts)

    """
    doc = nlp(leaf.name)
    list_cc = [",", "or", "OR"]

    list_tokens = [token.text for token in doc]

    if (',' in list_tokens) or ('or' in list_tokens) or ('OR' in list_tokens):
        #######################  ENGLISH  #######################
        list_id_cc = [tk.i for tk in doc if tk.text in list_cc]
        try:
            # print("ORIG :", doc.text)
            left = doc[:list_id_cc[0]-1].text
            # print("LEFT : ",left)
            right = doc[list_id_cc[-1]+2:].text
            # print("RIGHT : ",right)
            middle = doc[list_id_cc[0]-1:list_id_cc[-1]+2].text
            # print("MID : ",middle)
            list_term_alt = middle.replace(", ", "€€€").replace(" or ", "€€€").replace(" OR ", "€€€").split("€€€")
            # print(list_term_alt)
            list_term_alt = [t.replace('or of ', '').replace('or ', '') for t in list_term_alt if (t and t!="etc")]
            # print("-------------------")
        except IndexError:
            print("Error in splitting leaf node", leaf.name)
            return leaf

        for en in list_term_alt:
            Node(left+" "+en+" "+right, parent=leaf)
        return leaf



if __name__ == '__main__':
    preposition_list = open('./prep_en.txt','r').read().splitlines()
    # descriptive_patterns = r'accessories|arrangements?|applications?|apparatus?|appliances?|^methods?|methods or|methods for| methods|details|means for|devices? | devices?|^tools?|or methods?|or implements?|^machines for|machines or|instruments for|implements for|equipments? for|specially adapted|characterised by|^types? of|^special|systems using|instruments? employing|^measurement? of|therefor|thereof|therewith|thereby| designed for |^treatment |aspects of|particular use of|general design| them | their|^preparation|mechanisms?|^Processes for'
    descriptive_patterns = r'^methods?|methods or|methods for| methods|details|means for|or methods?|or implements?|^machines for|machines or|instruments for|implements for|equipments? for|specially adapted|characterised by|^types? of|^special|^measurement? of|therefor|thereof|therewith|thereby| designed for |^treatment |aspects of|particular use of|general design| them | their|^preparation|^Processes for'
    
    input_path = Path(sys.argv[1])
    tree_files = sorted([str(f) for f in input_path.glob("*.pickle")])

    if CREATE_CSV:
        # remove existing csv file
        try:
            Path(input_path / 'hHs_ents.csv').unlink()
            Path(input_path / 'hHs_desc.csv').unlink()
        except FileNotFoundError:
            pass
    
    for file in tree_files:
        tree_entities = copy.deepcopy(load_tree(file))
        tree_descriptions = copy.deepcopy(load_tree(file))

        nodes_entity_todo = [node for node in PostOrderIter(tree_entities)]
        nodes_desc_todo = [node for node in PostOrderIter(tree_descriptions)]
        
        for node in tqdm(nodes_entity_todo, desc=f"Processing entities in {file}"):
            if '@@@' in node.name or re.search(descriptive_patterns, node.name.lower(), re.IGNORECASE) or len([word for word in node.name.lower().split() if word in preposition_list])>2 or len(node.name.split())>5:
                node.name = ''
            if is_leaf_node(node):  node = split_leaf_node(node)

        for node in tqdm(nodes_desc_todo, desc=f"Processing descriptions in {file}"):
            if not ('@@@' in node.name or re.search(descriptive_patterns, node.name.lower(), re.IGNORECASE) or len([word for word in node.name.lower().split() if word in preposition_list])>2 or len(node.name.split())>5):
                 node.name = ''
            if is_leaf_node(node):  node = split_leaf_node(node)

        # save updated tree in a new file
        output_file_ents = input_path / (tree_entities.name + '_entities.txt')
        output_file_desc = input_path / (tree_entities.name + '_desc.txt')
        output_tree_ents = input_path / (tree_entities.name + '_entities.pickle')
        output_tree_desc = input_path / (tree_entities.name + '_desc.pickle')

        # print separated taxonomies in text files
        with output_file_ents.open('w') as out_f:
            list_entities = [node.name for _,_,node in RenderTree(tree_entities) if node.name]
            # freq_entities = freqs_solr(list_entities)
            for pre, _, node in RenderTree(tree_entities):
                # out_f.write("%s%s\t%s" % (pre, node.name, "\t".join(freq_entities.get(node.name, ("0","0")))))
                out_f.write("%s\t%s" % (pre, node.name))#, freq_entities[node.name]))
                out_f.write("\n")

        with output_file_desc.open('w') as out_f:
            list_desc = [node.name for _,_,node in RenderTree(tree_descriptions) if node.name]
            # freq_desc = freqs_solr(list_desc)
            for pre, _, node in RenderTree(tree_descriptions):
                # out_f.write("%s%s\t%s" % (pre, node.name, "\t".join(freq_desc.get(node.name, ("0","0")))))
                out_f.write("%s\t%s" % (pre, node.name))#, freq_desc[node.name]))
                out_f.write("\n")        

        # # save separated taxonomies as tree object
        # save_tree(tree_entities, output_tree_ents)
        # save_tree(tree_descriptions, output_tree_desc)

        if CREATE_CSV:  # save term-hyponym pairs content into csv file
            with open(input_path / 'hHs_ents.csv', "a", newline='') as csv_f:
                writer = csv.writer(csv_f, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
                f = file.split("/")[-1].split(".")[0]
                # in pre-order iteration of tree
                nodes = [node for node in PostOrderIter(tree_entities)]
                for node in nodes[2:]:
                    if node and len(node.name.strip())>0:
                        # find parent that are not empty
                        parent_node = node.parent
                        while parent_node and not parent_node.name:
                            parent_node = parent_node.parent
                        if parent_node and (parent_node.name not in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'Y']):
                            writer.writerow([f, node.name.replace("@@@", "").strip(" ."), parent_node.name.replace("@@@", "").strip(" .")])

            with open(input_path / 'hHs_desc.csv', "a", newline='') as csv_f:
                writer = csv.writer(csv_f, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
                f = file.split("/")[-1].split(".")[0]
                # in pre-order iteration of tree
                nodes = [node for node in PostOrderIter(tree_descriptions)]
                for node in nodes[2:]:
                    if node and len(node.name.strip())>0:
                        # find parent that are not empty
                        parent_node = node.parent
                        while parent_node and not parent_node.name:
                            parent_node = parent_node.parent
                        if parent_node and (parent_node.name not in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'Y']):
                            writer.writerow([f, node.name.replace("@@@", "").strip(" ."), parent_node.name.replace("@@@", "").strip(" .")])
        
        print(f"Done processing {file}!")