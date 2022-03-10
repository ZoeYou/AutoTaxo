from anytree import RenderTree, PostOrderIter, PreOrderIter
import Parser
import csv
from preprocessing import *

from pathlib import Path
from tqdm import tqdm
import multiprocessing, pickle

CREATE_CSV = True
PRINT_TREE = True
OUTPUT_DIR = "trees"

def save_tree(root_node, file_name):
    file_to_store = open(file_name, "wb")
    pickle.dump(root_node, file_to_store)
    file_to_store.close()

def load_tree(file_name):
    file_to_read = open(file_name, "rb")
    root_node = pickle.load(file_to_read)
    file_to_read.close()
    return root_node

def get_root_node(file):
    name = file.split("_")[0][-1]
    df = read_label_file(file)
    tree = build_tree(df)
    res_root = parser.get_taxonomy(tree.root, name) # could be slow for CPC8
    print(f"{name} done!")
    return res_root, name 


if __name__ == '__main__':
    # read tree files
    output_path = Path(OUTPUT_DIR)
    tree_files = sorted([str(f) for f in output_path.glob("*.pickle")])
    if len(tree_files) == 9:
        dict_trees = {}
        for f in tree_files:
            res_root = load_tree(f)
            dict_trees[str(f).split(".")[0].split("/")[-1]] = res_root
    else:
        files_list = [str(f) for f in Path("cpc-titles").glob("*.txt")]
        parser = Parser.Parser()
        pool = multiprocessing.Pool(36)
        for res_root, name in pool.imap_unordered(get_root_node, files_list):
            save_tree(res_root, output_path / (name + '.pickle'))
            if PRINT_TREE:
                # print taxonomy tree into an text file
                output_file = output_path / (name + '.txt')
                with output_file.open('w') as out_f:
                    for pre, _, node in RenderTree(res_root):
                        out_f.write("%s%s" % (pre, node.name))
                        out_f.write("\n")

    if CREATE_CSV:
        for f in tree_files:
            res_root = load_tree(f)
            dict_trees[str(f).split(".")[0].split("/")[-1]] = res_root

        with open("hH.csv", "w", newline='') as csv_f:
            writer = csv.writer(csv_f, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
            for f, root_node in dict_trees.items():
                # save term-hyponym pairs into csv file
                nodes = [node for node in PreOrderIter(root_node)]
                
                for node in nodes[2:]:
                    writer.writerow([f, node.name, node.parent.name])
            
