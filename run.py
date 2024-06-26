from anytree import RenderTree, PostOrderIter, PreOrderIter
import Parser
from preprocessing import *

from pathlib import Path
from tqdm import tqdm
import multiprocessing, pickle
import pandas as pd

CREATE_CSV = True   # create term-hypernym pairs based on taxonomy => save in csv file
PRINT_TREE = True
OUTPUT_DIR = f"trees_CPC{TARGET_LEVEL}-{MAX_DEPTH}"

def save_tree(root_node, file_name):
    with open(file_name, "wb") as outf:
        pickle.dump(root_node, outf)

def load_tree(file_name):
    with open(file_name, "rb") as inf:
        root_node = pickle.load(inf)
    return root_node

def get_root_node(file):
    name = file.split("_")[0][-1]
    df = read_label_file(file)
    tree = build_tree(df)
    res_root = parser.get_taxonomy(tree.root, name) # could be slow for deep layers...
    print(f"{name} done!")
    return res_root, name 



if __name__ == '__main__':
    # create output directory if not exists
    output_path = Path(OUTPUT_DIR) 
    output_path.mkdir(parents=True, exist_ok=True)

    # get all created taxonomy trees (pickle files) if exists
    tree_files = sorted([str(f) for f in output_path.glob("*.pickle")])

    dict_trees = {}
    if len(tree_files) != 9:  
        files_list = [str(f) for f in Path("cpc-titles").glob("*.txt")]
        already_done = [f"cpc-titles/cpc-section-{str(f).split('/')[1].split('.')[0]}_20240101.txt" for f in Path(OUTPUT_DIR).glob("*.txt")]
        files_list = list(set(files_list) - set(already_done))

        parser = Parser.Parser(output_dir = output_path)

        if len(files_list):
            pool = multiprocessing.Pool(len(files_list))
            
            for res_root, name in pool.imap_unordered(get_root_node, files_list):
                save_tree(res_root, output_path / (name + '.pickle'))
                if PRINT_TREE:
                    # print taxonomy tree in an text file
                    output_file = output_path / (name + '.txt')
                    with output_file.open('w') as out_f:
                        for pre, _, node in RenderTree(res_root):
                            out_f.write("%s%s" % (pre, node.name))
                            out_f.write("\n")
            pool.close()
            pool.join()

    if CREATE_CSV:  # save term-hyponym pairs into csv file
        tree_files = sorted([str(f) for f in output_path.glob("*.pickle")])
        for f in tree_files:
            res_root = load_tree(f)
            dict_trees[str(f).split(".")[0].split("/")[-1]] = res_root

        pairs = []
        print("Saving term-hypernym pairs into tsv file...")
        for f, root_node in tqdm(dict_trees.items()):
            # in pre-order iteration of tree
            nodes = [node for node in PreOrderIter(root_node)]
            for node in nodes[2:]:
                pairs.append([f, node.name, node.parent.name])

        df = pd.DataFrame(pairs, columns=["file", "term", "hypernym"])
        df.to_csv(output_path / 'hHs.tsv', header=False, index=False, sep="\t")
