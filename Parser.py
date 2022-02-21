from anytree import Node, RenderTree
import re
import copy

class Parser:
    """
    Basic parser for spliting cpc titles into its taxonomy structure.
    The parser divides a (two-layer) tree into branches.
    """

    def __init__(self, prepositions_file='prep_en.txt'):
        self.prep_file = prepositions_file

        self.prep_list = open(self.prep_file, 'r').read().splitlines()
        
    def split(self, title: str) -> dict:
        res_forest = {}

        # remove "not otherwise provided for" from title
        title = re.sub(r"[ ,;]?not otherwise provided for[,; ]?", "", title)

        # 1. split by semicolon
        titles = title.split(";")
        for t in titles:
            t = re.sub(r" ?in general$", "", t.strip()) # remove "in general" at the end of the title
            res_forest[t] = Node(t)

        # 2. split by e.g.
            if "e.g." in t:
                try:
                    eg_p, eg_c = t.split('e.g.')
                    eg_p = re.sub(r" ?in general$", "", eg_p.strip(" ,"))
                    eg_c_nodes = [eg_c]
                except ValueError: # more than one "e.g." in the title
                    eg_list = t.split('e.g.')
                    eg_p = re.sub(r" ?in general$", "", eg_list[0].strip(" ,")) 
                    
                    second_eg = " ".join([eg_list[1], eg_list[2]]) if eg_list[2].split()[0] in self.prep_list or eg_list[1].split()[-1] in self.prep_list else "e.g.".join([eg_list[1], eg_list[2]])
                    eg_c_nodes = [eg_list[1], second_eg]

                for eg_c in eg_c_nodes:
                    eg_c = re.sub(r" ?in general$", "", eg_c.strip())

                    # if the first word of example or the final word of head is a preprosition, concatename their names
                    if eg_c.split()[0] in self.prep_list or eg_p.split()[-1] in self.prep_list:   
                        child_title = " ".join([t, eg_c])
                    else:
                        child_title = eg_c
                    res_forest[child_title] = Node(child_title, parent = res_forest[t])  
        return res_forest

    def update_child_layer(self, p_node: Node, c_nodes: list, clean_p: bool) -> Node:
        """
        concatenate a parent node and its children nodes (childnode could be a tree depends on the results of splitting)
        """
        if clean_p:
            children_list = []
        else:
            children_list = list(p_node.children)   # original children nodes of the parent node

        for c in c_nodes:
            # if the first word of example or the final word of head is a preprosition, concatename their names
            if c.name.split()[0] in self.prep_list or p_node.name.split()[-1] in self.prep_list:
                c.name = " ".join([p_node.name, c.name])
        children_list.extend(c_nodes)
        p_node.children = children_list
        return p_node


    def dfs_postorder(self, arr: list, root_node: Node) -> list:
        """
        return a list of tree nodes by post order: first leaf nodes then parent node
        """
        children_list =  copy.deepcopy(root_node.children)
        if children_list:
            for c_node in children_list:
                self.dfs_postorder(arr,c_node)
        arr.append(root_node)
        return arr         

    def get_taxonomy(self, root_node: str, root_name: str) -> Node:
        # create new root node 
        res_root = Node(root_name)

        # all nodes in post order 
        nodes_to_search = self.dfs_postorder([], root_node)

        saved_nodes = {}
        top_layer = []

        for node in nodes_to_search:
            p_node = copy.deepcopy(node.parent)
            if p_node:
                c_layer = list(self.split(node.name).values())
                
                # if child node has children nodes already updated
                if node.name + str(node.depth) in saved_nodes:
                    siblings = saved_nodes[node.name + str(node.depth)]
                    for c in c_layer:
                        c.children = copy.deepcopy(siblings.children)  

                # print("parent", p_node)
                # print("children", c_layer)

                if p_node.name + str(p_node.depth) in saved_nodes:
                    new_p = False  
                    p_node = saved_nodes[p_node.name + str(p_node.depth)]
                else:
                    new_p = True
                p_v = self.update_child_layer(p_node, c_layer, clean_p = new_p)

                # update parent node in result dictionary
                saved_nodes[p_node.name+str(p_node.depth)] = p_v

            else:
                top_layer.append(saved_nodes[node.name + str(node.depth)])
            
        res_root.children = copy.deepcopy(top_layer)
        for pre, fill, node in RenderTree(res_root):
            print("%s%s" % (pre, node.name))
        return res_root



# example
parser = Parser()
parser.get_taxonomy(tree_A.root, 'A')