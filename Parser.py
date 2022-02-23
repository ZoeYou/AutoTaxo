from anytree import Node, RenderTree
from collections import defaultdict
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
        title = re.sub(r"[ ,]?not otherwise provided for[, ]?", "", title)

        # 1. split by semicolon
        titles = title.split(";")
        for t in titles:
            t = re.sub(r"[ ,]?in general$", "", t.strip(", ")).strip(", ") # remove "in general" at the end of the title
            res_forest[t] = Node(t)

        # 2. split by e.g.
            if "e.g." in t:
                try:
                    eg_p, eg_c = t.split('e.g.')
                    eg_p = re.sub(r"[ ,]?in general$", "", eg_p.strip(", "))
                    eg_c = re.sub(r"[ ,]?in general$", "", eg_c.strip(", "))
                    res_forest[t] = Node(eg_p)

                    # if the first word of example or the final word of head is a preprosition, concatename their names
                    try:
                        if eg_c.split()[0] in self.prep_list or eg_p.split()[-1] in self.prep_list:   
                            eg_c = " ".join([eg_p, eg_c])
                        res_forest[t].children = [Node(eg_c)]
                    except IndexError:
                         continue   # child node no content

                except ValueError: # more than one "e.g." in the title
                    eg_list = [eg.strip(", ") for eg in t.split('e.g.')]
                    eg_list = [re.sub(r"[ ,]?in general$", "", eg) for eg in eg_list]
                    eg_p = eg_list[0]

                    res_forest[t] = Node(eg_p)
                    Node_list = []                    
                    for i in range(1, len(eg_list))[::-1]:
                        eg_c = eg_list[i] 
                        # if the first word of example or the final word of head is a preprosition, concatename their names
                        if eg_c.split()[0] in self.prep_list or eg_list[i-1].split()[-1] in self.prep_list:   
                            eg_c = " ".join([eg_list[i-1], eg_c])                 
                        node_2_add = Node(eg_c)
                        try:
                            node_2_add.children = [Node_list[-1]]
                        except IndexError:
                            pass
                        Node_list.append(node_2_add)
                    res_forest[t].children = [Node_list[-1]]
        return res_forest

    def valide(self, p_node: str, c_nodes: list) -> list:
        """
        1. check whether children nodes have the same title as parent title, if true remove the title in children nodes
        2. check whether children nodes have repeated titles among each other, if true combine their sibling nodes 
        """
        p_name = p_node.name
        c_nodes = [c for c in c_nodes if c.name != p_name]

        dict_name_siblings = defaultdict(list)  # values of dictinary are lists of Nodes
        #nodes_name = [c.name for c in c_nodes]

        for c in c_nodes:
            dict_name_siblings[c.name].append(c)
        flag = False
        for k,v in dict_name_siblings.items():
            if len(v) > 1:
                flag = True
        if flag:
            for k, v in dict_name_siblings.items():
                print(k)
                print(v)
                print("\n")
            asdf
        return c_nodes
    

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
            try:
                if c.name.split()[0] in self.prep_list: # or p_node.name.split()[-1] in self.prep_list TODO
                    c.name = " ".join([p_node.name, c.name])
            except IndexError:
                c_nodes.remove(c)
        
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
            c_layer = list(self.split(node.name).values())  # returns a list of Nodes after splitting
                
            # if current node has children nodes already updated
            if node.name + str(node.depth) in saved_nodes:
                siblings = saved_nodes[node.name + str(node.depth)] # a list of children nodes of current node
                for c in c_layer:
                    c.children = self.valide(c, copy.deepcopy(c.children) + copy.deepcopy(siblings.children))

            p_node = copy.deepcopy(node.parent)
            if p_node:
                if p_node.name + str(p_node.depth) in saved_nodes:
                    new_p = False  
                    p_node = saved_nodes[p_node.name + str(p_node.depth)]
                else:
                    new_p = True
                p_v = self.update_child_layer(p_node, c_layer, clean_p = new_p)

                # update parent node in result dictionary
                saved_nodes[p_node.name+str(p_node.depth)] = p_v
            else:
                top_layer.extend(c_layer)
                #top_layer.append(saved_nodes[node.name + str(node.depth)])
            
        res_root.children = copy.deepcopy(top_layer)
        return res_root
