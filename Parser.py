from xml.dom.minicompat import NodeList
from anytree import Node, PostOrderIter
from collections import defaultdict
from tqdm import tqdm
import re, copy

class Parser:
    """
    Basic parser for spliting cpc titles into its taxonomy structure.
    """
    def __init__(self, prepositions_file='prep_en.txt'):
        self.prep_file = prepositions_file
        self.prep_list = open(self.prep_file, 'r').read().splitlines()
        self.there_dict = {"thereof": "of",
                           "therefor": "for",
                           "therefore": "for",
                           "therewith": "with"}

    def filter(self, title: str) -> bool:
        if "their" in title:
            return True

        title_words = title.split()
        if "such" in title_words:
            if "such as" in title or "such a" in title:
                return False
            else:
                # check whether the noun modified by "such" has already appeared in the title description
                such_noun = re.search("such (\w)*", title, re.IGNORECASE)
                if such_noun:
                    such_noun = such_noun.group(0)
                    word_after_such = such_noun.split()[-1]
                    if word_after_such in title.replace(such_noun, ""):
                        return False
                    else:
                        return True
                else:
                    return False
        
    def split(self, title: str) -> dict:
        res_forest = {}

        # remove "not otherwise provided for" from title
        title = re.sub(r"[ ,]?not otherwise provided for[, ]?", "", title, flags=re.IGNORECASE)

        # 1. split by semicolon
        titles = title.split(";")
        for t in titles:
            t = re.sub(r"[ ,]?(in general| or the like)$", "", t.strip(", "), flags=re.IGNORECASE).strip(", ") # remove "in general" or "all the like" at the end of the title
            # filter titles having "their" inside
            if self.filter(t):
                continue
            res_forest[t] = Node(t)

        # 2. split by e.g.
            if "e.g." in t:
                try:
                    eg_p, eg_c = t.split('e.g.')
                    eg_p = re.sub(r"[ ,]?(in general| or the like)$", "", eg_p.strip(", "), flags=re.IGNORECASE)
                    eg_c = re.sub(r"[ ,]?(in general| or the like)$", "", eg_c.strip(", "), flags=re.IGNORECASE)
                    res_forest[t] = Node(eg_p)

                    # if the first word of example or the final word of main description is a preprosition, concatename them
                    try:
                        if eg_c.split()[0].lower() in self.prep_list or eg_p.split()[-1].lower() in self.prep_list:   
                            eg_c = " ".join([eg_p, eg_c])
                        else:
                            eg_c = eg_c.capitalize()
                        res_forest[t].children = [Node(eg_c)]
                    except IndexError:
                         continue   # child node without content

                except ValueError: # more than one "e.g." in the title
                    eg_list = [eg.strip(", ") for eg in t.split('e.g.')]
                    eg_list = [re.sub(r"[ ,]?(in general| or the like)$", "", eg, flags=re.IGNORECASE) for eg in eg_list]
                    eg_p = eg_list[0]

                    res_forest[t] = Node(eg_p)
                    Node_list = []                    
                    for i in range(1, len(eg_list))[::-1]:
                        eg_c = eg_list[i] 
                        # if the first word of example or the final word of head is a preprosition, concatename their names
                        if eg_c.split()[0].lower() in self.prep_list or eg_list[i-1].split()[-1].lower() in self.prep_list:   
                            eg_c = " ".join([eg_list[i-1], eg_c])  
                        else:
                            eg_c = eg_c.capitalize()               
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
        3. check if children nodes are ended with "thereof", "therefor"/"therefore" and "therewith" (without "or" also in the title), if true remove them
        """
        p_name = p_node.name
        c_nodes = [c for c in c_nodes if c.name != p_name]

        dict_name_siblings = defaultdict(list)  # values of dictinary are lists of Nodes
        for c in c_nodes:
            dict_name_siblings[c.name].append(c)
        
        # remove duplication children nodes
        for k,v in dict_name_siblings.items():
            head_node = v[0]
            if len(v) > 1:   
                head_children = list(copy.deepcopy(head_node.children))
                for other_node in v[1:]:
                    head_children.extend(list(copy.deepcopy(other_node.children)))
                head_node.children = head_children
            dict_name_siblings[k] = head_node
        c_nodes = list(dict_name_siblings.values())

        # titles end with therefor etc. concatenate its parent node to it (if too long just ignore and remove the last word)
        # nodes_to_remove = []
        for i in range(len(c_nodes)): 
            c = c_nodes[i]
            words_list = c.name.strip().split()          
            last_word = words_list[-1].lower()

            if last_word in self.there_dict.keys() and (not (" or parts thereof" in c.name or " OR PARTS THEREOF" in c.name)):            
                if len(words_list) < 5 and len(p_name.split()) < 5:
                    c.name = " ".join(words_list[:-1] + [self.there_dict[last_word]] + p_name.lower().split())
                else:
                    words_list = words_list[:-1]
                    c.name = " ".join(words_list)

            # if the first character of children node is in lower case, then concatenate with parent node
            if c.name[0].islower(): 
                c.name = " ".join([p_name,c.name])

                c_siblings = list(copy.deepcopy(c.children))
                if c_siblings:
                    for j in range(len(c_siblings)):
                        sib = copy.deepcopy(c_siblings[j])
                        if sib.name[0].islower():                                                
                            sib.parent.name = c.name
                            sib.name = " ".join([p_name, sib.name])
                            c_siblings[j] = sib
                    c.childern = [] #TODO zy: no idea why it works :)
                    c.children = c_siblings
            c_nodes[i] = c
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
            # check if vide name, if yes delete it
            if not c.name:
                c_nodes.remove(c)
        
        children_list.extend(c_nodes)                   
        p_node.children = children_list
        return p_node        

    def get_taxonomy(self, root_node: str, root_name: str) -> Node:
        # create new root node 
        res_root = Node(root_name)

        # all nodes in post order 
        nodes_to_search = [node for node in PostOrderIter(root_node)]
        
        saved_nodes = {}
        top_layer = []
        
        print(f"Creating taxonomy tree for {root_name} ...")
        for node in tqdm(nodes_to_search):
            c_layer = list(self.split(node.name).values()) # returns a list of Nodes after splitting

            # preprocessing for titles with "therefor" "thereof" etc.
            nodes_to_remove = []
            for i in range(len(c_layer)):
                c = c_layer[i]

                words_list = c.name.strip().split()    
                try:      
                    last_word = words_list[-1].lower()
                except IndexError:
                        continue
                if last_word in self.there_dict.keys() and (not (" or parts thereof" in c.name or " OR PARTS THEREOF" in c.name)):
                    if len(c_layer) > 1:
                        # concatenate the content of current node with its previous brother node if possible
                        brother_name = c_layer[i-1].name
                        brother_lasw = brother_name.split()[-1].lower()
                        if len(words_list) < 5 and len(brother_name.split()) < 5 and not (brother_lasw in self.there_dict.keys()):
                            c.name = " ".join(words_list[:-1] + [self.there_dict[last_word]] + brother_name.lower().split())
                            c_layer[i] = c 
                        else:
                            nodes_to_remove.append(c)
            c_layer = [c for c in c_layer if not (c in nodes_to_remove)]
                
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
