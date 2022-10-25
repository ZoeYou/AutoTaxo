from xml.dom.minicompat import NodeList
from anytree import Node, PostOrderIter, PreOrderIter
from collections import defaultdict
from tqdm import tqdm
import re, copy
import spacy

class Parser:
    """
    Basic parser for spliting cpc titles into taxonomy structure.
    """
    def __init__(self, prepositions_file='prep_en.txt', synonym_file='synonyms.txt'):
        self.prep_file = prepositions_file
        self.synonym_outpath = synonym_file
        self.prep_list = open(self.prep_file, 'r').read().splitlines()
        self.there_dict = {"thereof": "of", "therefor": "for", "therefore": "for", "therewith": "with", "therein": "in"}
        # self.pos_tagger = spacy.load("en_core_web_sm", disable=["textcat", "ner", "lemmatizer"])

    def _filter(self, title: str) -> bool:
        if "their" in title:
            return True
        if "such as " in title or "such a " in title:
            return False
        if " such " in title:
            # check whether the noun modified by "such" has already appeared in the title description
            such_noun = re.search("such (\w*)", title, re.IGNORECASE)
            if such_noun:
                word_after_such  = such_noun.group(1) 
                if word_after_such in title.replace(word_after_such, ""):
                    return False
                else:
                    return True
            else:
                return False    # TODO

    def _get_syns_in_brackets(self, title: str) -> str:
        """
        Remove "[" "]"" from title and save synonyms (e.g. Support Vector Machine [SVM]) 

        """
        in_brackets = re.finditer(r"(\[\S*\]),?;?", title)
        if in_brackets:
            with open(self.synonym_outpath, "a") as out_f:
                out_f.write(title + "\t" + in_brackets[0].group(1).strip(" ,") + "\n")
                title_temp = title[:in_brackets[0].start()]
                for i in range(len(in_brackets)-1):
                    out_f.write(title + "\t" + in_brackets[i+1].group(1).strip(" ,") + "\n")
                    title_temp += title[in_brackets[i].end()+1:in_brackets[i+1].start()]
        
            title = title_temp
        return title
    
    def _get_syns_by_ie(self, title: str) -> str:
        starts_by_ie = re.finditer(r"i\.e\.?\s([^;,]*)", title)
        if starts_by_ie:
            with open(self.synonym_outpath, "a") as out_f:
                out_f.write(title + "\t" + starts_by_ie[0].group(1).strip(" ,") + "\n")
                title_temp = title[:starts_by_ie[0].start()]

                for i in range(len(starts_by_ie)-1):
                    out_f.write(title + "\t" + starts_by_ie[i+1].group(1).strip(" ,") + "\n")
                    title_temp += title[starts_by_ie[i].end()+1:starts_by_ie[i+1].start()]

            title = title_temp
        return title

    def _attach_to_parent(self, parent_title: str, child_title: str, child_fst_word: str, for_eg: bool = True) -> str:
        parent_words = parent_title.lower().split()

        cond = child_fst_word in parent_words
        if for_eg:
            cond = (child_fst_word in self.prep_list) and cond

        if cond:
            # find the position of the first word of child node in parent node
            pos = parent_words[::-1].index(child_fst_word)
            # remove tail part of parent node starting with the same preprosition
            parent_words = parent_words[:-(pos+1)]
            child_title = " ".join(parent_words + [child_title])
        else:
            child_title = " ".join([parent_title, child_title])
        return child_title

    def _split_eg(self, title: str, substitution_patterns: str) -> Node:
        try:
            eg_p, eg_c = title.split('e.g.', flags=re.IGNORECASE)
            eg_p = re.sub(substitution_patterns, "", eg_p.strip(", "), flags=re.IGNORECASE)
            eg_c = re.sub(substitution_patterns, "", eg_c.strip(", "), flags=re.IGNORECASE)
            eg_p_node = Node(eg_p)
            # if the first word of example or the final word of main description is a preprosition, concatenate them for the child node
            try:
                fst_word = eg_c.split()[0].lower()
                if fst_word in self.prep_list or eg_p.split()[-1].lower() in self.prep_list: 
                    eg_c = self._attach_to_parent(eg_p, eg_c, fst_word)
                else:
                    eg_c = eg_c.capitalize()
                eg_p_node.children = [Node(eg_c)]
            except IndexError:
                    pass   # child node without content

        except ValueError: # more than one "e.g." in the title
            eg_list = [eg.strip(", ") for eg in title.split('e.g.')]
            eg_list = [re.sub(substitution_patterns, "", eg, flags=re.IGNORECASE) for eg in eg_list]
            eg_p = eg_list[0]
            eg_p_node = Node(eg_p)

            Node_list = []                    
            for i in range(1, len(eg_list))[::-1]:
                eg_c = eg_list[i] 

                # if the first word of example or the final word of head is a preprosition, concatename their names for child node
                fst_word = eg_c.split()[0].lower()
                if fst_word in self.prep_list or eg_list[i-1].split()[-1].lower() in self.prep_list:
                    eg_c = self._attach_to_parent(eg_list[i-1], eg_c, fst_word)
                else:
                    eg_c = eg_c.capitalize() 

                node_to_add = Node(eg_c)
                try:
                    node_to_add.children = [Node_list[-1]]
                except IndexError:
                    pass
                Node_list.append(node_to_add)
            eg_p_node.children = [Node_list[-1]]
        return eg_p_node

    def _split(self, title: str) -> dict:
        res_forest = {}

        # remove "reference" patterns from title
        title = re.sub(r"[ ,]?(not otherwise provided for|general aspects or details|(referred to|as defined|covered) in the subgroups)[, ]?", "", title, flags=re.IGNORECASE)

        # remove abbreviations inside "[" and "]"
        title = self._get_syns_in_brackets(title)
        # remove content after i.e.
        title = self._get_syns_by_ie(title)

        # 1. split by semicolon 
        titles = title.split(";")
        if len(titles) == 1:
            pass # TODO or split by parallel structure (detected by pos tagging)
            
        patterns_to_remove = r"[ ,]?((in general|or the like|or other parts|not provided for elsewhere|specified in the subgroup)$|other than .*)"
        patterns_to_remove_head = r"^Other "

        for t in titles:
            t = re.sub(patterns_to_remove, "", t.strip(", "), flags=re.IGNORECASE).strip(", ") # remove "in general" or "all the like", etc. at the end of the title
            if re.match(patterns_to_remove_head, t):
                t = re.sub(patterns_to_remove_head,"",t).capitalize() # remove "Other" at the beginning of the title
            # remove titles with Pronouns like "such", "their"
            if self._filter(t):
                continue
            res_forest[t] = Node(t)

        # 2. split by "such as"
            if " such as " in t:
                sa_p, sa_c = re.split(" such as ", t, maxsplit=1, flags=re.IGNORECASE)
                sa_p = re.sub(patterns_to_remove, "", sa_p.strip(", "), flags=re.IGNORECASE)
                sa_c = re.sub(patterns_to_remove, "", sa_c.strip(", "), flags=re.IGNORECASE)
                sa_p_node = Node(sa_p)
                if "e.g." in sa_p_node.name:
                    sa_p_node = self._split_eg(sa_p_node.name, patterns_to_remove)
                # if the first word of example or the final word of main description is a preprosition, concatename them
                try:
                    fst_word = sa_c.split()[0].lower()
                    if fst_word in self.prep_list or sa_p.split()[-1].lower() in self.prep_list: 
                        sa_c = self._attach_to_parent(sa_p, sa_c, fst_word)
                    else:
                        sa_c = sa_c.capitalize()

                    # check if it has "e.g." in the title of child node
                    sa_c_node = Node(sa_c)
                    if "e.g." in sa_c_node.name:
                        sa_c_node = copy.deepcopy(self._split_eg(sa_c_node.name, patterns_to_remove))
                    sa_p_node.children = list(copy.deepcopy(sa_p_node.children)) + [sa_c_node]
                    res_forest[t] = sa_p_node
                except IndexError:
                        pass   # child node with empty content            

        # 3. split by "e.g." without "such as"
            if "e.g." in t:
                res_forest[t] = self._split_eg(t, patterns_to_remove)         
        return res_forest

    def _normalise(self, p_node: str, c_nodes: list) -> list:
        """
        1. check whether children nodes have the same title as parent title, if true remove the title in children nodes
        2. check whether children nodes have repeated titles among each other, if true combine their descendants nodes
        3. check if children nodes are ended with "thereof", "therefor"/"therefore" and "therewith" (without "or" also in the title), if true remove them   # TODO
        """
        p_name = p_node.name
        c_nodes = [c for c in c_nodes if c.name != p_name]

        dict_name_descendants = defaultdict(list)  # values of dictionary are lists of Nodes
        for c in c_nodes:
            dict_name_descendants[c.name].append(c)
        
        # remove duplicated children nodes
        for k,v in dict_name_descendants.items():
            head_node = v[0]
            if len(v) > 1:   
                head_children = list(copy.deepcopy(head_node.children))
                for other_node in v[1:]:
                    head_children.extend(list(copy.deepcopy(other_node.children)))
                head_node.children = head_children
            dict_name_descendants[k] = head_node
        c_nodes = list(dict_name_descendants.values())

        # titles end with therefor etc. concatenate its parent node to it (if too long just ignore and remove the last word)
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
                c.name = self._attach_to_parent(p_name, c.name, c.name.split()[0], for_eg= False)

                c_descendants = list(copy.deepcopy(c.children))
                if c_descendants:
                    for j in range(len(c_descendants)):
                        des = copy.deepcopy(c_descendants[j])
                        if des.name[0].islower():                                                
                            des.parent.name = c.name
                            des.name = self._attach_to_parent(p_name, des.name, des.name.split()[0], for_eg= False)
                            c_descendants[j] = des
                    c.children = c_descendants
            c_nodes[i] = c
        return c_nodes

    def _update_child_layer(self, p_node: Node, c_nodes: list, sterilized_p: bool) -> Node:
        """
        concatenate a parent node and its children nodes (childnode could be a tree depends on the results of splitting)
        """
        if sterilized_p:
            children_list = []
        else:
            children_list = list(p_node.children)   # original children nodes of the parent node
        for c in c_nodes:
            # delete vide node
            if not c.name: c_nodes.remove(c)
        
        children_list.extend(c_nodes)                   
        p_node.children = children_list
        return p_node 

    def _valide(self, root_node: str) -> Node:
        # all nodes in pre order for validation
        nodes_to_search = [node for node in PreOrderIter(root_node)][1:]
        for node in nodes_to_search:
            # check if node starts with lower case
            fst_word = node.name.split()[0]
            try:
                if node.name[0].islower() and fst_word in node.parent.name and fst_word != node.parent.name.lower().split()[0]:
                    node.name = self._attach_to_parent(node.parent.name, node.name, fst_word, for_eg = False)
            except AttributeError:  #  AttributeError: 'NoneType' object has no attribute 'name' (node.paren does not exist?)
                continue

            # check if there is words such as "thereof", "therefor", "therewith" etc. inside the titles 
            #TODO

            # check if there is "such as" in the title
            if " such as " in node.name.lower():
                sa_p, sa_c = re.split(" such as ", node.name, 1, flags=re.IGNORECASE)
                sa_p =  sa_p.strip(", ")
                sa_c =  sa_c.strip(", ")
                node.name = sa_p
                node.children = list(copy.deepcopy(node.children)) + [Node(sa_c)]    

            # check if node starts with "Details", lift the current node up one level if so
            if node.name[:7].lower() == "details":
                try:
                    if node.children:
                        node.parent.children = node.children
                        for c_node in node.children:
                            c_node.parent = node.parent
                    else:
                        node.parent.children = []
                except Exception as e:
                    print(e)
                    #print("error with it: ", node) #TODO

            # remove "[" at the end of title (happens in domain Y) or "[" at the end of title but without "[" before
            node.name = node.name.rstrip("[ ")
            if node.name[-1] == "]" and "[" not in node.name:
                node.name = node.name.rstrip("] ")
        return root_node        

    def get_taxonomy(self, root_node: Node, root_name: str) -> Node:
        # create new root node 
        res_root = Node(root_name)

        # all nodes in post order ()
        nodes_to_search = [node for node in PostOrderIter(root_node)]
        
        saved_nodes = {}
        top_layer = []
        
        print(f"Creating taxonomy tree for {root_name} ...")
        for node in tqdm(nodes_to_search):
            c_layer = list(self._split(node.name).values()) # returns a list of Nodes after splitting

            # preprocessing for titles with "therefor" "thereof" etc.
            nodes_to_remove = []
            for i in range(len(c_layer)):
                c = c_layer[i]

                words_list = c.name.strip().split()    
                try:      
                    last_word = words_list[-1].lower()
                except IndexError:
                        continue
                if last_word in self.there_dict and (not (" or parts thereof" in c.name or " OR PARTS THEREOF" in c.name)):
                    if len(c_layer) > 1:
                        # concatenate the content of current node with its previous brother node if possible
                        brother_name = c_layer[i-1].name
                        if brother_name:
                            brother_lasw = brother_name.split()[-1].lower()

                            if len(words_list) < 5 and len(brother_name.split()) < 5 and not (brother_lasw in self.there_dict.keys()):
                                c.name = " ".join(words_list[:-1] + [self.there_dict[last_word]] + brother_name.lower().split())
                                c_layer[i] = c 
                            else:
                                nodes_to_remove.append(c)
            c_layer = [c for c in c_layer if not (c in nodes_to_remove)]
                
            # if current node has children nodes already updated
            if node.name + str(node.depth) in saved_nodes:
                descendants = saved_nodes[node.name + str(node.depth)] # a list of children nodes of current node
                for c in c_layer:
                    c.children = self._normalise(c, copy.deepcopy(c.children) + copy.deepcopy(descendants.children))

            p_node = copy.deepcopy(node.parent)
            if p_node:
                if p_node.name + str(p_node.depth) in saved_nodes:
                    new_p = False  
                    p_node = saved_nodes[p_node.name + str(p_node.depth)]
                else:
                    new_p = True
                p_v = self._update_child_layer(p_node, c_layer, sterilized_p = new_p)

                # update parent node in result dictionary
                saved_nodes[p_node.name+str(p_node.depth)] = p_v
            else:
                top_layer.extend(c_layer)
            
        res_root.children = copy.deepcopy(top_layer)

        # to check last time the titles connatenated in pre-order iteration
        res_root = self._valide(res_root)
        return res_root
