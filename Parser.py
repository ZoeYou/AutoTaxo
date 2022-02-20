import re

class Parser:
    """
    Basic parser for spliting cpc titles into its taxonomy structure.
    The parser divides a (two-layer) tree into branches.
    """

    def __init__(self, prepositions_file='prep_en.txt', do_lower_case=True):
        self.do_lower_case = do_lower_case
        self.prep_file = prepositions_file

        self.prep_list = open(self.prep_file, 'r').read().splitlines()
        
    def split(self, title):
        res_forest = {}

        if self.do_lower_case:
            title = title.lower()

        # 1. split by semicolon
        titles = title.split(";")
        for t in titles:
            t = re.sub(r" ?in general$", "", t.strip()) # remove "in general" at the end of the title
            res_forest[t] = Node(t)

        # 2. split by e.g.
            if "e.g." in t:
                eg_p, eg_c = t.split('e.g.')
                eg_p = re.sub(r" ?in general$", "", eg_p.strip(" ,"))
                eg_c_nodes = eg_c.split(";")

                for eg_c in eg_c_nodes:
                    eg_c = re.sub(r" ?in general$", "", eg_c.strip())
                    if eg_c.split()[0] in self.prep_list:
                        child_title = " ".join([t, eg_c])
                    else:
                        child_title = eg_c
                    res_forest[child_title] = Node(child_title, parent = res_forest[t])  
        return res_forest  

    def connect_layer(p_tree, c_tree):
        pass


    def get_taxo(self, p_node, c_nodes):
        res_forest = {}

        # split parent node
        p_layer = self.split(p_node)
        c_layer = [self.split(n) for n in c_nodes]

        print(p_layer)

        for p_k, p_v in p_layer.items():
            res_forest[p_k] = p_v
            
            for c_tree in c_layer:
                for c_k, c_v in c_tree.items():
                    c_v.parent = res_forest[p_k]


                    node_dict[child_title] = Node(child_desc, parent = node_dict[father_title])
        print(res_forest)