from anytree import Node, RenderTree
import re
import pandas as pd

global TARGET_LEVEL, MAX_DEPTH
TARGET_LEVEL, MAX_DEPTH = 8, 2

global dict_lvl
dict_lvl = {1: -3, 3: -2, 4: -1, 6: 0}



def get_level(x):
    """
    get level for each title (finer level of class will get higher level)
    """
    lvl = x['lvl']
    try:
        return int(lvl)
    except ValueError:
        code = x['code']
        return dict_lvl[len(code)]

def clean_descr(description):
    """
    1. remove contents in the brakets with CPC codes from description, such as *** (preserving A23B; obtaining protein compositions for foodstuffs A23J1/00;)
    2. remove '{' and '}'
    """
    description = re.sub(r'\ ?\([\w\W]*([A-Z]{1}[0-9]{2}[A-Z]{1}[0-9]*[\/]*[0-9]*)*[\w\W]*\)', '', description)
    description = description.replace('{', '').replace('}','')
    description = re.sub('as specified in the subgroups? and ','', description)
    return description

def has_cpc(description):
    """
    4. check whether a title desciption has CPC codes
    e.g. having alternatively specified atoms bound to the phosphorus atom and not covered by a single one of groups A01N57/10, A01N57/18, A01N57/26, A01N57/34
    """
    cpc_pattern = r'[A-Za-z]{1}[0-9]{2}[A-Za-z]{0,1}[0-9]*[\/]*[0-9]*'
    match = re.search(cpc_pattern, description)
    if match:
        return True
    else:
        return False

def is_trash_title(description):
    """
    5. check whether a title description is a trash title (such as a complement class to other classes)
    """
    description = description.lower()
    if ("subject matter not" in description and "provided for in" in description) or ("covered by" in description and (" subclass" in description or " group" in description)) or ("dummy group" in description):
        return True
    else:
        return False

def next_same_lvl_index(subdf,lvl):
    """
    find index before the next parent title in the given dataframe
    """
    lvl_df = subdf[subdf['lvl'] <= lvl]
    idx_df = lvl_df.index

    if len(idx_df) == 1:
        return idx_df[0]

    res_index = idx_df[1] - 1 
    return res_index

def rm_title_with_subtree(dataframe):
    """
    6. remove those titles with codes in their description and its subtree titles
    """
    # get indices of description to remove 
    dataframe = dataframe.reset_index(drop=True)
    id_hascpc = dataframe[dataframe['description'].apply(has_cpc) | dataframe['description'].apply(is_trash_title)].index
    
    idx_to_drop = []    # to include also the subtitles of titles to remove
    for i in id_hascpc:
        l = dataframe.iloc[i]

        # if it is already the last line or the next node is a parent node
        if i == (dataframe.shape[0]-1) or l['lvl'] >= dataframe.iloc[i+1]['lvl']: 
            idx_to_drop.append(i)
        else:
            j = next_same_lvl_index(dataframe[i:], l['lvl'])
            if i == j:
                idx_to_drop.append(i)
            elif j > i:
                idx_to_drop.extend(range(i,j+1))
            else:
                raise ValueError("Problem with subtree index!")

    dataframe = dataframe.drop(idx_to_drop)
    return dataframe.reset_index(drop=True)

def rm_Details(dataframe):
    """
    7. Remove content of titles starts with "Details", and upgrade the level of its children titles
    """
    dataframe = dataframe.reset_index(drop=True)
    id_Details = dataframe[dataframe['description'].apply(lambda x: "details" == x[:7].lower())].index

    idx_to_drop = []    # to include also the subtitles of titles to remove
    for i in id_Details:
        l = dataframe.iloc[i] 

        j = next_same_lvl_index(dataframe[i:], l['lvl'])
        if j > i:
            dataframe.loc[i+1:j,'lvl'] = dataframe.loc[i+1:j,'lvl'].apply(lambda x: x-1)
            # remove those starts with lower case!
            idx_to_drop.extend([ind for ind in range(i+1, j+1) if dataframe.loc[ind,'description'].islower()])

        # if title description includes "e.g.", keep the part followed by e.g.
        if "e.g." in l['description']:
            eg_part = re.search("e.g. (.*)", l['description']) 
            dataframe.loc[i, 'description'] = eg_part.group(1)
        else:
            idx_to_drop.append(i)

    dataframe = dataframe.drop(idx_to_drop)
    return dataframe.reset_index(drop=True)

def read_label_file(file_name, max_level=TARGET_LEVEL):
    df = pd.read_csv(file_name, header=None, sep='\t', dtype=object, names=['code', 'lvl', 'description']) # cpc files downloaded from https://www.cooperativepatentclassification.org/cpcSchemeAndDefinitions/bulk
    df['lvl'] = df.apply(get_level, axis=1)

    if max_level in [1,3,4,6]:
        df = df[df['lvl']<=dict_lvl[max_level]]
    elif max_level == 8:
        df = df[df['lvl']<=MAX_DEPTH]

    df['description'] = df['description'].apply(clean_descr)
    df = rm_title_with_subtree(df)
    df = rm_Details(df)
    return df

def find_parent_node(df, child_lvl, diff=1):
    for _, row in df[::-1].iterrows():
        if row.lvl + diff == child_lvl:
            return row.code
    if child_lvl - diff >= -3 :
        return find_parent_node(df, child_lvl, diff+1)

def print_tree(root_node):
    for pre, _, node in RenderTree(root_node):
        print("%s%s" % (pre, node.name))

def build_tree(df):
    node_dict = {}
    root_title = df.loc[0, 'code']
    root_desc = df.loc[0, 'description']
    node_dict[root_title] = Node(root_desc)

    for i, row in df[1:].iterrows():
        child_lvl = row.lvl
        child_title = row.code
        child_desc = row.description
        father_title = find_parent_node(df[:i], child_lvl)
        node_dict[child_title] = Node(child_desc, parent = node_dict[father_title])

    return node_dict[root_title]
