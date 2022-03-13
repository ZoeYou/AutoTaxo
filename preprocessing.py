from anytree import Node
import re
import pandas as pd

TARGET_LEVEL = 8

global dict_lvl
dict_lvl = {1: -3, 3: -2, 4: -1, 6: 0}

def convert_to_int(x):
    lvl = x['lvl']
    try:
        return int(lvl)
    except ValueError:
        title = x['title']
        if len(title) == 1:
            return -3
        elif len(title) == 3:
            return -2
        elif len(title) == 4:
            return -1

def clean_descr(description):
    """
    1. remove references from the description, such as *** (preserving A23B; obtaining protein compositions for foodstuffs A23J1/00;)
    2. remove '{' and '}'
    """
    description = description.replace('{', '').replace('}','')
    description = re.sub(r'\ ?\([\w\W]*([A-Z]{1}[0-9]{2}[A-Z]{1}[0-9]*[\/]*[0-9]*)*[\w\W]*\)', '', description)
    description = re.sub('as specified in the subgroups? and ','', description)
    return description

def has_cpc(description):
    """
    4. check whether a title desciption has CPC code inside
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
    5. check whether a title description is a trash title
    """
    description = description.lower()
    if ("subject matter not" in description and "provided for in" in description) or ("covered by" in description and (" subclass" in description or " group" in description)) or ("dummy group" in description):
        return True
    else:
        return False

def next_same_lvl_index(subdf,lvl):
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
    # get indices of description 
    dataframe = dataframe.reset_index(drop=True)
    id_hascpc = dataframe[dataframe['description'].apply(has_cpc) | dataframe['description'].apply(is_trash_title)].index
    
    idx_to_drop = []
    for i in id_hascpc:
        l = dataframe.iloc[i]
        if i == (dataframe.shape[0]-1) or l['lvl'] >= dataframe.iloc[i+1]['lvl']: 
            idx_to_drop.append(i)
        else:
            j = next_same_lvl_index(dataframe[i:], l['lvl'])
            if i == j:
                idx_to_drop.append(i)
            elif j>i:
                idx_to_drop.extend(range(i,j+1))
            else:
                raise ValueError("Problem with subtree index!")

    dataframe = dataframe.drop(idx_to_drop)
    return dataframe.reset_index(drop=True)

def rm_Details_of(dataframe):
    """
    Remove content of titles starts with "Details of", and upgrade the level of its children titles
    """
    id_Details_of = dataframe[dataframe['description'].apply(lambda x: "Details of " == x[:11])].index

    idx_to_drop = []
    for i in id_Details_of:
        l = dataframe.iloc[i]
        j = next_same_lvl_index(dataframe[i:], l['lvl'])  
        if i==j:j += 1
        dataframe.loc[i:j,'lvl'] = dataframe.loc[i:j,'lvl'].apply(lambda x: x-1)
        idx_to_drop.append(i)

    dataframe = dataframe.drop(idx_to_drop)
    return dataframe

def read_label_file(file_name, max_level=TARGET_LEVEL):
    df = pd.read_csv(file_name, header=None, sep='\t', dtype=object, names=['title', 'lvl', 'description'])
    df['lvl'] = df.apply(convert_to_int, axis=1)

    if max_level in [1,3,4,6]:
        df = df[df['lvl']<=dict_lvl[max_level]]

    df['description'] = df['description'].apply(clean_descr)
    df = rm_title_with_subtree(df)
    df = rm_Details_of(df)
    return df.dropna().reset_index(drop=True)

def find_father(df, child_lvl):
    for i, row in df[::-1].iterrows():
        if row.lvl + 1 == child_lvl:
            return row.title

def build_tree(df):
    node_dict = {}
    root_title = df.loc[0, 'title']
    root_desc = df.loc[0, 'description']
    node_dict[root_title] = Node(root_desc)

    for i, row in df[1:].iterrows():
        child_lvl = row.lvl
        child_title = row.title
        child_desc = row.description

        father_title = find_father(df[:i], child_lvl)
        node_dict[child_title] = Node(child_desc, parent = node_dict[father_title])
    return node_dict[root_title]
