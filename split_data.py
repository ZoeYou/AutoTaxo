import pandas as pd
from sklearn.model_selection import train_test_split

df_hH = pd.read_csv("hH.csv", sep="\t", names=["section", "term", "hyper"], dtype=object).drop_duplicates()
df_hH['section'] = df_hH['section'].astype(str)
df_hH['term'] = df_hH['term'].astype(str).apply(lambda x:x.lower())
df_hH['hyper'] = df_hH['hyper'].astype(str).apply(lambda x:x.lower())

df_Hh = df_hH[["section", "hyper", "term"]]

# create train test data for hH
df_hH['source_text'] = ('predict hypernym: <' + df_hH['section'] + '> ' + df_hH['term']).astype(str)
df_hH['target_text'] = df_hH['hyper'].astype(str)
# split 
df_hH = df_hH[['source_text', 'target_text']].sample(frac=1)
train_hH, test_hH = train_test_split(df_hH, test_size=0.2)
print(train_hH)
print(test_hH)
# save
train_hH.to_csv("data/train_hH.csv", index=False)
test_hH.to_csv("data/test_hH.csv", index=False)


# create train test data for Hh
df_Hh['source_text'] = ('predict hyponym: <' + df_Hh['section'] + '> ' + df_Hh['term']).astype(str)
df_Hh['target_text'] = df_Hh['hyper'].astype(str)
# split 
df_Hh = df_Hh[['source_text', 'target_text']].sample(frac=1)
train_Hh, test_Hh = train_test_split(df_Hh, test_size=0.2)
print(train_Hh)
print(test_Hh)
# save
train_Hh.to_csv("data/train_Hh.csv", index=False)
test_Hh.to_csv("data/test_Hh.csv", index=False)