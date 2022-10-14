import pandas as pd
import os, ast

def get_perplexity():
    pass


IN_DIR = "hH_ipc8_all_lower_zh"
df = pd.read_csv(os.path.join(IN_DIR, "predictions.csv"), index_col=None)
df["Generated Text"] = df["Generated Text"].apply(lambda x: ast.literal_eval(x)).tolist()

print("hits@1: ", df["H@1"].mean())
print("hits@3: ", df["H@3"].mean())
print("hits@10: ", df["H@10"].mean())
print("mrr: ", df["MRR"].mean())
