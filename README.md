# Technological Taxonomies for Hypernym and Hyponym Retrieval in Patent Texts

This repository contains the implementation of the research paper titled "Technological Taxonomies for Hypernym and Hyponym Retrieval in Patent Texts".

Paper: [https://hal.science/hal-03850399](https://hal.science/hal-03850399)

## Implementation

### Step 1: Download CPC Titles

Download the CPC titles from [CPC bulk data](https://www.cooperativepatentclassification.org/cpcSchemeAndDefinitions/bulk):
```
wget https://www.cooperativepatentclassification.org/sites/default/files/cpc/bulk/CPCTitleList202401.zip
```

### Step 2: Unzip CPC Titles List
Once you have downloaded the CPC titles list, unzip it.



### Step 3: Create Taxonomy Trees
Run the following command to create taxonomy trees for each domain:
```
python tree2pairs.py
```

### Step 4: Implement Post-Processing
If you want to have separate pairs of concept-hypernym for entities and descriptions of concepts, continue implementing the post-processing with the previous output directory:
```
python tree2pairs ./trees_CPC8-2
```


## Citation
If you find this work useful for your research, please consider citing our paper:
```
@article{zuo2022technological,
  title={Technological taxonomies for hypernym and hyponym retrieval in patent texts},
  author={Zuo, You and Li, Yixuan and Garc{\'\i}a, Alma Parias and Gerdes, Kim},
  journal={arXiv preprint arXiv:2212.06039},
  year={2022}
}
  ```
