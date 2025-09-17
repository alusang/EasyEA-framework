# EasyEA Code Description

Here are the descriptions of the Python scripts in this project:

- **Trans_name.py**: Uses LLM to translate entity names.
- **Summary_att_rel.py**: Uses LLM to summarize entity information.
- **Process_trans_summary.py**: Processes the translated and summarized entity information.
- **Embedding.py**: Uses LLM to embed entity information. You need to manually change the entity info file and the output pkl file name.
- **Feature_fusion.py**: Fuses features, allowing manual selection of which information to fuse.
- **Reasoning.py**: Uses LLM to select the most probable target entity from 10 candidate entities.
- **Evaluate.py**: Tests the final Hits@1.

## Dataset Description

We provide one processed dataset contains 10 pairs of entities, with files name.txt, att.txt, and rel.txt, which are used to understand the data format and debug the EasyEA code.

All datasets come from commonly used entity alignment datasets. Below are the sources for the datasets used in this experiment:

- **DBP15K**: [Link](https://github.com/kosugi11037/bert-int)
- **ICEWS**: [Link](https://github.com/IDEA-FinAI/Simple-HHEA)
- **SRPRS**: [Link](https://github.com/DexterZeng/CEA/tree/master/data)
- **DWY**: [Link](https://github.com/THUDM/SelfKG/tree/main)

## Embedding Code

**Embedding.py** is based on [LLM2Vec](https://github.com/McGill-NLP/llm2vec), which provides integrated query and embedding code. The execution speed is much faster than the method of first processing with GPT.

# Citation

If you find this code to be useful for your research, please consider citing.

```
@inproceedings{cheng2025easyea,
  title={EasyEA: Large Language Model is All You Need in Entity Alignment Between Knowledge Graphs},
  author={Cheng, Jingwei and Lu, Chenglong and Yang, Linyan and Chen, Guoqing and Zhang, Fu},
  booktitle={Findings of the Association for Computational Linguistics: ACL 2025},
  pages={20981--20995},
  year={2025}
}
```
