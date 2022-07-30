
目录: data_analysis   
用于数据集分析, 分布、分词长度等等一系列处理脚本  
  
目录: word2vec  
用于得到的numpy.files用于初始化tri-gram Embedding bag  
  
1_query_hash.py  
根据Query Hash值划分训练集、验证集合  
  
3_create_pretrain_data_brand_color.py  
创建product2brand， product2color的预训练数据集合  
  
3_create_pretrain_data.py  
创建product2query的预训练数据集合  
  
4_translation_es_us.py  
使用翻译模型，进行数据翻译   
  
5_tokenizer_n-gram.py  
使用细粒度分词(tri-gram)的词表的处理脚本   
  
6_rebalance_negative_samples.py  
平衡数据集, 用于扩充complement、irrelevant数据集合   
  
7_confidient_learning_filter.py  
通过交叉验证, 删除"低置信度"的样本   