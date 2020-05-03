"""
choose integer number of ratio negative/positive to sample (0 will use all negative pairs)
"""
neg_sample = 2
"""
Choose one split trategy ["isolation","random"] : 
- random will randomly spread positive node pairs in 80-20 fashion
- isolation will isolate 1 node from some topics in test (none pair in train will have these nodes)
"""
strategy = "isolation"
"""
Choose to use the selected setup to create a new split 
or reuse a previously created one (useful to repeat exact same experiment)
"""
create_new_split = False

"""
You can choose to use one of ["FASTTEXT","BERT"] as initial word_embedding encoding for the nodes in the datasets
"""
word_embedding_encoding = "FASTTEXT"


"""
These are the default values

neg_sample = 2
strategy = "random"
create_new_split = True
word_embedding_encoding = "FASTTEXT"
"""
