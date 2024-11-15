import torch
import torch.nn as nn
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("Distributed Text Preprocessing") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()

batch_size = 32
block_size = 128
max_iters = 10000
learning_rate = 2e-5
eval_iters = 100
n_embd = 384
n_head = 4
n_layer = 4
dropout = 0.2

def loader():
    # Load text from HDFS
    text_rdd = spark.sparkContext.textFile("hdfs:///path/to/wizard_of_oz.txt")
    
    # Preprocess and split data
    text = "".join(text_rdd.collect())  # Optionally collect to process locally
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    string_to_int = {ch: i for i, ch in enumerate(chars)}
    int_to_string = {i: ch for i, ch in enumerate(chars)}
    
    data = torch.tensor([string_to_int[c] for c in text], dtype=torch.long)
    
    n = int(0.9 * len(data))
    train_data = data[:n]
    test_data = data[n:]
    
    # Convert to RDD and DataFrames for distributed processing
    train_rdd = spark.sparkContext.parallelize(train_data.tolist()).partitionBy(8).cache()
    test_rdd = spark.sparkContext.parallelize(test_data.tolist()).partitionBy(8).cache()
    
    train_df = train_rdd.zipWithIndex().toDF(["value", "index"]).orderBy("index").drop("index")
    test_df = test_rdd.zipWithIndex().toDF(["value", "index"]).orderBy("index").drop("index")
    
    return vocab_size, train_df, test_df, string_to_int, int_to_string


def get_batch(data_df, batch_size=batch_size, block_size=block_size):
    device = 'cuda' if torch.cuda.is_available() else 'cpu' ##### training done on gpu 
    
   
    sample_df = data_df.sample(fraction=batch_size * block_size / data_df.count())
    sample_list = sample_df.collect()
    
  
    data_tensor = torch.tensor([row["value"] for row in sample_list], dtype=torch.long)
    

    x = torch.stack([data_tensor[i:i + block_size] for i in range(0, len(data_tensor), block_size)])
    y = torch.stack([data_tensor[i + 1:i + block_size + 1] for i in range(0, len(data_tensor), block_size)])
    
    x, y = x.to(device), y.to(device)
    return x, y
