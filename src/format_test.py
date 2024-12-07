import argparse
import json
import os
import time
import pandas as pd
import shutil
import logging
from transformers import GPT2Tokenizer
import ray

from megatron.training.arguments import _add_tokenizer_args
from megatron.core.datasets import indexed_dataset
from tools.preprocess_data import get_file_name, check_files_exist, Partition
from megatron.core.datasets.indexed_dataset import (
    IndexedDataset,
    IndexedDatasetBuilder,
    get_bin_path,
    get_idx_path,
)

args = argparse.Namespace(
    input="/tokenization/Megatron-LM/000_00000.parquet",
    json_keys=["text"],
    split_sentences=False,
    keep_newlines=False,
    append_eod=False,
    lang="english",
    output_prefix="output",
    workers=1,
    partitions=1,
    log_interval=1000,
    keep_sequential_samples=False,
    tokenizer_type="GPT2",
    vocab_size=50257,
    vocab_file=None,
    merge_file=None,
    tokenizer_model="gpt2",
    tiktoken_pattern=None,
    tiktoken_num_special_tokens=0,
    tiktoken_special_tokens=None,
)

args.rank = 0

def preprocess_data(args):
    in_ss_out_names = []
    file_name, extension = os.path.splitext(args.input)
    sentence_split_file = file_name + "_ss" + extension
    file_names = {
        "partition": args.input,
        "sentence_split": sentence_split_file,
        "output_prefix": args.output_prefix,
    }
    in_ss_out_names.append(file_names)

    partition = Partition(args, args.workers)

    tokenized_data = process_parquet(args.input, args.json_keys, num_rows=200)
    
    start_time = time.time()
    tokenize_data_parallel(tokenized_data, args.output_prefix, args)
    end_time = time.time()

    processing_time_minutes = (end_time - start_time) / 60
    print(f"Processing time: {processing_time_minutes:.2f} minutes")

def process_parquet(input_file, json_keys, num_rows=None):
    df = pd.read_parquet(input_file)
    if num_rows:
        df = df.head(num_rows)
    processed_data = []
    for _, row in df.iterrows():
        processed_data.append({k: row[k] for k in json_keys})
    return processed_data

def tokenize_data_parallel(processed_data, output_prefix, args):
    tokenizer = GPT2Tokenizer.from_pretrained(args.tokenizer_model)
    if args.append_eod:
        tokenizer.add_special_tokens({'additional_special_tokens': ['<eod>']})
    
    chunk_size = len(processed_data) // 20
    data_chunks = [processed_data[i:i + chunk_size] for i in range(0, len(processed_data), chunk_size)]
    
    futures = [ray.remote(tokenize_chunk).remote(chunk, output_prefix, args) for chunk in data_chunks]
    ray.get(futures)

def tokenize_chunk(processed_data_chunk, output_prefix, args):
    tokenizer = GPT2Tokenizer.from_pretrained(args.tokenizer_model)
    tokenized_data = []
    
    for entry in processed_data_chunk:
        text = entry["text"]
        max_length = tokenizer.model_max_length
        tokenized_line = tokenizer.encode(text, add_special_tokens=False, truncation=True, max_length=max_length)
        tokenized_data.append(tokenized_line)

    with open(f"{output_prefix}_tokenized.jsonl", 'a') as f:
        for tokens in tokenized_data:
            json.dump({"tokens": tokens}, f)
            f.write("\n")

def merge_datasets(args):
    output_bin_files = {}
    output_idx_files = {}
    builders = {}
    tokenizer = GPT2Tokenizer.from_pretrained(args.tokenizer_model)

    for key in args.json_keys:
        output_bin_files[key] = f"{args.output_prefix}_{key}_document.bin"
        output_idx_files[key] = f"{args.output_prefix}_{key}_document.idx"
        builders[key] = indexed_dataset.IndexedDatasetBuilder(
            output_bin_files[key],
            dtype=indexed_dataset.DType.optimal_dtype(tokenizer.vocab_size),
        )

        tokenized_file = f"{args.output_prefix}_tokenized.jsonl"
        builders[key].add_index(tokenized_file)

        builders[key].finalize(output_idx_files[key])

if __name__ == "__main__":
    ray.init(ignore_reinit_error=True, num_cpus=50)
    preprocess_data(args)
    ray.shutdown()
