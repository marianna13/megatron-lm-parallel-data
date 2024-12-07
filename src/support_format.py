# this function directly reads the parquet file in adequate format to process them through the pipeline. 
# if you want to process all rows don't pass num_rows
# tested for https://huggingface.co/datasets/HuggingFaceFW/fineweb/resolve/main/data/CC-MAIN-2013-20/000_00000.parquet

def process_parquet(input_file, json_keys, num_rows=None):
    df = pd.read_parquet(input_file)
    if num_rows:
        df = df.head(num_rows)
    processed_data = []
    for _, row in df.iterrows():
        processed_data.append({k: row[k] for k in json_keys})
    return processed_data
