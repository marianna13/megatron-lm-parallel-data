import pyarrow.parquet as pq
import pyarrow.json as pa_json
import json

def changetojsonl(input_file, output_file):
    if input_file.endswith('.parquet'):
        table = pq.read_table(input_file)
        records = table.to_pandas().to_dict(orient='records')
    elif input_file.endswith('.json'):
        with open(input_file, 'r') as f:
            records = json.load(f)
    else:
        raise ValueError("Input file must be either a Parquet or JSON file.")
    
    with open(output_file, 'w') as f:
        for record in records:
            f.write(f"{json.dumps(record)}\n")
