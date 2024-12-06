# Megatron-LM distributed data preprocessing

## Table of Contents
- [Description](#description)
- [Installation](#installation)
- [Usage](#usage)
- [Ray cluster on Slurm](#start-ray-cluster-on-slurm)
- [Roadmap](#roadmap)
- [Contributing](#contributing)

## Description

The main script is [preprocess_data_parallel.py](src/preprocess_data_parallel.py). The script has the same functionality as Megatron-LM's [preprocess_data.py](https://github.com/NVIDIA/Megatron-LM/blob/main/tools/preprocess_data.py) script, however it can tokenize multiple files in parallel on N compute nodes. Two variables are responsible of how this parallelisation will works:

1. `--workers` - this is number of workers that a single file will be tokenized with. It cannot be more than number of CPU cores per compute node.
2. `--cpus-per-ray-worker` - number of CPU cores per Ray worker aka one job. The `workers` should be smaller or equal to `cpus-per-ray-worker`.

To make multi-node tokenization possible we use [Ray cluster](https://docs.ray.io/en/latest/cluster/getting-started.html). See more info [Ray cluster setup on Slurm](#start-ray-cluster-on-slurm). 

## Installation

1. Clone Megatron-LM repo (you can skip this step if it's already cloned):
    ```bash
    git clone https://github.com/NVIDIA/Megatron-LM.git
    ```
    Install Megatron-LM dependencies. See [here](https://github.com/NVIDIA/Megatron-LM/tree/main) more info on installation of Megatron-LM.

2. Install requirements:
    ```bash
    pip install -r requirements.txt
    ```

## Usage


The arguments are the same as Megatron-LM's original `preprocess_data.py` except for `--cpus-per-ray-worker` - this argument specifes how many cpus each Ray worker should use (default is 1). The script will created a temporary directory that for each `.jsonl` file it's corresponding tokenized `.bin` and `.idx` files will be saved. Then, after finishing tokenization, all intermediate files will be merged in one and removed afterwards.

Example:

```bash
MEGATRON_PATH="Megatron-LM"
cd $MEGATRON_PATH
export PYTHONPATH=$MEGATRON_PATH:$PYTHONPATH


INPUT="samples/c4"
OUTPUT_PREFIX="tokenized_c4_samples"
TOKENIZER_TYPE="HuggingFaceTokenizer"
TOKENIZER_MODEL="EleutherAI/gpt-neox-20b"
CPUS_PER_WORKER=12

SCRIPT="preprocess_data_parallel.py"


CMD="python $SCRIPT \
       --input $INPUT \
       --output-prefix $OUTPUT_PREFIX \
       --tokenizer-type $TOKENIZER_TYPE \
       --tokenizer-model $TOKENIZER_MODEL \
       --workers $SLURM_CPUS_PER_TASK \
       --cpus-per-ray-worker $CPUS_PER_WORKER \
       --append-eod"

bash -c $CMD
```


## Start Ray cluster on Slurm

```bash
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)

head_node=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

head_node_ip="$(nslookup "$head_node" | grep -oP '(?<=Address: ).*')"
echo "Head node: $head_node_ip"

export TOKENIZERS_PARALLELISM=false # important for HF tokenizers

port=20156
ip_head=$head_node_ip:$port
export ip_head
echo "IP Head: $ip_head"

echo "Starting HEAD at $head_node"
srun --nodes=1 --ntasks=1 -w "$head_node"  \
    ray start --head --node-ip-address="$head_node_ip" --port=$port \
    --num-cpus ${SLURM_CPUS_PER_TASK} --block & # start head node

sleep 10

worker_num=$((SLURM_JOB_NUM_NODES - 1)) # nu,ber of ray workers minus head node

for ((i = 1; i <= worker_num; i++)); do
    node=${nodes_array[$i]}
    node_i="${node}i"
    echo "Starting WORKER $i at $node"
    this_node_ip="$(nslookup "$node_i" | grep -oP '(?<=Address: ).*')"
    srun --nodes=1 --ntasks=1 -w "$node" \
        ray start --address "$ip_head" \
        --node-ip-address="$this_node_ip"  \
        --num-cpus ${SLURM_CPUS_PER_TASK} --block &
    sleep 10
done

export RAY_ADDRESS="$head_node_ip:$port"
```

## Roadmap

- Support different fileformats (`json`, `parquet`, `jsonl.gs` etc).
- Support multimodal datasets.

## Contributing

Use ruff for formatting and linting:
```bash
ruff check src && ruff format src
```
