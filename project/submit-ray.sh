#!/bin/bash
#SBATCH --job-name=ray_optuna_cifar_cpu
#SBATCH --nodes=2             # 1 head + 1 worker
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --partition=plgrid
#SBATCH --output=ray_job_cpu_%j.log

# ==========================================
# 1. Konfiguracja środowiska
# ==========================================
module load miniconda3
eval "$(conda shell.bash hook)"
conda activate $SCRATCH/ray_env
export LD_LIBRARY_PATH=$SCRATCH/ray_env/lib:$LD_LIBRARY_PATH
# ==========================================
# 2. Inicjalizacja Head Node
# ==========================================
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

port=6379
export ip_head=$head_node_ip:$port
echo "Head Node: $ip_head"

echo "Uruchamianie Ray Head na węźle: $head_node"
srun --nodes=1 --ntasks=1 -w "$head_node" \
    ray start --head --node-ip-address="$head_node_ip" --port=$port \
    --temp-dir="/tmp/$USER/ray" \
    --num-cpus "${SLURM_CPUS_PER_TASK}" --block &

sleep 10

# ==========================================
# 3. Inicjalizacja Worker Nodes
# ==========================================
worker_num=$((SLURM_JOB_NUM_NODES - 1))
if [ $worker_num -gt 0 ]; then
    echo "Uruchamianie $worker_num węzłów typu Worker..."
    for ((i = 1; i <= worker_num; i++)); do
        node_i=${nodes_array[$i]}
        echo "Uruchamianie Worker $i na węźle $node_i"
        srun --nodes=1 --ntasks=1 -w "$node_i" \
            ray start --address "$ip_head" \
            --temp-dir="/tmp/$USER/ray" \
            --num-cpus "${SLURM_CPUS_PER_TASK}" --block &
    done
fi

sleep 10

# ==========================================
# 4. Uruchomienie skryptu
# ==========================================
echo "Uruchamianie skryptu treningowego na CPU..."
python train.py
