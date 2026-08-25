#!/bin/bash

# Benchmark Newton-Schulz distribution strategies over the GTP axis.
#
# GTP shards dense weights. At the production degree of 64 the group fits inside one
# GB200/GB300 NVLink domain, so NVLink is left enabled here: that is the transport a
# real run uses. On Lyris (4 GPUs/node) a 64-rank group needs 16 nodes, and --segment=16
# keeps them inside a single NVLink domain.
#
# GTP degree and node count must agree: nodes * 4 (GPUs/node) == GTP. To try a smaller
# degree first, override both, e.g.
#   sbatch --nodes=2 --segment=2 --export=ALL,GTP=8 tools/muon_analysis/bench_ns_gtp.sbatch

#SBATCH -p gb300
#SBATCH --account=coreai_dlalgo_llm
#SBATCH --nodes=16
#SBATCH --exclusive
#SBATCH -t 00:30:00
#SBATCH --mem=0
# No --gpus-per-node: Lyris rejects it as an invalid gres spec. --exclusive gives all
# 4 GPUs on the node, matching Experiments/profile.sh.
#SBATCH --ntasks-per-node=4
#SBATCH --segment=16
#SBATCH --job-name=coreai_dlalgo_llm-dist_muon.bench_ns_gtp
#SBATCH --comment='{"APS":{}}'

export NCCL_DEBUG=WARN

GTP=${GTP:-64}

ROOT_DIR="/lustre/fsw/coreai_dlalgo_llm/zhengywang/dist_muon"
REPO_DIR="${ROOT_DIR}/Megatron-LM-experiments"
IMAGE_PATH="${IMAGE_PATH:-${ROOT_DIR}/images/mcore-moe-te-main-hybridep-cutedsl-arm.sqsh}"

# TAG separates runs on different hardware, e.g. sbatch -p gb200 --export=ALL,TAG=gb200
NAME="perfbot/run1/ns_bench/gtp${TAG:+_$TAG}"
LOGS_DIR="${ROOT_DIR}/${NAME}/logs"
mkdir -p "${LOGS_DIR}"

DATETIME=$(date +"date_%y-%m-%d_time_%H-%M-%S")

# torchrun is not used, so the rendezvous env has to come from Slurm explicitly.
export MASTER_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n1)
export MASTER_PORT=29512

# USE_SYRK=1 forwards --use-syrk to newton_schulz_tp (Triton triangular kernel).
# sbatch --export=ALL,USE_SYRK=1 bench_ns_gtp.sbatch
SYRK_OPTION=""
if [[ "${USE_SYRK:-0}" == "1" ]]; then SYRK_OPTION="--use-syrk"; fi

options=" \
    --group gtp \
    --modelled-world-size 12288 \
    --tp 1 --gtp ${GTP} --ep 64 --etp 1 --egtp 4 \
    --shard-latent-proj \
    --num-ns-steps 16 \
    --coefficient-type polar_express \
    --iters 20 \
    --warmup 5 \
    ${SYRK_OPTION} "

run_cmd="export RANK=\${SLURM_PROCID} WORLD_SIZE=\${SLURM_NTASKS} LOCAL_RANK=\${SLURM_LOCALID}; \
python -u ${REPO_DIR}/tools/muon_analysis/bench_ns_strategies.py ${options}"

srun -l \
    --mpi=none \
    --container-image "${IMAGE_PATH}" \
    --container-mounts "/lustre:/lustre" \
    --no-container-mount-home \
    --output="${LOGS_DIR}/%x_%j_${DATETIME}.log" \
    bash -c "${run_cmd}"
