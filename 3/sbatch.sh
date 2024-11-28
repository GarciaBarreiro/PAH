#!/bin/bash

#SBATCH -J gpu_job
#SBATCH -o gpu_job.o%j
#SBATCH -e gpu_job.o%j
#SBATCH -c 32
#SBATCH --mem=8G
#SBATCH --gres=gpu
#SBATCH -t 00:40:00

module load cesga/2020 cuda-samples/11.2

for m in 8 16 32 64 128 256 512 1024 2048 4196; do
    for n in 2 4 8 16 32 64 128 256 512 1024; do
        for p in 8 16 32; do
            if (( $n * $p < 2048 )); then
                srun $1 $m $n $p $3 > cuda.out # cuda
                srun $2 $m $n $p    > seq.out  # seq
                diff cuda.out seq.out   # unless error, output should be empty
            fi
        done
    done
done
