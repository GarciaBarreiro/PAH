#!/bin/bash

#SBATCH -J gpu_job
#SBATCH -o gpu_job.o%j
#SBATCH -e gpu_job.o%j
#SBATCH -c 32
#SBATCH --mem=4G
#SBATCH --gres=gpu
#SBATCH -t 00:10:00

module load cesga/2020 cuda-samples/11.2

for i in 8 16 32 64 128 256 512 1024; do
    srun $1 $i $2
done
