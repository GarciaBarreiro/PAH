#!/bin/bash
#SBATCH -J job
#SBATCH -o %j.out
#SBATCH -e %j.err
#SBATCH -n 1
#SBATCH -c 32
#SBATCH -t 00:10:00
#SBATCH --gres gpu:a100
#SBATCH --mem=32G

echo "m,n,repetitions,threads_per_block,transfer_time,kernel_time,total_time" > $2

for size in 4000 10000 20000; do
    for tpb in 32 64 128; do
        srun $1 $size $size 5 [0.1,0.2,0.3,0.4,0.5] $tpb $2
        srun $1 $size $size 10 [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.1] $tpb $2
    done
done
