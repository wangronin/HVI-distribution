#!/bin/env bash

#SBATCH --job-name=HVI_benchmark
#SBATCH --array=0-4
#SBATCH --partition=cpu-long
#SBATCH --mem-per-cpu=1G
#SBATCH --time=7-00:00:00
#SBATCH --mail-user=<email>
#SBATCH --mail-type=END,FAIL
#SBATCH --ntasks=15
#SBATCH --cpus-per-task=7
#SBATCH --error="./err/HVI/%x-%j-%a.err"
#SBATCH --output="./out/HVI/%x-%j-%a.out"

ARGS=(zdt1 zdt2 zdt3 zdt4 zdt6)
FLAGS="--algo hvi-ucb --n_init_sample 10 --n-iter 30 --batch-size 1"

for i in {1..15}
do
   srun -N1 -n1 -c7 --exclusive python main.py $FLAGS --seed $i --problem ${ARGS[$SLURM_ARRAY_TASK_ID]} &
done
