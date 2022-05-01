#!/bin/bash
#SBATCH --job-name=cv_check_3_30
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=annabelle.ahrv@gmail.com
#SBATCH --time=01:00:00
#SBATCH --mem=5G
#SBATCH --account=def-pbellec
#SBATCH --array=0-4
#SBATCH -o /home/harveyaa/projects/def-pbellec/harveyaa/slurm_output/output_3_30_%a.out


PLIM_ARRAY=(5 10 15 20 25)
p_out_parent='/home/harveyaa/projects/def-pbellec/harveyaa/data/ids/mleming/'

source /home/harveyaa/projects/def-pbellec/harveyaa/mtl_env/bin/activate

plim=${PLIM_ARRAY[$SLURM_ARRAY_TASK_ID]}
min_train=30
min_members=3

p_out=$p_out_parent'cv_'$min_members'_'$min_train'_'$plim
mkdir $p_out

echo 'Generating CV splits...'
python generate_cv_splits.py --plim $plim --min_train $min_train --min_members $min_members --p_out $p_out --generate_figures

echo 'Benchmarking confound accuracy on splits...'
python benchmark2.py --p_ids $p_out --p_out $p_out

echo 'plim '$plim >> $p_out'/README.md'
echo 'min_train '$min_train >> $p_out'/README.md'
echo 'min_members '$min_members >> $p_out'/README.md'