#!/bin/bash
#SBATCH -o TEST.out
#SBATCH --job-name=test_download
#SBATCH --ntasks=300
#SBATCH --time=1-0

wget -i all_dumps_full.txt
