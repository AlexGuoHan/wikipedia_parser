#! /bin/bash
#SBATCH -o TEST.out
#SBATCH --job-name=test_download
#SBATCH --ntasks=300
#SBATCH --time=1-0

mkdir all_dump_files
wget -i all_dumps_full.txt -O all_dump_files/
