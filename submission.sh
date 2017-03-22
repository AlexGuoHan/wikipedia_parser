#! /bin/bash
#SBATCH -o R-example-%A.out
#SBATCH --job-name=test_download
#SBATCH --ntasks=1
#SBATCH --time=3:00:00

module load python

pip3 install -r requirements.txt
python3 wikipedia_data_parser.py --pageTitleDir page_titles.txt --allDumpTextDir all_dumps.txt
