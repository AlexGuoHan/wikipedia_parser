#! /bin/bash
#SBATCH -o TEST.out
#SBATCH --job-name=sentiAnalysis
#SBATCH --cpus-per-task=40
#SBATCH --time=1-0

module load python
pip install -r requirements.txt --user
python3 sentiAnalysis_noMLP.py --wikiModelDir . --dataFileList dataFileList.txt --dataFileDir PARSED  --cpu 0

