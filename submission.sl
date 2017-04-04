#! /bin/bash
#SBATCH --array=0-215
#SBATCH -o slurmOut
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH -t 02:00:00
#SBATCH -N 1
#SBATCH -p general
#SBATCH --mem-per-cpu=20g

module load python
pip3 install -r requirements.txt --user
python3 sentiAnalysis.py --wikiModelDir . --trainDataDir trainData --dataFileList dataFileArray/dataFileArray$SLURM_ARRAY_TASK_ID.txt --dataFileDir PARSED
