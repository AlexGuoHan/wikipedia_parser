#! /bin/bash
#SBATCH -o R-example-%A.out
#SBATCH --job-name=test_download
#SBATCH --ntasks=1
#SBATCH --time=3:00:00

module load python

# download mwxml
wget https://pypi.python.org/packages/e6/71/2f2c1c72f9293b663e17bba6d714cc78dbb1972a2106857eca20048a716a/mwxml-0.2.2.tar.gz#md5=cedb7d210b883afbe21fb6b81f5e5bce
tar -xvzf mwxml-0.2.2.tar.gz

pip3 install -r requirements.txt
python3 wikipedia_data_parser.py --pageTitleDir page_titles.txt --allDumpTextDir all_dumps.txt
