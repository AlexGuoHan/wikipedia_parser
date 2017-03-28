# copy the files to LONGLEAF
cp sentiAnalysis.py LONGLEAF/
cp LoadData.py LONGLEAF/
cp CleanTextData.py LONGLEAF/

# python3 to python2
3to2 -w LONGLEAF/sentiAnalysis.py
3to2 -w LONGLEAF/LoadData.py
3to2 -w LONGLEAF/CleanTextData.py

# move it, but ignore *.py.bak
cp LONGLEAF/*.py ../longleaf/sentiFiles
