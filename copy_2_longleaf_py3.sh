# copy the files to LONGLEAF
cp sentiAnalysis.py LONGLEAF/
cp LoadData.py LONGLEAF/
cp CleanTextData.py LONGLEAF/

# move it, but ignore *.py.bak
cp LONGLEAF/*.py ../longleaf/sentiFiles
