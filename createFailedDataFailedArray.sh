rm -rf dataFailFileArray
mkdir dataFailFileArray

# get the list of JobID that went wrong
sh getJobInfo.sh past | grep -v COMPLETED | grep -v batch | cut -c9-12 | grep -v ID | grep -v \- > tmp_failed_jobid.txt

idx=0
for id in `cat tmp_failed_jobid.txt`
do
    cat dataFileArray/dataFileArray$id.txt >> dataFailFileArray/dataFileArray$idx.txt
    idx=$(($idx + 1))
done

rm tmp_failed_jobid.txt
