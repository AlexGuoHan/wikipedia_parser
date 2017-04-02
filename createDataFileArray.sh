rm -rf dataFileArray
mkdir dataFileArray

idx=0
for file in `cat dataFileList.txt`
do
    echo $file > dataFileArray/dataFileArray$idx.txt
    idx=$(($idx+1))
done

