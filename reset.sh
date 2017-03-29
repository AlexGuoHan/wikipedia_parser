rm *
rm -rf parserFiles
rm -rf sentiFiles
git clone https://github.com/AlexGuoHan/wikipedia_parser.git
mv wikipedia_parser/* .
rm -rf wikipedia_parser
ls PARSED | grep .tsv > dataFileList.txt
~
