wmModel="wiki-detox"
trainData="trainData"

# check wikiModel
if [ -d "$wmModel" ];
then
   echo "File $wmModel exists already."
else
   echo "File $wmModel does not exist, downloading ..."
   git clone https://github.com/ewulczyn/wiki-detox.git

   # wget https://pypi.python.org/pypi/3to2
   # find wiki-detox -name '*.py' | xargs 3to2
fi

# check trainData
if [ -d "$trainData" ];
then
   echo "File $trainData exists already."
else
   echo "File $trainData does not exist, downloading ..."
   mkdir $trainData
   wget https://ndownloader.figshare.com/articles/4563973/versions/2 -O toxicity.zip
   wget https://ndownloader.figshare.com/articles/4267550/versions/5 -O aggression.zip
   wget https://ndownloader.figshare.com/articles/4054689/versions/6 -O attack.zip
   unzip toxicity.zip -d $trainData
   unzip aggression.zip -d $trainData
   unzip attack.zip -d $trainData
   rm toxicity.zip
   rm aggression.zip
   rm attack.zip
fi

