BASEDIR=$(dirname "$0")
PARENTDIR=$(dirname "$BASEDIR")
DATA_DIR="$PARENTDIR/data"
echo "************************************************"
echo "Downloading datasets into $DATA_DIR"

mkdir $DATA_DIR/ARC -p
git clone https://github.com/UKPLab/coling2018_fake-news-challenge
mv coling2018_fake-news-challenge/data/fnc-1/corpora/ARC/arc_bodies.csv $DATA_DIR/ARC
mv coling2018_fake-news-challenge/data/fnc-1/corpora/ARC/arc_stances_train.csv $DATA_DIR/ARC
mv coling2018_fake-news-challenge/data/fnc-1/corpora/ARC/arc_stances_test.csv $DATA_DIR/ARC
rm -rf coling2018_fake-news-challenge

mkdir $DATA_DIR/FNC-1 -p 
git clone https://github.com/FakeNewsChallenge/fnc-1.git
mv fnc-1/README.md $DATA_DIR/FNC-1
mv fnc-1/train_bodies.csv $DATA_DIR/FNC-1
mv fnc-1/train_stances.csv $DATA_DIR/FNC-1
mv fnc-1/competition_test_bodies.csv $DATA_DIR/FNC-1
mv fnc-1/competition_test_stances.csv $DATA_DIR/FNC-1
rm -rf fnc-1

mkdir $DATA_DIR/IAC -p
wget http://nldslab.soe.ucsc.edu/iac/iac_v1.1.zip
unzip iac_v1.1.zip -d $DATA_DIR/IAC
mv $DATA_DIR/IAC/iac_v1.1/data/fourforums/discussions $DATA_DIR/IAC
mv $DATA_DIR/IAC/iac_v1.1/data/fourforums/annotations/author_stance.csv $DATA_DIR/IAC
rm -rf $DATA_DIR/IAC/iac_v1.1
rm -rf iac_v1.1.zip

mkdir $DATA_DIR/PERSPECTRUM -p
git clone https://github.com/CogComp/perspectrum.git
mv perspectrum/data/dataset/* $DATA_DIR/PERSPECTRUM
rm -rf perspectrum

mkdir $DATA_DIR/SemEval2016Task6 -p
wget http://www.saifmohammad.com/WebDocs/stance-data-all-annotations.zip
unzip -p stance-data-all-annotations.zip data-all-annotations/trialdata-all-annotations.txt >$DATA_DIR/SemEval2016Task6/trialdata-all-annotations.txt
unzip -p stance-data-all-annotations.zip data-all-annotations/trainingdata-all-annotations.txt >$DATA_DIR/SemEval2016Task6/trainingdata-all-annotations.txt
unzip -p stance-data-all-annotations.zip data-all-annotations/testdata-taskA-all-annotations.txt >$DATA_DIR/SemEval2016Task6/testdata-taskA-all-annotations.txt
unzip -p stance-data-all-annotations.zip data-all-annotations/readme.txt >$DATA_DIR/SemEval2016Task6/readme.txt
rm -rf stance-data-all-annotations.zip

mkdir $DATA_DIR/SemEval2019Task7 -p
wget https://ndownloader.figshare.com/files/16188500
mv 16188500 SemEval2019Task7.tar.bz2
tar -xvf SemEval2019Task7.tar.bz2 -C $DATA_DIR/SemEval2019Task7
mv $DATA_DIR/SemEval2019Task7/rumoureval2019/* $DATA_DIR/SemEval2019Task7/
unzip $DATA_DIR/SemEval2019Task7/rumoureval-2019-test-data -d $DATA_DIR/SemEval2019Task7
unzip $DATA_DIR/SemEval2019Task7/rumoureval-2019-training-data -d $DATA_DIR/SemEval2019Task7
rm $DATA_DIR/SemEval2019Task7/rumoureval-2019-test-data.zip 
rm $DATA_DIR/SemEval2019Task7/rumoureval-2019-training-data.zip 
rm -rf $DATA_DIR/SemEval2019Task7/rumoureval2019 
rm -rf SemEval2019Task7.tar.bz2