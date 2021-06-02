PARENTDIR=$(dirname "$0")
BASEDIR=$(dirname "$PARENTDIR")
DATADIR="${BASEDIR}/data"
echo "************************************************"
echo "Downloading datasets into $DATADIR"

mkdir -p "${DATADIR}/ARC/"
git clone https://github.com/UKPLab/coling2018_fake-news-challenge
mv coling2018_fake-news-challenge/data/fnc-1/corpora/ARC/arc_bodies.csv "${DATADIR}/ARC/"
mv coling2018_fake-news-challenge/data/fnc-1/corpora/ARC/arc_stances_train.csv "${DATADIR}/ARC/"
mv coling2018_fake-news-challenge/data/fnc-1/corpora/ARC/arc_stances_test.csv "${DATADIR}/ARC/"
rm -rf coling2018_fake-news-challenge

mkdir -p "${DATADIR}/FNC-1/" 
git clone https://github.com/FakeNewsChallenge/fnc-1.git
mv fnc-1/README.md "${DATADIR}/FNC-1/"
mv fnc-1/train_bodies.csv "${DATADIR}/FNC-1/"
mv fnc-1/train_stances.csv "${DATADIR}/FNC-1/"
mv fnc-1/competition_test_bodies.csv "${DATADIR}/FNC-1/"
mv fnc-1/competition_test_stances.csv "${DATADIR}/FNC-1/"
rm -rf fnc-1

mkdir -p "${DATADIR}/IAC/"
wget http://nldslab.soe.ucsc.edu/iac/iac_v1.1.zip
unzip iac_v1.1.zip -d "${DATADIR}/IAC/"
mv "${DATADIR}/IAC/"/iac_v1.1/data/fourforums/discussions "${DATADIR}/IAC/"
mv "${DATADIR}/IAC/"/iac_v1.1/data/fourforums/annotations/author_stance.csv "${DATADIR}/IAC/"
rm -rf "${DATADIR}/IAC/"/iac_v1.1
rm -rf iac_v1.1.zip

mkdir -p "${DATADIR}/PERSPECTRUM/"
git clone https://github.com/CogComp/perspectrum.git
mv perspectrum/data/dataset/* "${DATADIR}/PERSPECTRUM/"
rm -rf perspectrum

mkdir -p "${DATADIR}/SemEval2016Task6/"
wget http://www.saifmohammad.com/WebDocs/stance-data-all-annotations.zip
unzip -p stance-data-all-annotations.zip data-all-annotations/trialdata-all-annotations.txt >"${DATADIR}/SemEval2016Task6/"/trialdata-all-annotations.txt
unzip -p stance-data-all-annotations.zip data-all-annotations/trainingdata-all-annotations.txt >"${DATADIR}/SemEval2016Task6/"/trainingdata-all-annotations.txt
unzip -p stance-data-all-annotations.zip data-all-annotations/testdata-taskA-all-annotations.txt >"${DATADIR}/SemEval2016Task6/"/testdata-taskA-all-annotations.txt
unzip -p stance-data-all-annotations.zip data-all-annotations/readme.txt >"${DATADIR}/SemEval2016Task6/"/readme.txt
rm -rf stance-data-all-annotations.zip

mkdir -p "${DATADIR}/SemEval2019Task7/"
wget https://ndownloader.figshare.com/files/16188500
mv 16188500 SemEval2019Task7.tar.bz2
tar -xvf SemEval2019Task7.tar.bz2 -C "${DATADIR}/SemEval2019Task7/"
mv "${DATADIR}/SemEval2019Task7/"/rumoureval2019/* "${DATADIR}/SemEval2019Task7/"/
unzip "${DATADIR}/SemEval2019Task7/"/rumoureval-2019-test-data -d "${DATADIR}/SemEval2019Task7/"
unzip "${DATADIR}/SemEval2019Task7/"/rumoureval-2019-training-data -d "${DATADIR}/SemEval2019Task7/"
rm "${DATADIR}/SemEval2019Task7/"/rumoureval-2019-test-data.zip 
rm "${DATADIR}/SemEval2019Task7/"/rumoureval-2019-training-data.zip 
rm -rf "${DATADIR}/SemEval2019Task7/"/rumoureval2019 
rm -rf SemEval2019Task7.tar.bz2