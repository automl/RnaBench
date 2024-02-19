#! /bin/bash

################################################################################
# Extract Internal Data
################################################################################
tar -xvzf data/3D.tar.gz -C data
# rm data/3D.tar.gz
tar -xvzf RnaBench/lib/data/CMs.tar.gz -C RnaBench/lib/data
# rm RnaBench/lib/data/CMs.tar.gz

OUTDIR=external_algorithms
mkdir -p ${OUTDIR}
cd ${OUTDIR}


################################################################################
# Data pipelines
################################################################################
# NCBI's Blast
# wget 'ftp://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/2.12.0/ncbi-blast-2.12.0+-x64-linux.tar.gz'
wget 'https://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/2.12.0/ncbi-blast-2.12.0+-x64-linux.tar.gz'
tar -xvzf ncbi-blast-2.12.0+-x64-linux.tar.gz
rm ncbi-blast-2.12.0+-x64-linux.tar.gz
# bpRNA
git clone https://github.com/hendrixlab/bpRNA.git
# Get Rfam.cm, Rfam.clanin and index Rfam.cm
# python -m RnaBench.download  # download and unzip Rfam.cm and Rfam.clanin
# cmpress RnaBench/lib/data/CMs/Rfam.cm
################################################################################
# Folding Algorithms
################################################################################
TESTSEQ="ACGUCGUCAGUCGAUCGAUCGAUCCGCCUAGUCAAAGUCCUCGAAGCUCUCCCUUAG"
################################################################################
# LinearFold
################################################################################
echo "### Install LinearFold"
git clone https://github.com/LinearFold/LinearFold.git
cd LinearFold
make
echo "### Test LinearFold -C"
echo ${TESTSEQ} | ./linearfold
echo "### Test LinearFold -V"
echo ${TESTSEQ} | ./linearfold -V
cd ..
################################################################################
# IPKNOT
################################################################################
echo "Install IpKnot"
# cd ${OUTDIR}
gdown 1Oh3kNYbnv_22i4IIYXavPcOo1xSdm1vB
chmod u+x ipknot  # does this work on e.g. NEMO?
echo "### Test IpKnot"
echo ">test" >> test.fa
echo $TESTSEQ >> test.fa
./ipknot test.fa
rm test.fa
################################################################################
# SPOT-RNA
################################################################################
echo "Install SPOT-RNA"
git clone https://github.com/jaswindersingh2/SPOT-RNA.git
cd SPOT-RNA
wget 'https://www.dropbox.com/s/dsrcf460nbjqpxa/SPOT-RNA-models.tar.gz' || wget -O SPOT-RNA-models.tar.gz 'https://app.nihaocloud.com/f/fbf3315a91d542c0bdc2/?dl=1'
tar -xvzf SPOT-RNA-models.tar.gz && rm SPOT-RNA-models.tar.gz
cd ..
##############################################################################
# MXFOLD2
##############################################################################
mkdir MXFold2
cd MXFold2
wget https://github.com/mxfold/mxfold2/releases/download/v0.1.2/mxfold2-0.1.2.tar.gz
# wget https://github.com/mxfold/mxfold2/releases/download/v0.1.2/mxfold2-0.1.2-cp310-cp310-manylinux_2_17_x86_64.whl
### MAC
# wget https://github.com/mxfold/mxfold2/releases/download/v0.1.2/mxfold2-0.1.2-cp310-cp310-macosx_13_0_arm64.whl

# MAKE SURE RnaBench is activated!!
pip install mxfold2-0.1.2.tar.gz

wget https://github.com/mxfold/mxfold2/releases/download/v0.1.0/models-0.1.0.tar.gz
tar -zxvf models-0.1.0.tar.gz
cd ..
