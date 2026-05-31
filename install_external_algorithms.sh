#!/usr/bin/env bash
# One-time setup: extract data tarballs and install external CLI tools that
# cannot be packaged in conda (LinearFold, IpKnot, SPOT-RNA, VARNA).
#
# Run from the repo root AFTER activating the RnaBench conda environment:
#   conda activate RnaBench
#   bash install_external_algorithms.sh

set -euo pipefail
REPO_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "${REPO_DIR}"

################################################################################
# Data tarballs
################################################################################
echo "### Extracting data tarballs"
[ -f data/3D.tar.gz ] && tar -xzf data/3D.tar.gz -C data && echo "  extracted data/3D.tar.gz"
[ -f RnaBench/lib/data/CMs.tar.gz ] && tar -xzf RnaBench/lib/data/CMs.tar.gz -C RnaBench/lib/data && echo "  extracted CMs.tar.gz"

################################################################################
# External algorithm binaries
################################################################################
OUTDIR="${REPO_DIR}/external_algorithms"
mkdir -p "${OUTDIR}"
TESTSEQ="ACGUCGUCAGUCGAUCGAUCGAUCCGCCUAGUCAAAGUCCUCGAAGCUCUCCCUUAG"

################################################################################
# LinearFold
################################################################################
if [ ! -f "${OUTDIR}/LinearFold/linearfold" ]; then
    echo "### Installing LinearFold"
    git clone https://github.com/LinearFold/LinearFold.git "${OUTDIR}/LinearFold"
    make -C "${OUTDIR}/LinearFold"
fi
echo "### Test LinearFold-C"
echo "${TESTSEQ}" | "${OUTDIR}/LinearFold/linearfold"
echo "### Test LinearFold-V"
echo "${TESTSEQ}" | "${OUTDIR}/LinearFold/linearfold" -V

################################################################################
# IpKnot
################################################################################
if [ ! -f "${OUTDIR}/ipknot" ]; then
    echo "### Installing IpKnot"
    curl -sL 'https://github.com/satoken/ipknot/releases/download/v1.1.0/ipknot-1.1.0-x86_64-linux.zip' -o /tmp/ipknot_release.zip
    unzip -p /tmp/ipknot_release.zip ipknot-1.1.0-x86_64-linux/ipknot > "${OUTDIR}/ipknot"
    chmod +x "${OUTDIR}/ipknot"
    rm /tmp/ipknot_release.zip
fi
echo "### Test IpKnot"
printf ">test\n%s\n" "${TESTSEQ}" | "${OUTDIR}/ipknot" /dev/stdin

################################################################################
# SPOT-RNA
################################################################################
if [ ! -d "${OUTDIR}/SPOT-RNA/SPOT-RNA-models" ]; then
    echo "### Installing SPOT-RNA"
    [ -d "${OUTDIR}/SPOT-RNA" ] || git clone https://github.com/jaswindersingh2/SPOT-RNA.git "${OUTDIR}/SPOT-RNA"
    cd "${OUTDIR}/SPOT-RNA"
    wget -q 'https://www.dropbox.com/s/dsrcf460nbjqpxa/SPOT-RNA-models.tar.gz' \
        || wget -qO SPOT-RNA-models.tar.gz 'https://app.nihaocloud.com/f/fbf3315a91d542c0bdc2/?dl=1'
    tar -xzf SPOT-RNA-models.tar.gz && rm SPOT-RNA-models.tar.gz
    cd "${REPO_DIR}"
fi

################################################################################
# VARNA (Java jar + shell wrapper)
################################################################################
VARNA_DIR="${OUTDIR}/VARNA"
mkdir -p "${VARNA_DIR}"
if [ ! -s "${VARNA_DIR}/VARNAv3-93.jar" ]; then
    echo "### Installing VARNA"
    # SPOT-RNA (cloned above) bundles the jar — use it directly
    if [ -f "${OUTDIR}/SPOT-RNA/utils/VARNAv3-93.jar" ]; then
        cp "${OUTDIR}/SPOT-RNA/utils/VARNAv3-93.jar" "${VARNA_DIR}/VARNAv3-93.jar"
    else
        wget -q "https://varna.lri.fr/bin/VARNAv3-93.jar" -O "${VARNA_DIR}/VARNAv3-93.jar"
    fi
fi
# shell wrapper (idempotent)
cat > "${VARNA_DIR}/varna" <<'WRAPPER'
#!/usr/bin/env bash
exec java -cp "$(dirname "$0")/VARNAv3-93.jar" fr.orsay.lri.varna.applications.VARNAcmd "$@"
WRAPPER
chmod +x "${VARNA_DIR}/varna"

################################################################################
# NCBI BLAST (used by data pipeline build_cm.py — optional)
################################################################################
if [ ! -d "${OUTDIR}/ncbi-blast-2.12.0+" ]; then
    echo "### Installing NCBI BLAST"
    wget -q 'https://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/2.12.0/ncbi-blast-2.12.0+-x64-linux.tar.gz' -O /tmp/blast.tar.gz
    tar -xzf /tmp/blast.tar.gz -C "${OUTDIR}" && rm /tmp/blast.tar.gz
fi

echo ""
echo "### install_external_algorithms.sh complete"
echo "### Add the following to your PATH before running benchmarks:"
echo "    ${OUTDIR}/VARNA:${OUTDIR}/LinearFold:${OUTDIR}:\$PATH"
