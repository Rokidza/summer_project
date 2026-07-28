#!/bin/bash
# Build the Apptainer image. Run on a LOGIN NODE - this needs internet.
#
#     ./supek/build_image.sh
#
# Produces $SIF (see config.sh). Takes a while: the base vLLM image is several GB.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=config.sh
source "$HERE/config.sh"

# Large builds need a big temp dir; the default one is on the small OS disk.
export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-/lustre/scratch/$USER/apptainer_tmp}"
export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-/lustre/scratch/$USER/apptainer_cache}"
mkdir -p "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR" "$(dirname "$SIF")"

echo "Building $SIF from $HERE/apptainer/router-lab.def"
echo "  APPTAINER_TMPDIR=$APPTAINER_TMPDIR"

apptainer build --force "$SIF" "$HERE/apptainer/router-lab.def"

echo
echo "Built $SIF ($(du -h "$SIF" | cut -f1))"
echo
echo "Smoke-test it on a GPU node - the login node has no GPU for --nv to expose:"
echo "  qsub -I -q gpu-test -l select=1:ngpus=1:ncpus=8:mem=64gb -l walltime=00:30:00"
echo "  apptainer test --nv $SIF"
