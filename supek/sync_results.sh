#!/bin/bash
# Pull the results database down from Supek. Run on the LAPTOP.
#
#     ./supek/sync_results.sh                    # -> ./results.sqlite
#     ./supek/sync_results.sh ~/work/sweep.sqlite
#
# This is the only link between the cluster and the dashboard: the dashboard is
# a passive viewer over whatever file this leaves behind, and never talks to
# Supek itself.
set -euo pipefail

# This runs on the laptop, where config.sh's Lustre paths do not apply, so it
# takes only the login host from there and keeps the rest local.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${SUPEK_USER:=$USER}"
# The remote paths are the cluster's, not this machine's, so LUSTRE_HOME is set
# for the remote user before config.sh derives RESULTS_DB from it.
export LUSTRE_HOME="/lustre/home/${SUPEK_USER}"
# shellcheck source=config.sh
source "$HERE/config.sh"
: "${REMOTE_DB:=$RESULTS_DB}"

LOCAL_DB="${1:-results.sqlite}"

echo "Pulling $SUPEK_USER@$SUPEK_LOGIN:$REMOTE_DB -> $LOCAL_DB"
rsync -avh --progress \
    "$SUPEK_USER@$SUPEK_LOGIN:$REMOTE_DB" "$LOCAL_DB"

echo
echo "Runs in $LOCAL_DB:"
python3 - "$LOCAL_DB" <<'EOF'
import sys
from router_lab.store import ResultsStore

with ResultsStore.open(sys.argv[1]) as store:
    for run in store.list_runs():
        print(f"  {run.run_id}  {run.n_results:6d} results  "
              f"datasets={','.join(run.datasets)}  "
              f"approaches={','.join(run.approaches)}")
EOF

echo
echo "Browse it:  streamlit run router_lab/dashboard.py -- --db $LOCAL_DB"
