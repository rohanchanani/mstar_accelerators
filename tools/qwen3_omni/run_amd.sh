#!/usr/bin/env bash
# One target command: environment, target gates, assets, qualification and service.
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 MSTAR_CHECKOUT WORK_DIRECTORY [ROCM_DIRECTORY] [run_foundation.py options]" >&2
  exit 2
fi
mstar_checkout=$(realpath "$1")
mkdir -p "$2"
work_directory=$(realpath "$2")
shift 2
rocm_directory=/opt/rocm-7.2.0
if [[ $# -gt 0 && "$1" != --* ]]; then
  rocm_directory=$1
  shift
fi
tool_directory=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
bootstrap_log="$work_directory/bootstrap-$(date -u +%Y%m%dT%H%M%SZ).log"
bash "$tool_directory/bootstrap_rocm.sh" "$mstar_checkout" "$work_directory" "$rocm_directory" \
  2>&1 | tee "$bootstrap_log"
source "$work_directory/environment.sh"
exec "$VIRTUAL_ENV/bin/python" "$tool_directory/run_foundation.py" \
  --mstar-root "$mstar_checkout" --work-dir "$work_directory" "$@"
