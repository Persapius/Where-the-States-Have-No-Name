#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs
inicio=${1:-01}
for s in 01_preparar_dados 02_validacao_sintetica 03_comunidades 04_fronteiras 05_varredura_alpha 06_robustez 07_figuras 08_indice_alcance 09_numeros_texto; do
  [[ "${s:0:2}" < "$inicio" ]] && continue
  echo "== $s: início $(date '+%d/%m %H:%M')"
  python3 -u "$s.py" > "logs/$s.log" 2>&1
  echo "== $s: fim $(date '+%d/%m %H:%M')"
done
