# Where the States Have No Name

Code for the paper *Where the States Have No Name: Community Structure of Brazilian Municipalities from Aggregated Income Tax Data*.

The pipeline builds similarity networks of the 5,570 Brazilian municipalities from municipal aggregates of individual income tax (IRPF) declarations, detects communities, and tests their agreement with states and IBGE regions against random contiguous partitions and Moran spectral randomization surrogates.

## Data

- `painel_perfil_declarante_irpf.xlsx`: snapshot of the public dashboard *Painel Perfil do Declarante IRPF* (Brazilian Federal Revenue), fiscal years 2025 and 2026, retrieved on 24 October 2026. It contains only the dashboard's municipal aggregates.
- `pareamento_manual.csv`: manual matches between dashboard names and IBGE codes (renamed municipalities).
- The 2022 municipal mesh, the IBGE regional division and the 2022 Census population are downloaded from IBGE by `01_preparar_dados.py` (URLs in `config.py`).

## Requirements

Python 3.10+ and the packages in `requirements.txt`:

```
pip install -r requirements.txt
```

The figures use the TeX Gyre Termes font when available

## Running

```
bash rodar_tudo.sh        # all steps, logs in logs/
```

Each script can also be run on its own (`python3 03_comunidades.py`), in order. Parameters (seeds, k values, number of surrogates, processes) are in `config.py`.

| Script | Content | Paper |
|---|---|---|
| `01_preparar_dados.py` | Matching to IBGE codes, mesh, contiguity, indicators | Sec. III-A, III-B |
| `02_validacao_sintetica.py` | Synthetic benchmark with planted and twin regions | Table 2 |
| `03_comunidades.py` | Similarity networks, consensus partitions, BESTest | Table 3, Sec. IV-B |
| `04_fronteiras.py` | Border ensemble and porosity | Sec. IV-C |
| `05_varredura_alpha.py` | Fusion of contiguity and similarity | Fig. 4a |
| `06_robustez.py` | Resolution, sensitivity, Moran spectral randomization null | Table 4, Figs. 1, 2, 4b |
| `07_figuras.py` | Figures and table bodies for the paper | Figs. 1–4, Tables 2–5 |
| `08_indice_alcance.py` | Outreach index and its contiguous null | Sec. IV-C |
| `09_numeros_texto.py` | Other numbers quoted in the text | `saidas/numeros_texto.csv` |

Outputs go to `saidas/` (figures and tables for the paper in `saidas/artigo/`); intermediate data and caches go to `dados/`.

## Notes

- Results are deterministic given the seeds in `config.py` (`SEMENTE_GLOBAL = 42`, 100 Louvain runs per configuration).
- Consensus clustering follows Lancichinetti and Fortunato (2012). In the main partitions (NG null model, k = 6, 9, 12), municipalities left without co-assignments above the threshold are reconnected to their most frequent partners; in the other analyses they are kept as singletons.
- Variable names and console messages are in Portuguese.
