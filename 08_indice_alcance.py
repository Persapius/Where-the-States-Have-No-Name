import numpy as np
import pandas as pd
import scipy.sparse as sp

import comum
import config as C

N_NULAS = 100


def alcance(A, part, uf, xy):
    A = sp.triu(A, 1).tocoo()
    d = np.hypot(*(xy[A.row] - xy[A.col]).T)
    w = A.data * d
    linhas = []
    for c in np.unique(part):
        intra = (part[A.row] == c) & (part[A.col] == c)
        if not intra.any():
            continue
        forca = pd.Series(np.r_[A.data[intra], A.data[intra]]).groupby(
            np.r_[uf[A.row[intra]], uf[A.col[intra]]]).sum()
        sede = forca.idxmax()
        dentro = intra & (uf[A.row] == sede) & (uf[A.col] == sede)
        linhas.append({"comunidade": int(c), "municipios": int((part == c).sum()), "sede": sede,
                       "alcance": 1 - w[dentro].sum() / w[intra].sum()})
    return pd.DataFrame(linhas).sort_values("municipios", ascending=False)


def main():
    base = comum.carregar_base()
    geo, contig = comum.carregar_geo()
    xy = geo[["x_km", "y_km"]].to_numpy()
    uf = geo["UF"].to_numpy()
    pe = pd.read_csv(C.DIR_SAIDAS / "comunidades_pertenca.csv").set_index("codigo_ibge").loc[geo["codigo_ibge"]]
    rng = np.random.default_rng(C.SEMENTE_GLOBAL)
    por_com, resumo = [], []
    for ano in C.EXERCICIOS:
        A = comum.rede_knn(comum.matriz_indicadores(base, geo, ano), C.K_PADRAO)
        part = pe[f"{ano}_k{C.K_PADRAO}_ng"].to_numpy()
        obs = alcance(A, part, uf, xy)
        por_com.append(obs.assign(exercicio=ano))
        tam = np.bincount(part)
        nulos = np.array([alcance(A, comum.particao_contigua_aleatoria(contig, tam, xy, rng), uf, xy)["alcance"].mean()
                          for _ in range(N_NULAS)])
        resumo.append({"exercicio": ano, "alcance_medio": obs["alcance"].mean(),
                       "alcance_minimo": obs["alcance"].min(),
                       "nulo_contiguo_media": nulos.mean(), "nulo_p2.5": np.quantile(nulos, 0.025),
                       "nulo_p97.5": np.quantile(nulos, 0.975), "nulo_max": nulos.max(),
                       "n_nulas_maior_ou_igual": int((nulos >= obs["alcance"].mean()).sum())})
        print(resumo[-1])
    pd.concat(por_com).to_csv(C.DIR_SAIDAS / "indice_alcance_comunidades.csv", index=False)
    pd.DataFrame(resumo).to_csv(C.DIR_SAIDAS / "indice_alcance_resumo.csv", index=False)


if __name__ == "__main__":
    main()
