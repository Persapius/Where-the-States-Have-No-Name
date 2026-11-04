import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score

import comum
import config as C


def conjunto_particoes(X, F, rng):
    p = X.shape[1]
    particoes = []
    for k in C.KS:
        for s in range(C.FRONTEIRAS_SEMENTES_POR_K):
            Xb = X[:, rng.choice(p, p, replace=True)] if C.FRONTEIRAS_BOOTSTRAP else X
            A = comum.rede_knn(Xb, k)
            P = comum.nulo_espacial(A, F)[0] if C.FRONTEIRAS_NULO == "espacial" else comum.nulo_ng(A)
            B = comum.matriz_modularidade(A, P)
            particoes.append(comum.louvain_generalizado(B, semente=int(rng.integers(1 << 30))))
            print(f"   k={k}, repetição {s + 1}: {particoes[-1].max() + 1} comunidades")
    return np.array(particoes)


def porosidade(A, P, uf):
    linhas = []
    Ad = A.toarray()
    for u in np.unique(uf):
        dentro = uf == u
        obs = Ad[dentro][:, ~dentro].sum() / Ad[dentro].sum()
        esp = P[dentro][:, ~dentro].sum() / P[dentro].sum()
        linhas.append({"UF": u, "municipios": int(dentro.sum()),
                       "fracao_cruza_obs": obs, "fracao_cruza_esperada": esp,
                       "porosidade_obs_sobre_esp": obs / esp})
    return pd.DataFrame(linhas).sort_values("porosidade_obs_sobre_esp")


def segmentos_fronteira(g, pares):
    import shapely
    gi = g.geometry.iloc[pares["i"].to_numpy()].to_numpy()
    gj = g.geometry.iloc[pares["j"].to_numpy()].to_numpy()
    inter = shapely.intersection(gi, gj)
    partes, idx = shapely.get_parts(inter, return_index=True)
    linear = shapely.get_type_id(partes) == 1
    partes, idx = partes[linear], idx[linear]
    s = pares["s"].to_numpy()[idx]
    coords = [np.asarray(p.coords)[:, :2] for p in partes]
    return coords, s


def mapa_fronteiras(malha, pares, ano, arquivo):
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.collections import LineCollection
    from matplotlib.colors import Normalize
    g = malha.to_crs(C.CRS_METRICO).reset_index(drop=True)
    coords, s = segmentos_fronteira(g, pares)
    ordem = np.argsort(s)
    coords = [coords[i] for i in ordem]
    s = s[ordem]
    cmap, norm = plt.get_cmap("magma_r"), Normalize(0, 1)
    fig, ax = plt.subplots(figsize=(9, 9))
    g.plot(ax=ax, color="#f4f4f4", linewidth=0)
    ax.add_collection(LineCollection(coords, colors=cmap(norm(s)), linewidths=0.3 + 1.5 * s))
    g.dissolve("UF").boundary.plot(ax=ax, color="#2b8cbe", linewidth=0.6)
    fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, shrink=0.6,
                 label="significância da fronteira (s)")
    ax.set_title(f"Fronteiras socioeconômicas efetivas, exercício {ano} (azul: divisas de UF)",
                 fontsize=10)
    ax.set_axis_off()
    fig.savefig(arquivo, dpi=250, bbox_inches="tight")
    plt.close(fig)


def main():
    C.DIR_SAIDAS.mkdir(parents=True, exist_ok=True)
    base = comum.carregar_base()
    geo, contig = comum.carregar_geo()
    malha = comum.carregar_malha()
    D = comum.matriz_distancias(geo)
    F = comum.faixas_distancia(D, C.N_BINS_PADRAO)
    uf = geo["UF"].to_numpy()
    c = sp.triu(contig, 1).tocoo()

    resumo = []
    for ano in C.EXERCICIOS:
        print(f"\n== Exercício {ano}")
        rng = np.random.default_rng(C.SEMENTE_GLOBAL + ano)
        X = comum.matriz_indicadores(base, geo, ano)
        parts = conjunto_particoes(X, F, rng)
        s = (parts[:, c.row] != parts[:, c.col]).mean(0)
        inter = uf[c.row] != uf[c.col]
        pares = pd.DataFrame({"i": c.row, "j": c.col,
                              "codigo_i": geo["codigo_ibge"].to_numpy()[c.row],
                              "codigo_j": geo["codigo_ibge"].to_numpy()[c.col],
                              "UF_i": uf[c.row], "UF_j": uf[c.col],
                              "interestadual": inter, "s": s})
        pares.to_csv(C.DIR_SAIDAS / f"fronteiras_{ano}.csv", index=False)

        auc = roc_auc_score(inter, s)
        rng_p = np.random.default_rng(C.SEMENTE_GLOBAL)
        nulos = np.array([roc_auc_score(rng_p.permutation(inter), s) for _ in range(1000)])
        resumo.append({"exercicio": ano, "particoes": len(parts),
                       "s_medio_interestadual": s[inter].mean(),
                       "s_medio_intraestadual": s[~inter].mean(),
                       "auc_s_separa_divisas": auc,
                       "p_permutacao": (1 + (nulos >= auc).sum()) / 1001})
        print(resumo[-1])

        A = comum.rede_knn(X, C.K_PADRAO)
        P = comum.nulo_espacial(A, F)[0]
        por = porosidade(A, P, uf)
        por.to_csv(C.DIR_SAIDAS / f"porosidade_uf_{ano}.csv", index=False)
        print(por.to_string(index=False))

        try:
            mapa_fronteiras(malha, pares, ano, C.DIR_SAIDAS / f"mapa_fronteiras_{ano}.png")
        except Exception as e:
            print(f"ATENÇÃO: mapa de {ano} não gerado ({type(e).__name__}: {e}). "
                  f"Rode `python 04_fronteiras.py --so-mapas` depois de corrigir.")

    pd.DataFrame(resumo).to_csv(C.DIR_SAIDAS / "fronteiras_resumo.csv", index=False)


def so_mapas():
    malha = comum.carregar_malha()
    for ano in C.EXERCICIOS:
        arq = C.DIR_SAIDAS / f"fronteiras_{ano}.csv"
        if not arq.exists():
            print(f"{arq.name} não existe; rode o script completo.")
            continue
        mapa_fronteiras(malha, pd.read_csv(arq), ano, C.DIR_SAIDAS / f"mapa_fronteiras_{ano}.png")
        print(f"mapa_fronteiras_{ano}.png gerado")


if __name__ == "__main__":
    import sys
    so_mapas() if "--so-mapas" in sys.argv else main()
