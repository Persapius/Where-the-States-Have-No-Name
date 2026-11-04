import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from sklearn.cluster import AgglomerativeClustering

import comum
import config as C


def plantar_regioes(geo, contig, rng):
    xy = geo[["x_km", "y_km"]].to_numpy()
    agl = AgglomerativeClustering(n_clusters=C.SINT_N_REGIOES, linkage="ward",
                                  connectivity=contig)
    reg = agl.fit_predict(xy)
    centro = np.array([xy[reg == r].mean(0) for r in range(C.SINT_N_REGIOES)])
    D = np.linalg.norm(centro[:, None] - centro[None], axis=-1)
    livres = set(range(C.SINT_N_REGIOES))
    gemeas = []
    for i, j in zip(*np.unravel_index(np.argsort(-D, axis=None), D.shape)):
        if len(gemeas) == C.SINT_N_GEMEAS:
            break
        if i < j and i in livres and j in livres:
            gemeas.append((int(i), int(j)))
            livres -= {i, j}
    rotulo = reg.copy()
    for i, j in gemeas:
        rotulo[reg == j] = i
    return reg, rotulo, gemeas


def gerar_indicadores(rotulo, contig, rho, rng):
    n = len(rotulo)
    grau = np.asarray(contig.sum(1)).ravel()
    W = sp.diags(np.divide(1.0, grau, out=np.zeros(n), where=grau > 0)) @ contig
    rotulos = np.unique(rotulo)
    mu = rng.normal(0, C.SINT_SEPARACAO, size=(rotulos.max() + 1, C.SINT_N_INDICADORES))
    Z = mu[rotulo] + rng.normal(0, 1, size=(n, C.SINT_N_INDICADORES))
    if rho > 0:
        Z = spla.splu((sp.eye(n) - rho * W).tocsc()).solve(Z)
    return (Z - Z.mean(0)) / Z.std(0)


def recuperacao_gemeas(pertenca, reg, gemeas):
    acertos = []
    for i, j in gemeas:
        moda_i = np.bincount(pertenca[reg == i]).argmax()
        moda_j = np.bincount(pertenca[reg == j]).argmax()
        acertos.append(moda_i == moda_j)
    return float(np.mean(acertos))


def main():
    C.DIR_SAIDAS.mkdir(parents=True, exist_ok=True)
    geo, contig = comum.carregar_geo()
    D = comum.matriz_distancias(geo)
    F = comum.faixas_distancia(D, C.N_BINS_PADRAO)
    linhas = []
    for rep in range(C.SINT_REPETICOES):
        rng = np.random.default_rng(C.SEMENTE_GLOBAL + rep)
        reg, rotulo, gemeas = plantar_regioes(geo, contig, rng)
        for rho in C.SINT_RHOS:
            X = gerar_indicadores(rotulo, contig, rho, rng)
            A = comum.rede_knn(X, C.K_PADRAO)
            m2 = A.sum()
            nulos = {"ng": comum.nulo_ng(A), "espacial": comum.nulo_espacial(A, F)[0]}
            for nome, P in nulos.items():
                B = comum.matriz_modularidade(A, P)
                part, Q, _, _ = comum.detectar(B, m2, C.SEMENTES, C.SEMENTE_GLOBAL)
                linhas.append({
                    "repeticao": rep, "rho": rho, "nulo": nome, "Q": Q,
                    "n_comunidades": int(part.max() + 1),
                    "ami_plantado": comum.ami(rotulo, part),
                    "ami_uf": comum.ami(geo["UF"], part),
                    "recuperacao_gemeas": recuperacao_gemeas(part, reg, gemeas),
                    "join_count": comum.razao_join_count(contig, part),
                })
                print(linhas[-1])
    res = pd.DataFrame(linhas)
    res.to_csv(C.DIR_SAIDAS / "validacao_sintetica.csv", index=False)
    resumo = res.groupby(["rho", "nulo"])[["ami_plantado", "recuperacao_gemeas", "join_count",
                                           "n_comunidades"]].agg(["mean", "std"]).round(3)
    print(resumo)
    resumo.to_csv(C.DIR_SAIDAS / "validacao_sintetica_resumo.csv")

    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.5))
    for ax, met, tit in zip(axs, ["ami_plantado", "recuperacao_gemeas"],
                            ["AMI com regiões plantadas", "Gêmeas recuperadas (fração)"]):
        for nome, g in res.groupby("nulo"):
            m = g.groupby("rho")[met].agg(["mean", "std"])
            ax.errorbar(m.index, m["mean"], yerr=m["std"], marker="o", capsize=3, label=nome)
        ax.set_xlabel("rho (autocorrelação espacial)")
        ax.set_title(tit, fontsize=10)
        ax.set_ylim(0, 1.05)
    axs[0].legend(title="nulo")
    fig.tight_layout()
    fig.savefig(C.DIR_SAIDAS / "validacao_sintetica.png", dpi=200)


if __name__ == "__main__":
    main()
