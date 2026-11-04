import numpy as np
import pandas as pd

import comum
import config as C


def main():
    C.DIR_SAIDAS.mkdir(parents=True, exist_ok=True)
    base = comum.carregar_base()
    geo, contig = comum.carregar_geo()
    Wg = contig / contig.sum()
    linhas = []
    for ano in C.EXERCICIOS:
        X = comum.matriz_indicadores(base, geo, ano)
        A = comum.rede_knn(X, C.K_PADRAO)
        Wa = A / A.sum()
        for alfa in C.ALFAS:
            W = (alfa * Wg + (1 - alfa) * Wa).tocsr()
            B = comum.matriz_modularidade(W, comum.nulo_ng(W))
            part, Q, _, _ = comum.detectar(B, W.sum(), C.SEMENTES, C.SEMENTE_GLOBAL)
            linhas.append({"exercicio": ano, "alfa": alfa, "Q": Q,
                           "n_comunidades": int(part.max() + 1),
                           "ami_UF": comum.ami(geo["UF"], part),
                           "zrand_UF": comum.z_rand(geo["UF"], part),
                           "ami_intermediaria": comum.ami(geo["regiao_intermediaria"], part),
                           "join_count": comum.razao_join_count(contig, part),
                           "cosseno_intra": comum.similaridade_cosseno_intra(X, part)})
            print(linhas[-1])
    res = pd.DataFrame(linhas)
    res.to_csv(C.DIR_SAIDAS / "varredura_alfa.csv", index=False)

    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.5))
    for ax, met, tit in zip(axs, ["ami_UF", "join_count", "cosseno_intra"],
                            ["AMI com UF", "Razão de join count", "Cosseno intracomunidade"]):
        for ano, g in res.groupby("exercicio"):
            ax.plot(g["alfa"], g[met], marker="o", label=str(ano))
        ax.set_xlabel("alfa (0 = só indicadores, 1 = só contiguidade)")
        ax.set_title(tit, fontsize=10)
    axs[0].legend(title="exercício")
    fig.tight_layout()
    fig.savefig(C.DIR_SAIDAS / "varredura_alfa.png", dpi=200)


if __name__ == "__main__":
    main()
