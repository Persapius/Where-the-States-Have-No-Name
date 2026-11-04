from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.neighbors import NearestNeighbors

import config as C


def carregar_base():
    caminho = C.DIR_DADOS / "base_municipios.parquet"
    if not caminho.exists():
        raise SystemExit("Rode 01_preparar_dados.py antes.")
    return pd.read_parquet(caminho)


def carregar_geo():
    geo = pd.read_parquet(C.DIR_DADOS / "geo_municipios.parquet")
    contig = sp.load_npz(C.DIR_DADOS / "contiguidade.npz").tocsr()
    return geo, contig


def carregar_malha():
    import geopandas as gpd
    return gpd.read_parquet(C.DIR_DADOS / "malha_municipios.parquet")


def matriz_indicadores(base: pd.DataFrame, geo: pd.DataFrame, exercicio: int,
                       indicadores: dict | None = None) -> np.ndarray:
    indicadores = indicadores or C.INDICADORES
    df = base[base["Exercício"] == exercicio].set_index("codigo_ibge")
    df = df.reindex(geo["codigo_ibge"].values)
    faltantes = df.index[df[list(indicadores)].isna().any(axis=1)]
    if len(faltantes):
        raise ValueError(f"{len(faltantes)} municípios sem indicadores em {exercicio}: "
                         f"{list(faltantes[:5])}...")
    colunas = []
    for nome, transf in indicadores.items():
        v = df[nome].astype(float).to_numpy()
        if C.WINSORIZAR:
            lo, hi = np.quantile(v, C.WINSORIZAR)
            v = np.clip(v, lo, hi)
        if transf == "log":
            if (v <= 0).any():
                raise ValueError(f"{nome} tem valores <= 0; use 'log1p'.")
            v = np.log(v)
        elif transf == "log1p":
            v = np.log1p(np.clip(v, 0, None))
        colunas.append(v)
    X = np.column_stack(colunas)
    return (X - X.mean(0)) / X.std(0)


def rede_knn(X: np.ndarray, k: int, modo: str = C.KNN_MODO) -> sp.csr_matrix:
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
    dist, idx = nn.kneighbors(X)
    dist, idx = dist[:, 1:], idx[:, 1:]
    sigma = np.median(dist[:, -1])
    n = X.shape[0]
    linhas = np.repeat(np.arange(n), k)
    viz = sp.csr_matrix((np.ones(n * k), (linhas, idx.ravel())), shape=(n, n))
    if modo == "mutuo":
        mascara = viz.multiply(viz.T)
    elif modo == "uniao":
        mascara = ((viz + viz.T) > 0).astype(float)
    else:
        raise ValueError(modo)
    mascara = sp.triu(mascara, 1).tocoo()
    d2 = ((X[mascara.row] - X[mascara.col]) ** 2).sum(1)
    w = np.maximum(np.exp(-d2 / sigma ** 2), 1e-8)
    A = sp.coo_matrix((w, (mascara.row, mascara.col)), shape=(n, n))
    return (A + A.T).tocsr()


def matriz_distancias(geo: pd.DataFrame) -> np.ndarray:
    xy = geo[["x_km", "y_km"]].to_numpy()
    diff = xy[:, None, :] - xy[None, :, :]
    return np.sqrt((diff ** 2).sum(-1)).astype(np.float32)


def faixas_distancia(D: np.ndarray, n_bins: int) -> np.ndarray:
    iu = np.triu_indices_from(D, 1)
    cortes = np.quantile(D[iu], np.linspace(0, 1, n_bins + 1)[1:-1])
    F = np.searchsorted(cortes, D).astype(np.int16)
    np.fill_diagonal(F, -1)
    return F


def nulo_ng(A: sp.csr_matrix) -> np.ndarray:
    k = np.asarray(A.sum(1)).ravel()
    return np.outer(k, k) / k.sum()


def nulo_espacial(A: sp.csr_matrix, F: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n_bins = F.max() + 1
    Ad = A.toarray()
    validos = F >= 0
    soma = np.bincount(F[validos], weights=Ad[validos], minlength=n_bins)
    cont = np.bincount(F[validos], minlength=n_bins)
    f = np.divide(soma, cont, out=np.zeros(n_bins), where=cont > 0)
    P = np.where(validos, f[np.clip(F, 0, None)], 0.0)
    return P, f


def matriz_modularidade(A: sp.csr_matrix, P: np.ndarray) -> np.ndarray:
    B = A.toarray() - P
    return (B + B.T) / 2


def _movimento_local(B: np.ndarray, rng: np.random.Generator, tol: float = 1e-12):
    n = B.shape[0]
    comm = np.arange(n)
    S = B.copy()
    diag = np.diag(B).copy()
    houve_mudanca = False
    melhorou = True
    while melhorou:
        melhorou = False
        for i in rng.permutation(n):
            a = comm[i]
            linha = S[i].copy()
            linha[a] -= diag[i]
            b = int(np.argmax(linha))
            if linha[b] > linha[a] + tol:
                comm[i] = b
                col = B[:, i]
                S[:, a] -= col
                S[:, b] += col
                melhorou = True
                houve_mudanca = True
    _, comm = np.unique(comm, return_inverse=True)
    return comm, houve_mudanca


def louvain_generalizado(B: np.ndarray, semente: int = 0, max_niveis: int = 50) -> np.ndarray:
    rng = np.random.default_rng(semente)
    n = B.shape[0]
    pertenca = np.arange(n)
    Bc = B
    for _ in range(max_niveis):
        comm, mudou = _movimento_local(Bc, rng)
        if not mudou:
            break
        pertenca = comm[pertenca]
        K = comm.max() + 1
        M = sp.csr_matrix((np.ones(len(comm)), (np.arange(len(comm)), comm)),
                          shape=(len(comm), K))
        Bc = np.asarray((M.T @ (M.T @ Bc).T))
        Bc = (Bc + Bc.T) / 2
    _, pertenca = np.unique(pertenca, return_inverse=True)
    return pertenca


def modularidade(B: np.ndarray, pertenca: np.ndarray, peso_total: float) -> float:
    K = pertenca.max() + 1
    M = sp.csr_matrix((np.ones(len(pertenca)), (np.arange(len(pertenca)), pertenca)),
                      shape=(len(pertenca), K))
    return float((M.T @ (M.T @ B).T).trace() / peso_total)


def melhor_de_varias(B: np.ndarray, peso_total: float, sementes: int, semente0: int = 0):
    resultados = []
    for s in range(sementes):
        p = louvain_generalizado(B, semente=semente0 + s)
        resultados.append((modularidade(B, p, peso_total), p))
    resultados.sort(key=lambda t: -t[0])
    return resultados[0][1], resultados[0][0], [p for _, p in resultados]


def matriz_coassociacao(particoes) -> np.ndarray:
    n = len(particoes[0])
    D = np.zeros((n, n), dtype=np.float32)
    for p in particoes:
        M = sp.csr_matrix((np.ones(n, dtype=np.float32), (np.arange(n), p)),
                          shape=(n, int(p.max()) + 1))
        D += (M @ M.T).toarray()
    D /= len(particoes)
    return D


def consenso(particoes, tau: float, rodadas: int, max_iter: int, semente0: int = 0,
             reconectar: bool = False):
    P = [np.asarray(p) for p in particoes]
    for it in range(1, max_iter + 1):
        if all(np.array_equal(pd.factorize(P[0])[0], pd.factorize(p)[0]) for p in P[1:]):
            return P[0], True, it - 1
        D = matriz_coassociacao(P)
        np.fill_diagonal(D, 0)
        Dt = np.where(D >= tau, D, 0).astype(D.dtype)
        if reconectar:
            for i in np.flatnonzero(Dt.sum(1) == 0):
                maior = D[i].max()
                if maior > 0:
                    j = np.flatnonzero(D[i] == maior)
                    Dt[i, j] = D[i, j]
                    Dt[j, i] = D[j, i]
        A = sp.csr_matrix(Dt.astype(np.float64))
        B = matriz_modularidade(A, nulo_ng(A))
        P = [louvain_generalizado(B, semente=semente0 + 1000 * it + s) for s in range(rodadas)]
    media = [np.mean([ami(p, q) for q in P if q is not p]) for p in P]
    return P[int(np.argmax(media))], False, max_iter


def detectar(B: np.ndarray, peso_total: float, sementes: int, semente0: int = 0,
             reconectar: bool = False):
    melhor, Q_melhor, todas = melhor_de_varias(B, peso_total, sementes, semente0)
    info = {"Q_melhor_semente": Q_melhor}
    if C.METODO_PARTICAO == "melhor_q":
        return melhor, Q_melhor, todas, info
    if C.METODO_PARTICAO != "consenso":
        raise ValueError(C.METODO_PARTICAO)
    part, convergiu, iteracoes = consenso(todas, C.CONSENSO_TAU, sementes,
                                          C.CONSENSO_MAX_ITER, semente0, reconectar)
    _, part = np.unique(part, return_inverse=True)
    info.update({"consenso_convergiu": convergiu, "consenso_iteracoes": iteracoes})
    return part, modularidade(B, part, peso_total), todas, info


def ami(a, b) -> float:
    return float(adjusted_mutual_info_score(a, b))


def z_rand(a, b) -> float:
    a = pd.factorize(np.asarray(a))[0]
    b = pd.factorize(np.asarray(b))[0]
    n = len(a)
    nij = sp.coo_matrix((np.ones(n), (a, b))).tocsr()
    ni = np.asarray(nij.sum(1)).ravel()
    nj = np.asarray(nij.sum(0)).ravel()
    M = n * (n - 1) / 2
    M1 = (ni ** 2 - ni).sum() / 2
    M2 = (nj ** 2 - nj).sum() / 2
    v = nij.data
    w11 = (v ** 2 - v).sum() / 2
    media = M1 * M2 / M
    C1 = 4 * (ni ** 3).sum() - 8 * (n + 1) * M1 + n * (n ** 2 - 3 * n - 2)
    C2 = 4 * (nj ** 3).sum() - 8 * (n + 1) * M2 + n * (n ** 2 - 3 * n - 2)
    var = (M / 16
           - (4 * M1 - 2 * M) ** 2 * (4 * M2 - 2 * M) ** 2 / (256 * M ** 2)
           + C1 * C2 / (16 * n * (n - 1) * (n - 2))
           + ((4 * M1 - 2 * M) ** 2 - 4 * C1 - 4 * M) * ((4 * M2 - 2 * M) ** 2 - 4 * C2 - 4 * M)
           / (64 * n * (n - 1) * (n - 2) * (n - 3)))
    return float((w11 - media) / np.sqrt(var))


def logvero_dcsbm(A_bin: sp.csr_matrix, rotulos: np.ndarray) -> float:
    g = pd.factorize(np.asarray(rotulos))[0]
    K = g.max() + 1
    M = sp.csr_matrix((np.ones(len(g)), (np.arange(len(g)), g)), shape=(len(g), K))
    m_rs = (M.T @ A_bin @ M).toarray()
    kappa = m_rs.sum(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        termo = m_rs * np.log(m_rs / np.outer(kappa, kappa))
    return float(np.nansum(termo))


def bestest(A: sp.csr_matrix, rotulos, n_perm: int, semente: int = 0) -> dict:
    rng = np.random.default_rng(semente)
    A_bin = (A > 0).astype(float).tocsr()
    rotulos = np.asarray(rotulos)
    obs = logvero_dcsbm(A_bin, rotulos)
    nulos = np.array([logvero_dcsbm(A_bin, rng.permutation(rotulos)) for _ in range(n_perm)])
    p = (1 + (nulos >= obs).sum()) / (1 + n_perm)
    return {"logvero_obs": obs, "logvero_nulo_media": float(nulos.mean()),
            "logvero_nulo_dp": float(nulos.std()),
            "z": float((obs - nulos.mean()) / nulos.std()), "p": float(p)}


def particao_contigua_aleatoria(contig: sp.csr_matrix, tamanhos, xy: np.ndarray,
                                rng: np.random.Generator) -> np.ndarray:
    n = contig.shape[0]
    alvo = rng.permutation(np.asarray(tamanhos))
    K = len(alvo)
    viz = [contig.indices[contig.indptr[i]:contig.indptr[i + 1]] for i in range(n)]
    rot = -np.ones(n, dtype=int)
    tam = np.ones(K, dtype=int)
    sementes = rng.choice(n, K, replace=False)
    rot[sementes] = np.arange(K)
    fronteira = [list(viz[s]) for s in sementes]
    ativos = [r for r in range(K) if tam[r] < alvo[r]]
    while ativos:
        deficit = alvo[ativos] - tam[ativos]
        j = int(rng.choice(len(ativos), p=deficit / deficit.sum()))
        r = ativos[j]
        fr = fronteira[r]
        v = -1
        while fr:
            t = int(rng.integers(len(fr)))
            cand = fr[t]
            fr[t] = fr[-1]
            fr.pop()
            if rot[cand] < 0:
                v = cand
                break
        if v < 0 or tam[r] + 1 >= alvo[r]:
            ativos.pop(j)
        if v >= 0:
            rot[v] = r
            tam[r] += 1
            fr.extend(viz[v])
    livres = np.flatnonzero(rot < 0)
    while len(livres):
        mudou = False
        for v in rng.permutation(livres):
            rotulados = [u for u in viz[v] if rot[u] >= 0]
            if rotulados:
                rot[v] = rot[rotulados[int(rng.integers(len(rotulados)))]]
                mudou = True
        livres = np.flatnonzero(rot < 0)
        if not mudou:
            rotulados = np.flatnonzero(rot >= 0)
            for v in livres:
                d = ((xy[rotulados] - xy[v]) ** 2).sum(1)
                rot[v] = rot[rotulados[int(np.argmin(d))]]
            break
    return rot


def particoes_contiguas(contig, rotulos, xy, n: int, semente: int, cache=None) -> np.ndarray:
    if cache is not None and cache.exists():
        P = np.load(cache)
        if (P.shape[0] >= n and P.shape[1] == len(rotulos)
                and P.max() + 1 == pd.Series(rotulos).nunique()):
            return P[:n]
        print(f"   cache {cache.name} desatualizado; gerando de novo")
    rng = np.random.default_rng(semente)
    tamanhos = pd.Series(rotulos).value_counts().to_numpy()
    P = np.array([particao_contigua_aleatoria(contig, tamanhos, xy, rng) for _ in range(n)])
    if cache is not None:
        np.save(cache, P)
    return P


def bestest_contiguo(A: sp.csr_matrix, rotulos, nulas: np.ndarray) -> dict:
    A_bin = (A > 0).astype(float).tocsr()
    obs = logvero_dcsbm(A_bin, np.asarray(rotulos))
    nulos = np.array([logvero_dcsbm(A_bin, p) for p in nulas])
    p = (1 + (nulos >= obs).sum()) / (1 + len(nulos))
    return {"logvero_obs": obs, "logvero_nulo_media": float(nulos.mean()),
            "logvero_nulo_dp": float(nulos.std()),
            "z": float((obs - nulos.mean()) / nulos.std()), "p": float(p),
            "percentil_obs": float((nulos < obs).mean())}


def razao_join_count(contig: sp.csr_matrix, pertenca: np.ndarray) -> float:
    c = sp.triu(contig, 1).tocoo()
    return float((pertenca[c.row] == pertenca[c.col]).mean())


def similaridade_cosseno_intra(X: np.ndarray, pertenca: np.ndarray) -> float:
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
    soma, pares = 0.0, 0
    for c in np.unique(pertenca):
        idx = np.flatnonzero(pertenca == c)
        if len(idx) < 2:
            continue
        G = Xn[idx] @ Xn[idx].T
        soma += (G.sum() - len(idx)) / 2
        pares += len(idx) * (len(idx) - 1) // 2
    return soma / pares if pares else float("nan")


def mapa_comunidades(malha, pertenca: np.ndarray, titulo: str, arquivo):
    import matplotlib.pyplot as plt
    g = malha.copy()
    g["comunidade"] = pertenca
    tam = g["comunidade"].value_counts()
    grandes = tam.index[:20]
    g["cor"] = np.where(g["comunidade"].isin(grandes), g["comunidade"], -1)
    fig, ax = plt.subplots(figsize=(9, 9))
    g[g["cor"] >= 0].plot(column="cor", categorical=True, cmap="tab20", ax=ax, linewidth=0)
    if (g["cor"] < 0).any():
        g[g["cor"] < 0].plot(color="#dddddd", ax=ax, linewidth=0)
    g.dissolve("UF").boundary.plot(ax=ax, color="black", linewidth=0.4)
    ax.set_title(titulo, fontsize=10)
    ax.set_axis_off()
    fig.savefig(arquivo, dpi=200, bbox_inches="tight")
    plt.close(fig)
