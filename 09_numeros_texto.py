import importlib.util
import sys

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from scipy.stats import spearmanr
from sklearn.metrics import adjusted_rand_score

import comum
import config as C

METADADOS = {"UF": "UF", "intermediaria": "regiao_intermediaria", "imediata": "regiao_imediata"}
N_SUBST_EXTRA = 200
SEMENTE_EXTRA = 10_000
GRAFOS_NX = [(10, 100, 0.10, 0.01), (10, 100, 0.08, 0.02), (20, 50, 0.20, 0.02), (5, 200, 0.05, 0.02)]

LINHAS = []


def reg(secao, quantidade, valor, exercicio=None):
    LINHAS.append({"secao": secao, "quantidade": quantidade, "exercicio": exercicio, "valor": valor})
    if isinstance(valor, (float, np.floating)):
        v = f"{valor:.4f}" if valor == 0 or abs(valor) >= 1e-3 else f"{valor:.2e}"
    else:
        v = valor
    print(f"   [{secao}] {quantidade}{f' ({exercicio})' if exercicio else ''}: {v}")


def modulo_robustez():
    spec = importlib.util.spec_from_file_location("robustez", C.RAIZ / "06_robustez.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def dados(base):
    print("\n== Dados (Seção III-A e Limitações)")
    painel = pd.read_excel(C.ARQUIVO_PAINEL, sheet_name=C.ABA_MUNICIPIO, dtype={"Exercício": str})
    painel["Exercício"] = painel["Exercício"].astype(int)
    for ano, g in painel.groupby("Exercício"):
        reg("III-A", "unidades territoriais no painel", len(g), ano)
        reg("III-A", "unidades do exterior (UF = EX)", int(g["UF"].eq("EX").sum()), ano)
        reg("III-A", "declarantes no painel", int(g["Declarantes"].sum()), ano)
        reg("Limitações", "declarantes com renda variável no país (%)",
            100 * float((g["% Renda variável"] * g["Declarantes"]).sum() / g["Declarantes"].sum()), ano)
    par = pd.read_csv(C.DIR_SAIDAS / "pareamento_municipios.csv")
    for metodo, n in par["metodo"].value_counts().items():
        reg("III-A", f"pareamento painel-IBGE: {metodo}", int(n))
    nao = par[par["codigo_ibge"].isna()]
    reg("III-A", "municípios do painel sem código IBGE",
        "; ".join(f"{m} ({u})" for u, m in zip(nao["UF"], nao["Município"])) or "nenhum")
    reg("III-A", "municípios na análise", int(base["codigo_ibge"].nunique()))
    for ano, g in base.groupby("Exercício"):
        reg("III-A", "menor número de declarantes de um município", int(g["Declarantes"].min()), ano)
        reg("III-A", "municípios com patrimônio médio negativo", int((g["Média patrimônio (R$)"] < 0).sum()), ano)
        reg("III-A", "maior despesa médica média (R$)", float(g["Média desp. médicas (R$)"].max()), ano)
        reg("III-A", "percentil 99 da despesa médica média (R$)",
            float(g["Média desp. médicas (R$)"].quantile(0.99)), ano)


def knn_mutuo(base, geo):
    print("\n== kNN mútuo (Seção III-B)")
    for ano in C.EXERCICIOS:
        X = comum.matriz_indicadores(base, geo, ano)
        for k in C.KS:
            A = comum.rede_knn(X, k, modo="mutuo")
            reg("III-B", f"municípios isolados no kNN mútuo, k = {k}",
                int((np.asarray((A > 0).sum(1)).ravel() == 0).sum()), ano)


def sem_winsorizacao(base, geo):
    print("\n== Sem winsorização (Seção III-A)")
    original = C.WINSORIZAR
    C.WINSORIZAR = None
    try:
        for ano in C.EXERCICIOS:
            X = comum.matriz_indicadores(base, geo, ano)
            A = comum.rede_knn(X, C.K_PADRAO)
            B = comum.matriz_modularidade(A, comum.nulo_ng(A))
            part, _, _, _ = comum.detectar(B, A.sum(), C.SEMENTES, C.SEMENTE_GLOBAL)
            b = base[base["Exercício"] == ano].set_index("codigo_ibge").loc[geo["codigo_ibge"]]
            neg = (b["Média patrimônio (R$)"] < 0).to_numpy()
            coms = np.unique(part[neg])
            so_eles = len(coms) == 1 and (part == coms[0]).sum() == neg.sum()
            reg("III-A", "sem winsorização: comunidades que contêm os municípios de patrimônio negativo",
                "; ".join(f"{int((part == c).sum())} municípios, {int((part[neg] == c).sum())} negativos"
                          for c in coms), ano)
            reg("III-A", "sem winsorização: os negativos formam uma comunidade só deles", bool(so_eles), ano)
    finally:
        C.WINSORIZAR = original


def louvain_networkx():
    print("\n== Louvain generalizado x NetworkX (Seção III-C)")
    try:
        import networkx as nx
    except ImportError:
        print("   networkx não instalado; teste pulado")
        return
    dif = []
    for l, n, p_in, p_out in GRAFOS_NX:
        for rep in range(3):
            G = nx.planted_partition_graph(l, n, p_in, p_out, seed=rep)
            A = sp.csr_matrix(nx.to_scipy_sparse_array(G, nodelist=range(l * n), dtype=float))
            B = comum.matriz_modularidade(A, comum.nulo_ng(A))
            nosso = max(comum.modularidade(B, comum.louvain_generalizado(B, semente=s), A.sum())
                        for s in range(10))
            ref = max(nx.community.modularity(G, nx.community.louvain_communities(G, seed=s))
                      for s in range(10))
            dif.append(nosso - ref)
            print(f"   l={l}, n={n}, p_in={p_in}, p_out={p_out}, rep={rep}: Q = {nosso:.4f} (NetworkX {ref:.4f})")
    dif = np.array(dif)
    reg("III-C", "grafos de partição plantada comparados", len(dif))
    reg("III-C", "grafos com a mesma modularidade (|diferença| < 1e-4)", int((np.abs(dif) < 1e-4).sum()))
    reg("III-C", "maior diferença Q(nosso) - Q(NetworkX)", float(dif.max()))
    reg("III-C", "menor diferença Q(nosso) - Q(NetworkX)", float(dif.min()))


def particoes_contiguas(geo, contig):
    print("\n== Partições contíguas aleatórias (Seção III-E)")
    n = contig.shape[0]
    xy = geo[["x_km", "y_km"]].to_numpy()
    ilha = np.asarray(contig.sum(1)).ravel() == 0
    c = sp.triu(contig, 1).tocoo()
    for nome, col in METADADOS.items():
        P = comum.particoes_contiguas(contig, geo[col].to_numpy(), xy, C.BESTEST_CONTIGUAS, C.SEMENTE_GLOBAL,
                                      cache=C.DIR_DADOS / f"particoes_contiguas_{nome}.npy")
        alvo = np.sort(pd.Series(geo[col]).value_counts().to_numpy())
        desc = np.array([np.abs(np.sort(np.bincount(p, minlength=len(alvo))) - alvo).sum() / 2 / n for p in P])
        desconexas = 0
        for p in P:
            mesma = p[c.row] == p[c.col]
            G = sp.coo_matrix((np.ones(mesma.sum()), (c.row[mesma], c.col[mesma])), shape=(n, n))
            comp = connected_components(G, directed=False)[1]
            desconexas += int((pd.Series(comp[~ilha]).groupby(p[~ilha]).nunique() > 1).sum())
        reg("III-E", f"municípios fora do tamanho-alvo, média (%), {nome}", 100 * float(desc.mean()))
        reg("III-E", f"municípios fora do tamanho-alvo, máximo (%), {nome}", 100 * float(desc.max()))
        reg("III-E", f"regiões desconexas (sem contar ilhas) em {len(P)} partições, {nome}", desconexas)


def substitutos_moran(base, geo, contig, rob):
    print("\n== Substitutos de Moran (Seção III-F)")
    V = rob.autovetores_moran(contig)
    for ano in C.EXERCICIOS:
        X = comum.matriz_indicadores(base, geo, ano)
        p = X.shape[1]
        I0 = np.array([rob.i_moran(X[:, j], contig) for j in range(p)])
        cor, dif_I = [], 0.0
        for i in range(N_SUBST_EXTRA):
            rng = np.random.default_rng([C.SEMENTE_GLOBAL, ano, SEMENTE_EXTRA + i])
            Xs = rob.substituto_moran(X, V, rng)
            cor.append([np.corrcoef(X[:, j], Xs[:, j])[0, 1] for j in range(p)])
            dif_I = max(dif_I, max(abs(rob.i_moran(Xs[:, j], contig) - I0[j]) for j in range(p)))
        cor = np.array(cor)
        reg("III-F", f"correlação média substituto-original em {N_SUBST_EXTRA} substitutos: menor entre indicadores",
            float(cor.mean(0).min()), ano)
        reg("III-F", f"correlação média substituto-original em {N_SUBST_EXTRA} substitutos: maior entre indicadores",
            float(cor.mean(0).max()), ano)
        reg("III-F", "média do |r| substituto-original: menor entre indicadores", float(np.abs(cor).mean(0).min()), ano)
        reg("III-F", "média do |r| substituto-original: maior entre indicadores", float(np.abs(cor).mean(0).max()), ano)
        reg("III-F", "maior |I de Moran do substituto - I original|", float(dif_I), ano)
        nomes = list(C.INDICADORES)
        reg("III-F", "I de Moran dos indicadores: menor", float(I0.min()), ano)
        reg("III-F", "I de Moran dos indicadores: maior", float(I0.max()), ano)
        reg("III-F", "I de Moran da taxa de declarantes", float(I0[nomes.index("taxa_declarantes")]), ano)
        reg("III-F", "I de Moran do patrimônio médio", float(I0[nomes.index("Média patrimônio (R$)")]), ano)


def benjamini_hochberg(p):
    p = np.asarray(p, dtype=float)
    m = len(p)
    o = np.argsort(p)
    ajust = np.minimum.accumulate((p[o] * m / np.arange(1, m + 1))[::-1])[::-1]
    out = np.empty(m)
    out[o] = np.minimum(ajust, 1)
    return out


def porosidade(geo):
    print("\n== Porosidade (Seção IV-C)")
    r = pd.read_csv(C.DIR_SAIDAS / "robustez_moran_resumo.csv")
    p = r[r["metrica"].str.startswith("porosidade_")].copy()
    p["UF"] = p["metrica"].str.replace("porosidade_", "", regex=False)
    p["p_bh"] = benjamini_hochberg(p["p_bilateral"])
    p["abaixo_do_nulo"] = p["observado"] < p["nulo_media"]
    p[["exercicio", "UF", "observado", "nulo_media", "nulo_p2.5", "nulo_p97.5", "p_bilateral", "p_bh",
       "abaixo_do_nulo"]].to_csv(C.DIR_SAIDAS / "porosidade_classificacao.csv", index=False)
    reg("IV-C", "testes de porosidade (UF x exercício)", len(p))
    reg("IV-C", "IC 95% do nulo: menor limite inferior", float(p["nulo_p2.5"].min()))
    reg("IV-C", "IC 95% do nulo: maior limite superior", float(p["nulo_p97.5"].max()))
    anos = sorted(p["exercicio"].unique())
    w = {a: p[p["exercicio"] == a].set_index("UF") for a in anos}
    ufs = sorted(w[anos[0]].index)
    sig = {a: w[a]["p_bilateral"] < 0.05 for a in anos}
    baixo = [u for u in ufs if all(sig[a][u] and w[a].at[u, "abaixo_do_nulo"] for a in anos)]
    baixo_bh = [u for u in baixo if all(w[a].at[u, "p_bh"] < 0.05 for a in anos)]
    um_ano = [u for u in ufs if sum(sig[a][u] for a in anos) == 1]
    nenhum = [u for u in ufs if not any(sig[a][u] for a in anos)]
    opostos = [u for u in ufs if all(sig[a][u] for a in anos)
               and len({bool(w[a].at[u, "abaixo_do_nulo"]) for a in anos}) > 1]
    reg("IV-C", "UFs com porosidade significativamente abaixo do nulo em todos os exercícios", len(baixo))
    reg("IV-C", "dessas, significativas após Benjamini-Hochberg em todos os exercícios", len(baixo_bh))
    reg("IV-C", "UFs significativas em um só exercício", ", ".join(um_ano))
    reg("IV-C", "UFs sem significância em nenhum exercício", ", ".join(nenhum))
    reg("IV-C", "UFs significativas em sentidos opostos", ", ".join(opostos))
    obs = pd.DataFrame({a: w[a]["observado"] for a in anos})
    for a in anos:
        reg("IV-C", "UF mais fechada (menor porosidade)", f"{obs[a].idxmin()} ({obs[a].min():.3f})", a)
        for u in ("SP", "RJ", "MG"):
            reg("IV-C", f"porosidade observada, {u}", float(obs.at[u, a]), a)
    norte = ["AP", "RO", "AC", "RR", "AM"]
    reg("IV-C", "porosidade observada de AP, RO, AC, RR e AM: menor", float(obs.loc[norte].min().min()))
    reg("IV-C", "porosidade observada de AP, RO, AC, RR e AM: maior", float(obs.loc[norte].max().max()))
    if len(anos) > 1:
        reg("IV-C", f"Spearman da porosidade entre {anos[0]} e {anos[-1]}",
            float(spearmanr(obs[anos[0]], obs[anos[-1]])[0]))
    n_mun = geo["UF"].value_counts()
    for a in anos:
        reg("IV-C", "Spearman da porosidade com o número de municípios da UF",
            float(spearmanr(obs[a], n_mun.loc[obs.index])[0]), a)
    return obs


def entre_anos():
    print("\n== Partições entre exercícios (Seção IV-B)")
    pe = pd.read_csv(C.DIR_SAIDAS / "comunidades_pertenca.csv")
    a0, a1 = min(C.EXERCICIOS), max(C.EXERCICIOS)
    for k in C.KS:
        x, y = pe[f"{a0}_k{k}_ng"], pe[f"{a1}_k{k}_ng"]
        reg("IV-B", f"AMI entre as partições de {a0} e {a1}, k = {k}", comum.ami(x, y))
        reg("IV-B", f"ARI entre as partições de {a0} e {a1}, k = {k}", float(adjusted_rand_score(x, y)))


def entropia(geo, porosidade_obs):
    print("\n== Entropia das UFs entre as comunidades (Seção V)")
    pe = pd.read_csv(C.DIR_SAIDAS / "comunidades_pertenca.csv").set_index("codigo_ibge").loc[geo["codigo_ibge"]]
    uf = geo["UF"].to_numpy()
    linhas = []
    for ano in C.EXERCICIOS:
        part = pe[f"{ano}_k{C.K_PADRAO}_ng"].to_numpy()
        K = len(np.unique(part))
        H = {}
        for u in np.unique(uf):
            q = pd.Series(part[uf == u]).value_counts(normalize=True).to_numpy()
            H[u] = float(-(q * np.log(q)).sum() / np.log(K))
        H = pd.Series(H)
        linhas += [{"exercicio": ano, "UF": u, "entropia": h,
                    "porosidade": porosidade_obs.at[u, ano] if u in porosidade_obs.index else np.nan}
                   for u, h in H.items()]
        reg("V", "número de comunidades K", K, ano)
        reg("V", "entropia de MG", float(H["MG"]), ano)
        reg("V", "posição de MG entre as UFs (1 = maior entropia)", int(H.rank(ascending=False)["MG"]), ano)
        u = porosidade_obs.index
        rho, pval = spearmanr(H[u], porosidade_obs.loc[u, ano])
        reg("V", "Spearman entre entropia e porosidade", float(rho), ano)
        reg("V", "p-valor do Spearman entre entropia e porosidade", float(pval), ano)
    pd.DataFrame(linhas).to_csv(C.DIR_SAIDAS / "entropia_uf.csv", index=False)


def main():
    rapido = "--rapido" in sys.argv
    base = comum.carregar_base()
    geo, contig = comum.carregar_geo()
    rob = modulo_robustez()

    dados(base)
    knn_mutuo(base, geo)
    if not rapido:
        sem_winsorizacao(base, geo)
    louvain_networkx()
    particoes_contiguas(geo, contig)
    substitutos_moran(base, geo, contig, rob)
    obs = porosidade(geo)
    entre_anos()
    entropia(geo, obs)

    out = pd.DataFrame(LINHAS)
    out["exercicio"] = out["exercicio"].astype("Int64")
    out.to_csv(C.DIR_SAIDAS / "numeros_texto.csv", index=False)
    print(f"\n{len(LINHAS)} números em", C.DIR_SAIDAS / "numeros_texto.csv")


if __name__ == "__main__":
    main()
