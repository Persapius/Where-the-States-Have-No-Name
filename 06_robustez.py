import os
import sys

import config as C

if C.N_PROCESSOS > 1:
    for v in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
        os.environ.setdefault(v, "1")

import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

import comum

METADADOS = {"UF": "UF", "intermediaria": "regiao_intermediaria", "imediata": "regiao_imediata"}

G = {}


def porosidade(A, P, uf):
    Ad = A.toarray()
    saida = {}
    for u in np.unique(uf):
        dentro = uf == u
        if dentro.sum() < 2:
            continue
        obs = Ad[dentro][:, ~dentro].sum() / Ad[dentro].sum()
        esp = P[dentro][:, ~dentro].sum() / P[dentro].sum()
        saida[f"porosidade_{u}"] = obs / esp
    return saida


def metricas(X, semente, part=None):
    geo, contig, F = G["geo"], G["contig"], G["F"]
    A = comum.rede_knn(X, C.MORAN_K)
    B = comum.matriz_modularidade(A, comum.nulo_ng(A))
    if part is None:
        part, Q, _, _ = comum.detectar(B, A.sum(), C.SEMENTES, semente)
    else:
        Q = comum.modularidade(B, part, A.sum())
    tam = np.bincount(part)
    m = {"Q": Q, "n_comunidades": int(len(tam)),
         "join_count": comum.razao_join_count(contig, part),
         "cosseno_intra": comum.similaridade_cosseno_intra(X, part)}
    for nome, col in METADADOS.items():
        m[f"ami_{nome}"] = comum.ami(geo[col], part)
        m[f"bestest_contiguo_z_{nome}"] = comum.bestest_contiguo(
            A, geo[col].to_numpy(), G["nulas"][nome])["z"]
    m.update(porosidade(A, comum.nulo_espacial(A, F)[0], geo["UF"].to_numpy()))
    return m


def autovetores_moran(contig):
    cache = C.DIR_DADOS / "moran_autovetores.npy"
    n = contig.shape[0]
    if cache.exists():
        V = np.load(cache)
        if V.shape == (n, n):
            return V
    print("Calculando os autovetores de Moran (uma vez; alguns minutos)...")
    W = contig.toarray()
    W = (W + W.T) / 2
    W -= W.mean(0, keepdims=True)
    W -= W.mean(1, keepdims=True)
    _, V = np.linalg.eigh(W)
    np.save(cache, V)
    return V


def substituto_moran(X, V, rng):
    coef = V.T @ X
    sinais = rng.choice([-1.0, 1.0], size=V.shape[1])
    Xs = V @ (sinais[:, None] * coef)
    return (Xs - Xs.mean(0)) / Xs.std(0)


def i_moran(x, contig):
    z = x - x.mean()
    return float(len(x) / contig.sum() * (z @ (contig @ z)) / (z @ z))


def _tarefa_moran(args):
    ano, i = args
    rng = np.random.default_rng([C.SEMENTE_GLOBAL, ano, i])
    Xs = substituto_moran(G["X"][ano], G["V"], rng)
    t0 = time.time()
    m = metricas(Xs, C.SEMENTE_GLOBAL + i)
    return {"exercicio": ano, "substituto": i, "segundos": time.time() - t0, **m}


def _descarta_moran_se_config_mudou():
    assinatura = {"sementes": C.SEMENTES, "k": C.MORAN_K, "tau": C.CONSENSO_TAU,
                  "metodo": C.METODO_PARTICAO, "rotulos_regioes": "codigo"}
    arq_assin = C.DIR_SAIDAS / "robustez_moran_config.json"
    if arq_assin.exists() and json.loads(arq_assin.read_text()) == assinatura:
        return
    for nome in ("robustez_moran_substitutos.csv", "robustez_moran_observado.csv"):
        a = C.DIR_SAIDAS / nome
        if a.exists():
            a.rename(a.with_name(a.stem + "_antigo.csv"))
            print(f"Parâmetros mudaram: {nome} renomeado para {a.stem}_antigo.csv")
    C.DIR_SAIDAS.mkdir(parents=True, exist_ok=True)
    arq_assin.write_text(json.dumps(assinatura))


def rodar_moran():
    _descarta_moran_se_config_mudou()
    arq = C.DIR_SAIDAS / "robustez_moran_substitutos.csv"
    feitos = set()
    if arq.exists():
        antigo = pd.read_csv(arq)
        feitos = set(zip(antigo["exercicio"], antigo["substituto"]))
        print(f"Retomando: {len(feitos)} substitutos já calculados.")

    V = autovetores_moran(G["contig"])
    G["V"] = V
    chec = []
    for ano in C.EXERCICIOS:
        X = G["X"][ano]
        Xs = substituto_moran(X, V, np.random.default_rng(0))
        for j, nome in enumerate(C.INDICADORES):
            chec.append({"exercicio": ano, "indicador": nome,
                         "moran_original": i_moran(X[:, j], G["contig"]),
                         "moran_substituto": i_moran(Xs[:, j], G["contig"])})
        print(f"{ano}: maior diferença entre correlações originais e do substituto = "
              f"{np.abs(np.corrcoef(X.T) - np.corrcoef(Xs.T)).max():.2e}")
    pd.DataFrame(chec).to_csv(C.DIR_SAIDAS / "robustez_moran_checagem.csv", index=False)

    obs = [{"exercicio": ano, **metricas(G["X"][ano], C.SEMENTE_GLOBAL, G["ref"][ano])}
           for ano in C.EXERCICIOS]
    pd.DataFrame(obs).to_csv(C.DIR_SAIDAS / "robustez_moran_observado.csv", index=False)

    tarefas = [(ano, i) for ano in C.EXERCICIOS for i in range(C.MORAN_SUBSTITUTOS)
               if (ano, i) not in feitos]
    print(f"{len(tarefas)} substitutos a calcular com {C.N_PROCESSOS} processos.")
    t0 = time.time()
    with ProcessPoolExecutor(C.N_PROCESSOS) as ex:
        futuros = [ex.submit(_tarefa_moran, t) for t in tarefas]
        for n, fut in enumerate(as_completed(futuros), 1):
            linha = pd.DataFrame([fut.result()])
            linha.to_csv(arq, mode="a", header=not arq.exists(), index=False)
            if n % 10 == 0 or n == len(tarefas):
                resta = (time.time() - t0) / n * (len(tarefas) - n) / 3600
                print(f"   {n}/{len(tarefas)} concluídos; faltam ~{resta:.1f} h")
    resumir_moran()


def resumir_moran():
    obs = pd.read_csv(C.DIR_SAIDAS / "robustez_moran_observado.csv")
    sub = pd.read_csv(C.DIR_SAIDAS / "robustez_moran_substitutos.csv")
    linhas = []
    for ano in C.EXERCICIOS:
        o = obs[obs["exercicio"] == ano].iloc[0]
        s = sub[sub["exercicio"] == ano]
        for col in [c for c in obs.columns if c != "exercicio" and c in s]:
            nulos = s[col].dropna().to_numpy()
            if len(nulos) < 2:
                continue
            dev = np.abs(nulos - np.median(nulos))
            p = (1 + (dev >= abs(o[col] - np.median(nulos))).sum()) / (1 + len(nulos))
            linhas.append({"exercicio": ano, "metrica": col, "observado": o[col],
                           "nulo_media": nulos.mean(), "nulo_dp": nulos.std(),
                           "nulo_p2.5": np.quantile(nulos, 0.025),
                           "nulo_p97.5": np.quantile(nulos, 0.975),
                           "z": (o[col] - nulos.mean()) / nulos.std() if nulos.std() > 0 else np.nan,
                           "p_bilateral": p, "n_substitutos": len(nulos)})
    res = pd.DataFrame(linhas)
    res.to_csv(C.DIR_SAIDAS / "robustez_moran_resumo.csv", index=False)
    print(res[~res["metrica"].str.startswith("porosidade")].round(3).to_string(index=False))

    import matplotlib.pyplot as plt
    principais = ["ami_UF", "ami_intermediaria", "join_count", "cosseno_intra",
                  "bestest_contiguo_z_UF", "n_comunidades"]
    fig, axs = plt.subplots(len(C.EXERCICIOS), len(principais),
                            figsize=(3 * len(principais), 2.6 * len(C.EXERCICIOS)), squeeze=False)
    for r, ano in enumerate(C.EXERCICIOS):
        o = obs[obs["exercicio"] == ano].iloc[0]
        s = sub[sub["exercicio"] == ano]
        for c, met in enumerate(principais):
            ax = axs[r, c]
            ax.hist(s[met].dropna(), bins=30, color="#bbbbbb")
            ax.axvline(o[met], color="#c0392b", linewidth=2)
            ax.set_title(f"{met} ({ano})", fontsize=8)
            ax.tick_params(labelsize=7)
    fig.suptitle("Nulo de Moran: distribuição nos substitutos (cinza) vs. observado (vermelho)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(C.DIR_SAIDAS / "robustez_moran.png", dpi=200)


def _tarefa_resolucao(args):
    ano, gama = args
    geo, contig, X = G["geo"], G["contig"], G["X"][ano]
    A = comum.rede_knn(X, C.K_PADRAO)
    B = comum.matriz_modularidade(A, gama * comum.nulo_ng(A))
    part, Q, todas, info = comum.detectar(B, A.sum(), C.SEMENTES, C.SEMENTE_GLOBAL)
    tam = np.bincount(part)
    linha = {"exercicio": ano, "gama": gama, "Q_gama": Q, "n_comunidades": int(len(tam)),
             "n_comunidades_5_ou_mais": int((tam >= 5).sum()),
             "estabilidade": float(np.mean([comum.ami(part, p) for p in todas])),
             "join_count": comum.razao_join_count(contig, part),
             "cosseno_intra": comum.similaridade_cosseno_intra(X, part)}
    for nome, col in METADADOS.items():
        linha[f"ami_{nome}"] = comum.ami(geo[col], part)
    return linha


def rodar_resolucao():
    tarefas = [(ano, g) for ano in C.EXERCICIOS for g in C.GAMAS]
    with ProcessPoolExecutor(C.N_PROCESSOS) as ex:
        res = [f.result() for f in as_completed([ex.submit(_tarefa_resolucao, t) for t in tarefas])]
    res = pd.DataFrame(res).sort_values(["exercicio", "gama"])
    res.to_csv(C.DIR_SAIDAS / "robustez_resolucao.csv", index=False)
    print(res.round(3).to_string(index=False))

    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
    for ano, g in res.groupby("exercicio"):
        for nome, est in zip(METADADOS, ["-", "--", ":"]):
            axs[0].plot(g["n_comunidades"], g[f"ami_{nome}"], est, marker="o",
                        label=f"{nome} ({ano})")
        axs[1].plot(g["n_comunidades"], g["join_count"], marker="o", label=str(ano))
    for ax in axs:
        ax.set_xscale("log")
        ax.set_xlabel("nº de comunidades (varrendo gama)")
    axs[0].set_title("AMI com as divisões do IBGE", fontsize=10)
    axs[1].set_title("Razão de join count", fontsize=10)
    axs[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(C.DIR_SAIDAS / "robustez_resolucao.png", dpi=200)


def _tarefa_sensibilidade(args):
    ano, tipo, valor = args
    geo, contig = G["geo"], G["contig"]
    if tipo == "sem_grupo":
        ind = {k: v for k, v in C.INDICADORES.items() if k not in C.GRUPOS_INDICADORES[valor]}
        X, k = comum.matriz_indicadores(G["base"], geo, ano, ind), C.K_PADRAO
    else:
        X, k = G["X"][ano], int(valor)
    A = comum.rede_knn(X, k)
    B = comum.matriz_modularidade(A, comum.nulo_ng(A))
    if tipo == "k" and int(valor) == C.K_PADRAO:
        part = G["ref"][ano]
        Q = comum.modularidade(B, part, A.sum())
    else:
        part, Q, _, _ = comum.detectar(B, A.sum(), C.SEMENTES, C.SEMENTE_GLOBAL)
    return {"exercicio": ano, "variacao": tipo, "valor": valor, "part": part, "Q": Q,
            "n_comunidades": int(part.max() + 1),
            "join_count": comum.razao_join_count(contig, part),
            "ami_UF": comum.ami(geo["UF"], part)}


def rodar_sensibilidade():
    tarefas = [(ano, "k", k) for ano in C.EXERCICIOS for k in C.KS_SENSIBILIDADE]
    tarefas += [(ano, "sem_grupo", g) for ano in C.EXERCICIOS for g in C.GRUPOS_INDICADORES]
    with ProcessPoolExecutor(C.N_PROCESSOS) as ex:
        res = [f.result() for f in as_completed([ex.submit(_tarefa_sensibilidade, t) for t in tarefas])]
    for r in res:
        r["ami_com_referencia"] = comum.ami(G["ref"][r["exercicio"]], r.pop("part"))
    out = pd.DataFrame(res).sort_values(["exercicio", "variacao", "valor"])
    out.to_csv(C.DIR_SAIDAS / "robustez_sensibilidade.csv", index=False)
    print(out.round(3).to_string(index=False))


def preparar():
    base = comum.carregar_base()
    geo, contig = comum.carregar_geo()
    D = comum.matriz_distancias(geo)
    xy = geo[["x_km", "y_km"]].to_numpy()
    G.update(base=base, geo=geo, contig=contig, F=comum.faixas_distancia(D, C.N_BINS_PADRAO),
             X={ano: comum.matriz_indicadores(base, geo, ano) for ano in C.EXERCICIOS})
    G["nulas"] = {nome: comum.particoes_contiguas(
        contig, geo[col].to_numpy(), xy, C.BESTEST_CONTIGUAS, C.SEMENTE_GLOBAL,
        cache=C.DIR_DADOS / f"particoes_contiguas_{nome}.npy") for nome, col in METADADOS.items()}
    pe = pd.read_csv(C.DIR_SAIDAS / "comunidades_pertenca.csv").set_index("codigo_ibge").loc[geo["codigo_ibge"]]
    G["ref"] = {ano: pe[f"{ano}_k{C.K_PADRAO}_ng"].to_numpy() for ano in C.EXERCICIOS}


def main():
    import multiprocessing as mp
    mp.set_start_method("fork", force=True)
    alvo = sys.argv[1] if len(sys.argv) > 1 else "tudo"
    C.DIR_SAIDAS.mkdir(parents=True, exist_ok=True)
    preparar()
    if alvo in ("resolucao", "tudo"):
        rodar_resolucao()
    if alvo in ("sensibilidade", "tudo"):
        rodar_sensibilidade()
    if alvo in ("moran", "tudo"):
        rodar_moran()
    if alvo == "resumo_moran":
        resumir_moran()


if __name__ == "__main__":
    main()
