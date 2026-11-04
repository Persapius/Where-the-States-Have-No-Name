import numpy as np
import pandas as pd

import comum
import config as C

METADADOS = {"UF": "UF", "intermediaria": "regiao_intermediaria", "imediata": "regiao_imediata"}


def main():
    C.DIR_SAIDAS.mkdir(parents=True, exist_ok=True)
    base = comum.carregar_base()
    geo, contig = comum.carregar_geo()
    malha = comum.carregar_malha()
    D = comum.matriz_distancias(geo)
    faixas = {nb: comum.faixas_distancia(D, nb) for nb in C.N_BINS_SENSIBILIDADE}
    xy = geo[["x_km", "y_km"]].to_numpy()

    nulas = {}
    for nome, col in METADADOS.items():
        print(f"Gerando {C.BESTEST_CONTIGUAS} partições contíguas aleatórias ({nome})...")
        nulas[nome] = comum.particoes_contiguas(
            contig, geo[col].to_numpy(), xy, C.BESTEST_CONTIGUAS, C.SEMENTE_GLOBAL,
            cache=C.DIR_DADOS / f"particoes_contiguas_{nome}.npy")
        rot = pd.factorize(geo[col])[0]
        jc = [comum.razao_join_count(contig, p) for p in nulas[nome][:50]]
        print(f"   join count: divisão real = {comum.razao_join_count(contig, rot):.3f}; "
              f"aleatórias contíguas = {np.mean(jc):.3f} ± {np.std(jc):.3f}")

    linhas, testes, redes, pertencas = [], [], [], {}
    for ano in C.EXERCICIOS:
        X = comum.matriz_indicadores(base, geo, ano)
        for k in C.KS:
            A = comum.rede_knn(X, k)
            m2 = A.sum()
            grau = np.asarray((A > 0).sum(1)).ravel()
            redes.append({"exercicio": ano, "k": k, "arestas": int(A.nnz / 2),
                          "isolados": int((grau == 0).sum()), "grau_medio": float(grau.mean())})
            print(f"\n== {ano}, k={k}: {redes[-1]}")

            for nome, col in METADADOS.items():
                r1 = comum.bestest(A, geo[col].to_numpy(), C.BESTEST_PERMUTACOES,
                                   semente=C.SEMENTE_GLOBAL)
                r2 = comum.bestest_contiguo(A, geo[col].to_numpy(), nulas[nome])
                testes.append({"exercicio": ano, "k": k, "metadado": nome, "nulo": "permutacao", **r1})
                testes.append({"exercicio": ano, "k": k, "metadado": nome, "nulo": "contiguo", **r2})
                print(f"   BESTest {nome}: permutação z = {r1['z']:.1f} (p = {r1['p']:.4f}); "
                      f"contíguo z = {r2['z']:.1f} (p = {r2['p']:.4f}, percentil {r2['percentil_obs']:.2f})")

            configs = [("ng", None, comum.nulo_ng(A))]
            bins = C.N_BINS_SENSIBILIDADE if k == C.K_PADRAO else [C.N_BINS_PADRAO]
            for nb in bins:
                configs.append(("espacial", nb, comum.nulo_espacial(A, faixas[nb])[0]))

            for nulo, nb, P in configs:
                B = comum.matriz_modularidade(A, P)
                part, Q, todas, info = comum.detectar(B, m2, C.SEMENTES, C.SEMENTE_GLOBAL,
                                                      reconectar=(nulo == "ng"))
                chave = f"{ano}_k{k}_{nulo}" + (f"_b{nb}" if nb else "")
                pertencas[chave] = part
                tam = np.bincount(part)
                linha = {"exercicio": ano, "k": k, "nulo": nulo, "n_bins": nb,
                         "metodo_particao": C.METODO_PARTICAO, "Q": Q, **info,
                         "n_comunidades": int(len(tam)),
                         "n_comunidades_5_ou_mais": int((tam >= 5).sum()),
                         "maior_comunidade": int(tam.max()),
                         "estabilidade_ami_sementes_vs_final": float(np.mean(
                             [comum.ami(part, p) for p in todas])),
                         "estabilidade_ami_entre_sementes": float(np.mean(
                             [comum.ami(todas[i], todas[j]) for i in range(len(todas))
                              for j in range(i + 1, len(todas))])),
                         "join_count": comum.razao_join_count(contig, part),
                         "join_count_aleatorio_esperado": float(((tam / tam.sum()) ** 2).sum()),
                         "cosseno_intra": comum.similaridade_cosseno_intra(X, part)}
                for nome, col in METADADOS.items():
                    linha[f"ami_{nome}"] = comum.ami(geo[col], part)
                    linha[f"zrand_{nome}"] = comum.z_rand(geo[col], part)
                linhas.append(linha)
                print(f"   {chave}: Q={Q:.3f}, {linha['n_comunidades']} comunidades, "
                      f"AMI(UF)={linha['ami_UF']:.3f}, join count={linha['join_count']:.3f} "
                      f"(aleatório {linha['join_count_aleatorio_esperado']:.3f}), "
                      f"estabilidade={linha['estabilidade_ami_sementes_vs_final']:.2f}")

                if k == C.K_PADRAO and nb in (None, C.N_BINS_PADRAO):
                    comum.mapa_comunidades(
                        malha, part,
                        f"Exercício {ano}, k = {k}, nulo {nulo}: {linha['n_comunidades']} comunidades, "
                        f"AMI(UF) = {linha['ami_UF']:.2f}",
                        C.DIR_SAIDAS / f"mapa_comunidades_{chave}.png")

    for ano in C.EXERCICIOS:
        X = comum.matriz_indicadores(base, geo, ano)
        for nome, col in METADADOS.items():
            rot = pd.factorize(geo[col])[0]
            tam = np.bincount(rot)
            linhas.append({"exercicio": ano, "k": None, "nulo": f"referencia_{nome}",
                           "n_comunidades": int(len(tam)), "maior_comunidade": int(tam.max()),
                           "join_count": comum.razao_join_count(contig, rot),
                           "join_count_aleatorio_esperado": float(((tam / tam.sum()) ** 2).sum()),
                           "cosseno_intra": comum.similaridade_cosseno_intra(X, rot)})

    pd.DataFrame(linhas).to_csv(C.DIR_SAIDAS / "comunidades_resumo.csv", index=False)
    pd.DataFrame(testes).to_csv(C.DIR_SAIDAS / "bestest.csv", index=False)
    pd.DataFrame(redes).to_csv(C.DIR_SAIDAS / "redes_resumo.csv", index=False)
    perten = pd.DataFrame(pertencas)
    perten.insert(0, "codigo_ibge", geo["codigo_ibge"].to_numpy())
    perten.to_csv(C.DIR_SAIDAS / "comunidades_pertenca.csv", index=False)
    print("\nResultados em", C.DIR_SAIDAS)


if __name__ == "__main__":
    main()
