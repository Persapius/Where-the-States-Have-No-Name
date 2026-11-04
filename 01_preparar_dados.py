import difflib
import json
import re
import unicodedata

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
import scipy.sparse as sp

import config as C


def baixar(url: str, destino, binario: bool = True):
    if destino.exists():
        return destino
    print(f"Baixando {url}")
    r = requests.get(url, timeout=600)
    r.raise_for_status()
    destino.write_bytes(r.content)
    return destino


def normalizar(nome: str) -> str:
    s = unicodedata.normalize("NFKD", str(nome)).encode("ascii", "ignore").decode()
    s = s.upper().replace("'", "").replace("`", "")
    s = re.sub(r"[-.]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def ler_painel() -> pd.DataFrame:
    df = pd.read_excel(C.ARQUIVO_PAINEL, sheet_name=C.ABA_MUNICIPIO,
                       dtype={"Exercício": str})
    df["Exercício"] = df["Exercício"].astype(int)
    exterior = df["UF"].eq("EX")
    print(f"Painel: {len(df)} linhas; removidas {exterior.sum()} do exterior (UF = EX).")
    return df[~exterior].copy()


def ler_malha() -> gpd.GeoDataFrame:
    zipf = baixar(C.URL_MALHA, C.DIR_BRUTOS / "BR_Municipios_2022.zip")
    g = gpd.read_file(f"zip://{zipf}")
    g = g.rename(columns={"CD_MUN": "codigo_ibge", "NM_MUN": "nome_ibge", "SIGLA_UF": "UF"})
    g["codigo_ibge"] = g["codigo_ibge"].astype(int)
    return g[["codigo_ibge", "nome_ibge", "UF", "geometry"]]


def ler_regioes() -> pd.DataFrame:
    arq = baixar(C.URL_LOCALIDADES, C.DIR_BRUTOS / "ibge_localidades_municipios.json")
    linhas = []
    for m in json.loads(arq.read_text(encoding="utf-8")):
        imed = m.get("regiao-imediata") or {}
        inter = imed.get("regiao-intermediaria") or {}
        linhas.append({
            "codigo_ibge": int(m["id"]),
            "regiao_imediata": imed.get("nome"),
            "cod_regiao_imediata": imed.get("id"),
            "regiao_intermediaria": inter.get("nome"),
            "cod_regiao_intermediaria": inter.get("id"),
        })
    return pd.DataFrame(linhas)


def ler_populacao() -> pd.DataFrame:
    arq = baixar(C.URL_POPULACAO, C.DIR_BRUTOS / "sidra_4709_populacao_2022.json")
    dados = json.loads(arq.read_text(encoding="utf-8"))[1:]
    pop = pd.DataFrame({"codigo_ibge": [int(d["D1C"]) for d in dados],
                        "populacao_2022": [pd.to_numeric(d["V"], errors="coerce") for d in dados]})
    return pop


def parear(painel: pd.DataFrame, malha: gpd.GeoDataFrame) -> pd.DataFrame:
    unicos = painel[["UF", "Município"]].drop_duplicates().copy()
    unicos["chave"] = unicos["Município"].map(normalizar)
    ibge = malha[["codigo_ibge", "nome_ibge", "UF"]].copy()
    ibge["chave"] = ibge["nome_ibge"].map(normalizar)

    res = unicos.merge(ibge, on=["UF", "chave"], how="left")
    res["metodo"] = np.where(res["codigo_ibge"].notna(), "exato", None)
    n_cand = res.groupby(["UF", "Município"])["codigo_ibge"].transform("size")
    res.loc[n_cand > 1, ["codigo_ibge", "nome_ibge"]] = np.nan
    res.loc[n_cand > 1, "metodo"] = "ambiguo"
    res = res.drop_duplicates(["UF", "Município"]).reset_index(drop=True)

    usados = set(res["codigo_ibge"].dropna().astype(int))
    for i in res.index[res["metodo"].isna()]:
        uf, chave = res.at[i, "UF"], res.at[i, "chave"]
        cand = ibge[(ibge["UF"] == uf) & (~ibge["codigo_ibge"].isin(usados))]
        achado = difflib.get_close_matches(chave, cand["chave"].tolist(), n=1, cutoff=0.8)
        if achado:
            linha = cand[cand["chave"] == achado[0]].iloc[0]
            res.at[i, "codigo_ibge"] = linha["codigo_ibge"]
            res.at[i, "nome_ibge"] = linha["nome_ibge"]
            res.at[i, "metodo"] = "aproximado"
            usados.add(int(linha["codigo_ibge"]))

    manual = C.RAIZ / "pareamento_manual.csv"
    if manual.exists():
        man = pd.read_csv(manual)
        for _, m in man.iterrows():
            sel = (res["UF"] == m["UF"]) & (res["Município"] == m["Município"])
            res.loc[sel, "codigo_ibge"] = int(m["codigo_ibge"])
            res.loc[sel, "metodo"] = "manual"
        res = res.drop(columns="nome_ibge").merge(
            ibge[["codigo_ibge", "nome_ibge"]], on="codigo_ibge", how="left")

    res["metodo"] = res["metodo"].fillna("nao_pareado")
    return res.drop(columns="chave")


def contiguidade(malha: gpd.GeoDataFrame) -> sp.csr_matrix:
    g = malha.reset_index(drop=True)
    pares = gpd.sjoin(g[["geometry"]], g[["geometry"]], predicate="intersects")
    pares = pares[pares.index != pares["index_right"]]
    n = len(g)
    A = sp.coo_matrix((np.ones(len(pares)), (pares.index.to_numpy(), pares["index_right"].to_numpy())),
                      shape=(n, n)).tocsr()
    A.data[:] = 1.0
    return ((A + A.T) > 0).astype(float).tocsr()


def main():
    C.DIR_BRUTOS.mkdir(parents=True, exist_ok=True)
    C.DIR_SAIDAS.mkdir(parents=True, exist_ok=True)

    painel = ler_painel()
    malha = ler_malha()
    regioes = ler_regioes()
    pop = ler_populacao()

    par = parear(painel, malha)
    par.to_csv(C.DIR_SAIDAS / "pareamento_municipios.csv", index=False)
    resumo = par["metodo"].value_counts()
    print("Pareamento:", resumo.to_dict())
    dup = par[par["codigo_ibge"].notna() & par.duplicated("codigo_ibge", keep=False)]
    if len(dup):
        print(f"ATENÇÃO: {len(dup)} linhas do painel caíram no mesmo código IBGE; veja o relatório.")
    nao = par[par["codigo_ibge"].isna()]
    if len(nao):
        print(f"ATENÇÃO: {len(nao)} municípios do painel sem código IBGE (serão descartados; "
              f"corrija em pareamento_manual.csv):")
        print(nao[["UF", "Município", "metodo"]].to_string(index=False))
    aprox = par[par["metodo"] == "aproximado"]
    if len(aprox):
        print(f"Revise os {len(aprox)} pareamentos aproximados em saidas/pareamento_municipios.csv:")
        print(aprox[["UF", "Município", "nome_ibge"]].to_string(index=False))
    sem_painel = malha[~malha["codigo_ibge"].isin(par["codigo_ibge"].dropna().astype(int))]
    print(f"Municípios da malha IBGE sem correspondente no painel: {len(sem_painel)}")
    if len(sem_painel):
        print(sem_painel[["codigo_ibge", "nome_ibge", "UF"]].to_string(index=False))

    ok = par[par["codigo_ibge"].notna()].drop_duplicates("codigo_ibge", keep=False)
    ok = ok.assign(codigo_ibge=ok["codigo_ibge"].astype(int))

    base = painel.merge(ok[["UF", "Município", "codigo_ibge"]], on=["UF", "Município"])
    base = base.merge(pop, on="codigo_ibge", how="left").merge(regioes, on="codigo_ibge", how="left")
    base["taxa_declarantes"] = base["Declarantes"] / base["populacao_2022"]
    for col in ["populacao_2022", "regiao_imediata", "regiao_intermediaria"]:
        n_na = base[col].isna().sum()
        if n_na:
            print(f"ATENÇÃO: {n_na} linhas sem {col}.")

    cols = list(C.INDICADORES)
    completos = base.dropna(subset=cols + ["regiao_intermediaria"])
    presentes = completos.groupby("codigo_ibge")["Exercício"].nunique()
    manter = presentes.index[presentes == len(C.EXERCICIOS)]
    base = completos[completos["codigo_ibge"].isin(manter)].copy()
    print(f"Municípios na análise: {base['codigo_ibge'].nunique()}")
    base.to_parquet(C.DIR_DADOS / "base_municipios.parquet", index=False)

    m = malha[malha["codigo_ibge"].isin(manter)].sort_values("codigo_ibge").reset_index(drop=True)
    reg = base.drop_duplicates("codigo_ibge").set_index("codigo_ibge")
    for nivel in ("imediata", "intermediaria"):
        cod = reg.loc[m["codigo_ibge"], f"cod_regiao_{nivel}"].astype("int64").astype(str)
        m[f"regiao_{nivel}"] = cod.to_numpy()
        m[f"nome_regiao_{nivel}"] = reg.loc[m["codigo_ibge"], f"regiao_{nivel}"].to_numpy()
    print(f"Regiões imediatas: {m['regiao_imediata'].nunique()}; "
          f"intermediárias: {m['regiao_intermediaria'].nunique()}")
    m_proj = m.to_crs(C.CRS_METRICO)
    cent = m_proj.geometry.centroid
    geo = pd.DataFrame({
        "codigo_ibge": m["codigo_ibge"], "UF": m["UF"],
        "regiao_intermediaria": m["regiao_intermediaria"],
        "regiao_imediata": m["regiao_imediata"],
        "nome_regiao_intermediaria": m["nome_regiao_intermediaria"],
        "nome_regiao_imediata": m["nome_regiao_imediata"],
        "x_km": cent.x / 1000, "y_km": cent.y / 1000,
    })
    geo.to_parquet(C.DIR_DADOS / "geo_municipios.parquet", index=False)
    m.to_parquet(C.DIR_DADOS / "malha_municipios.parquet")

    A = contiguidade(m_proj)
    sp.save_npz(C.DIR_DADOS / "contiguidade.npz", A)
    isolados = np.flatnonzero(np.asarray(A.sum(1)).ravel() == 0)
    print(f"Contiguidade: {int(A.nnz / 2)} pares; {len(isolados)} municípios sem vizinhos "
          f"(ilhas): {m.loc[isolados, 'nome_ibge'].tolist()}")


if __name__ == "__main__":
    main()
