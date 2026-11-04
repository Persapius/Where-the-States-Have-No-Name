import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

import config as C

S = C.DIR_SAIDAS
OUT = S / "artigo"
ANO = max(C.EXERCICIOS)

AZUL, LARANJA, AQUA, CINZA = "#2a78d6", "#eb6834", "#1baf7a", "#9a9993"
TINTA, TINTA2 = "#0b0b0b", "#52514e"
DIV = {"UF": ("State", AZUL, "-", "o"),
       "intermediaria": ("Intermediate region", LARANJA, "--", "s"),
       "imediata": ("Immediate region", AQUA, ":", "^")}

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["TeX Gyre Termes", "Times New Roman", "Liberation Serif", "DejaVu Serif"],
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "axes.edgecolor": TINTA2, "axes.labelcolor": TINTA, "xtick.color": TINTA2,
    "ytick.color": TINTA2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.5, "axes.axisbelow": True,
    "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})


def fig_moran():
    obs = pd.read_csv(S / "robustez_moran_observado.csv").set_index("exercicio")
    sub = pd.read_csv(S / "robustez_moran_substitutos.csv")
    mets = [("ami_UF", "AMI with states"), ("ami_intermediaria", "AMI with intermediate regions"),
            ("bestest_contiguo_z_UF", "Contiguous BESTest $z$, states")]
    anos = sorted(obs.index)
    fig, axs = plt.subplots(len(anos), len(mets), figsize=(6.3, 1.45 * len(anos) + 0.3), squeeze=False)
    for r, ano in enumerate(anos):
        s = sub[sub["exercicio"] == ano]
        for c, (m, rot) in enumerate(mets):
            ax = axs[r, c]
            v, o = s[m].dropna(), obs.loc[ano, m]
            lo, hi = min(v.min(), o), max(v.max(), o)
            xmin, xmax = lo - 0.08 * (hi - lo), hi + 0.22 * (hi - lo)
            ax.hist(v, bins=np.linspace(xmin, xmax, 80), color=CINZA, linewidth=0)
            ax.axvline(o, color=AZUL, linewidth=1.6)
            ax.set_xlim(xmin, xmax)
            ax.text(o, ax.get_ylim()[1] * 0.92, f" observed\n {o:.3f}" if "ami" in m else f" observed\n {o:.2f}",
                    color=TINTA, va="top", ha="left", fontsize=6.5)
            ax.set_yticks([])
            ax.grid(False)
            if r == 0:
                ax.set_title(rot)
            if c == 0:
                ax.set_ylabel(str(ano))
    fig.tight_layout(h_pad=0.6, w_pad=0.8)
    fig.savefig(OUT / "fig_moran_null.pdf")
    plt.close(fig)


def fig_porosidade():
    r = pd.read_csv(S / "robustez_moran_resumo.csv")
    p = r[r["metrica"].str.startswith("porosidade_")].copy()
    p["UF"] = p["metrica"].str.replace("porosidade_", "", regex=False)
    anos = sorted(p["exercicio"].unique())
    ordem = p[p["exercicio"] == ANO].sort_values("observado")["UF"].tolist()
    y = {u: i for i, u in enumerate(ordem)}
    cores = {anos[0]: LARANJA, anos[-1]: AZUL}
    marc = {anos[0]: "s", anos[-1]: "o"}
    desl = {anos[0]: 0.17, anos[-1]: -0.17}
    fig, ax = plt.subplots(figsize=(3.3, 4.3))
    for ano in anos:
        q = p[p["exercicio"] == ano]
        for _, l in q.iterrows():
            yy = y[l["UF"]] + desl[ano]
            ax.plot([l["nulo_p2.5"], l["nulo_p97.5"]], [yy, yy], color=CINZA, linewidth=2.2,
                    solid_capstyle="round", zorder=1)
            sig = l["p_bilateral"] < 0.05
            ax.scatter(l["observado"], yy, s=16, marker=marc[ano], zorder=3,
                       facecolor=cores[ano] if sig else "white", edgecolor=cores[ano], linewidth=1)
    ax.axvline(1, color=TINTA2, linewidth=0.6, linestyle="--", zorder=0)
    ax.set_yticks(range(len(ordem)))
    ax.set_yticklabels(ordem)
    ax.set_ylim(-0.7, len(ordem) - 0.3)
    ax.set_xlabel("Border porosity (observed / expected)")
    ax.grid(axis="y", visible=False)
    from matplotlib.lines import Line2D
    leg = [Line2D([], [], color=CINZA, linewidth=2.2, label="Moran null, 95% interval")]
    leg += [Line2D([], [], linestyle="", marker=marc[a], markerfacecolor=cores[a], markeredgecolor=cores[a],
                   markersize=4.5, label=f"Observed {a}") for a in anos]
    leg += [Line2D([], [], linestyle="", marker="o", markerfacecolor="white", markeredgecolor=TINTA2,
                   markersize=4.5, label="hollow: $p \\geq 0.05$")]
    ax.legend(handles=leg, loc="upper center", bbox_to_anchor=(0.45, -0.1), ncol=2, frameon=False,
              columnspacing=1.0, handletextpad=0.4)
    fig.savefig(OUT / "fig_porosity.pdf")
    plt.close(fig)


def fig_escala():
    a = pd.read_csv(S / "varredura_alfa.csv")
    a = a[a["exercicio"] == ANO]
    g = pd.read_csv(S / "robustez_resolucao.csv")
    g = g[g["exercicio"] == ANO].sort_values("n_comunidades")
    fig, axs = plt.subplots(1, 2, figsize=(6.3, 2.2))
    for nome, col in [("UF", "ami_UF"), ("intermediaria", "ami_intermediaria")]:
        rot, cor, est, mk = DIV[nome]
        axs[0].plot(a["alfa"], a[col], color=cor, linestyle=est, marker=mk, markersize=3.5, linewidth=1.5)
    axs[0].set_xlabel(r"$\alpha$ (0: indicators only; 1: contiguity only)")
    axs[0].set_ylabel("AMI")
    axs[0].set_title(r"(a) Fusion of contiguity and similarity, $W(\alpha)$", loc="left")
    axs[0].set_xlim(-0.03, 1.03)
    axs[0].set_ylim(0, 0.8)
    for nome, (rot, cor, est, mk) in DIV.items():
        axs[1].plot(g["n_comunidades"], g[f"ami_{nome}"], color=cor, linestyle=est, marker=mk,
                    markersize=3.5, linewidth=1.5, label=f"AMI with {rot.lower()}s")
    axs[1].set_xscale("log")
    axs[1].set_xticks([3, 5, 10, 20, 50])
    axs[1].xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    axs[1].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    axs[1].set_xlim(g["n_comunidades"].min() * 0.85, g["n_comunidades"].max() * 1.2)
    axs[1].set_ylim(0, 0.8)
    axs[1].set_xlabel(r"Number of communities (resolution $\gamma$ from 0.25 to 5)")
    axs[1].set_title("(b) Similarity network across resolutions", loc="left")
    fig.legend(*axs[1].get_legend_handles_labels(), loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 1.07))
    fig.tight_layout(w_pad=1.5)
    fig.savefig(OUT / "fig_scale.pdf")
    plt.close(fig)


def mapa():
    from PIL import Image
    arq = S / f"mapa_comunidades_{ANO}_k{C.K_PADRAO}_ng.png"
    if not arq.exists():
        print(f"{arq.name} não encontrado; mapa não gerado.")
        return
    im = Image.open(arq).convert("RGB")
    a = np.asarray(im)
    linhas = np.flatnonzero((a < 245).any(axis=(1, 2)))
    titulo_fim = linhas[0] + np.argmax(np.diff(linhas) > 20) + 1
    corpo = a[linhas[titulo_fim]:]
    cols = np.flatnonzero((corpo < 245).any(axis=(0, 2)))
    lins = np.flatnonzero((corpo < 245).any(axis=(1, 2)))
    Image.fromarray(corpo[lins[0]:lins[-1] + 1, cols[0]:cols[-1] + 1]).save(OUT / "fig_communities.png")


def fmt(x, d=3):
    return "--" if pd.isna(x) else f"{x:.{d}f}"


def tabelas():
    v = pd.read_csv(S / "validacao_sintetica.csv")
    t = v.groupby(["rho", "nulo"]).agg(ami=("ami_plantado", "mean"), ami_sd=("ami_plantado", "std"),
                                       gem=("recuperacao_gemeas", "mean"), k=("n_comunidades", "mean"),
                                       jc=("join_count", "mean")).reset_index()
    linhas = [r"\begin{tabular}{@{}llcccc@{}}", r"\toprule",
              r"$\rho$ & Null model & AMI with planted regions & Twins recovered & Communities & Join count \\",
              r"\midrule"]
    for rho, g in t.groupby("rho"):
        for _, l in g.sort_values("nulo", ascending=False).iterrows():
            nome = "NG" if l["nulo"] == "ng" else "Spatial"
            linhas.append(f"{rho:.1f} & {nome} & {l['ami']:.3f} $\\pm$ {l['ami_sd']:.3f} & "
                          f"{l['gem']:.2f} & {l['k']:.1f} & {l['jc']:.3f} \\\\")
        linhas.append(r"\addlinespace")
    linhas[-1] = r"\bottomrule"
    linhas.append(r"\end{tabular}")
    (OUT / "tab_validation_body.tex").write_text("\n".join(linhas))

    c = pd.read_csv(S / "comunidades_resumo.csv")
    linhas = [r"\begin{tabular}{@{}llrrcccccc@{}}", r"\toprule",
              r"Year & Partition & Groups & $Q$ & Join count & (random) & Cosine & AMI$_{\mathrm{state}}$ & AMI$_{\mathrm{int}}$ & AMI$_{\mathrm{imm}}$ \\",
              r"\midrule"]
    for ano in sorted(C.EXERCICIOS):
        sel = c[(c["exercicio"] == ano)]
        rows = []
        for k in C.KS:
            rows.append((f"NG, $k={k}$", sel[(sel["k"] == k) & (sel["nulo"] == "ng")]))
        rows.append((f"Spatial, $k={C.K_PADRAO}$", sel[(sel["k"] == C.K_PADRAO) & (sel["nulo"] == "espacial")
                                                       & (sel["n_bins"] == C.N_BINS_PADRAO)]))
        for nome, rot in [("UF", "States"), ("intermediaria", "Intermediate regions"), ("imediata", "Immediate regions")]:
            rows.append((rot, sel[sel["nulo"] == f"referencia_{nome}"]))
        for i, (rot, l) in enumerate(rows):
            l = l.iloc[0]
            linhas.append(f"{ano if i == 0 else ''} & {rot} & {int(l['n_comunidades'])} & {fmt(l['Q'])} & "
                          f"{fmt(l['join_count'])} & {fmt(l['join_count_aleatorio_esperado'])} & "
                          f"{fmt(l['cosseno_intra'])} & {fmt(l.get('ami_UF'))} & "
                          f"{fmt(l.get('ami_intermediaria'))} & {fmt(l.get('ami_imediata'))} \\\\")
        linhas.append(r"\midrule")
    linhas[-1] = r"\bottomrule"
    linhas.append(r"\end{tabular}")
    (OUT / "tab_communities_body.tex").write_text("\n".join(linhas))

    r = pd.read_csv(S / "robustez_moran_resumo.csv")
    mets = [("ami_UF", "AMI with states", 3), ("ami_intermediaria", "AMI with intermediate regions", 3),
            ("ami_imediata", "AMI with immediate regions", 3),
            ("bestest_contiguo_z_UF", "Contiguous BESTest $z$, states", 1),
            ("bestest_contiguo_z_intermediaria", "Contiguous BESTest $z$, intermediate", 1),
            ("bestest_contiguo_z_imediata", "Contiguous BESTest $z$, immediate", 1),
            ("join_count", "Join count", 3)]
    anos = sorted(r["exercicio"].unique())
    linhas = [r"\begin{tabular}{@{}l" + "cc" * len(anos) + r"@{}}", r"\toprule",
              " & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{a}}}" for a in anos) + r" \\",
              "".join(f"\\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(anos))),
              "Metric & " + " & ".join(["Observed & Null mean [95\\%]"] * len(anos)) + r" \\", r"\midrule"]
    for m, rot, d in mets:
        cel = []
        for a in anos:
            l = r[(r["exercicio"] == a) & (r["metrica"] == m)].iloc[0]
            cel.append(f"{l['observado']:.{d}f} & {l['nulo_media']:.{d}f} [{l['nulo_p2.5']:.{d}f}, {l['nulo_p97.5']:.{d}f}]")
        linhas.append(f"{rot} & " + " & ".join(cel) + r" \\")
    linhas += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "tab_moran_body.tex").write_text("\n".join(linhas))


def tab_perfis():
    import comum
    base = comum.carregar_base()
    base = base[base["Exercício"] == ANO]
    col = f"{ANO}_k{C.K_PADRAO}_ng"
    pe = pd.read_csv(S / "comunidades_pertenca.csv")[["codigo_ibge", col]].rename(columns={col: "com"})
    d = pe.merge(base, on="codigo_ibge", how="left")
    if d["UF"].isna().any():
        raise ValueError("municípios da partição sem dados na base")
    g = d.groupby("com").agg(n=("codigo_ibge", "size"), ufs=("UF", "nunique"),
                             decl=("Declarantes", "median"), renda=("Média rend. totais (R$)", "median"),
                             patr=("Média patrimônio (R$)", "median"), rural=("% Atividade rural", "median"),
                             rv=("% Renda variável", "median"))
    g["top"] = d.groupby("com")["UF"].apply(
        lambda s: ", ".join(f"{u} {c / len(s):.0%}".replace("%", "\\%") for u, c in s.value_counts().head(3).items()))
    g = g.sort_values("n", ascending=False).reset_index(drop=True)
    linhas = [r"\begin{tabular}{@{}lrrlrrrr@{}}", r"\toprule",
              r"& Mun. & States & Largest shares & Declarants & Income (R\$k) & Net worth (R\$k) & Rural (\%) \\",
              r"\midrule"]
    for i, l in g.iterrows():
        linhas.append(f"C{i + 1} & {l['n']} & {l['ufs']} & {l['top']} & {l['decl']:,.0f} & "
                      f"{l['renda'] / 1e3:.0f} & {l['patr'] / 1e3:.0f} & {100 * l['rural']:.1f} \\\\")
    linhas += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "tab_profiles_body.tex").write_text("\n".join(linhas))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fig_moran()
    fig_porosidade()
    fig_escala()
    mapa()
    tabelas()
    tab_perfis()
    print("Figuras e tabelas em", OUT)


if __name__ == "__main__":
    main()
