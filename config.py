from pathlib import Path

RAIZ = Path(__file__).resolve().parent

ARQUIVO_PAINEL = RAIZ / "painel_perfil_declarante_irpf.xlsx"
ABA_MUNICIPIO = "Município"

DIR_DADOS = RAIZ / "dados"
DIR_BRUTOS = DIR_DADOS / "brutos"
DIR_SAIDAS = RAIZ / "saidas"

URL_MALHA = ("https://geoftp.ibge.gov.br/organizacao_do_territorio/malhas_territoriais/"
             "malhas_municipais/municipio_2022/Brasil/BR/BR_Municipios_2022.zip")
URL_LOCALIDADES = "https://servicodados.ibge.gov.br/api/v1/localidades/municipios"
URL_POPULACAO = "https://apisidra.ibge.gov.br/values/t/4709/n6/all/v/93/p/2022"

CRS_METRICO = "EPSG:5880"

EXERCICIOS = [2025, 2026]

INDICADORES = {
    "Declarantes": "log",
    "% Simplificada": None,
    "% Renda variável": None,
    "Média de idade": None,
    "% Cônjuges": None,
    "% Feminino": None,
    "Média de dependentes": None,
    "Média desp. médicas (R$)": "log1p",
    "Média imposto devido (R$)": "log1p",
    "Média rend. totais (R$)": "log",
    "Média rend. tributáveis (R$)": "log",
    "Média patrimônio (R$)": "log1p",
    "% Atividade rural": None,
    "% Conta gov.br ouro/prata": None,
    "taxa_declarantes": None,
}

WINSORIZAR = (0.01, 0.99)

KS = [6, 9, 12]
K_PADRAO = 9
KNN_MODO = "uniao"

N_BINS_PADRAO = 50
N_BINS_SENSIBILIDADE = [25, 50, 100]

SEMENTES = 100
METODO_PARTICAO = "consenso"
CONSENSO_TAU = 0.5
CONSENSO_MAX_ITER = 10

FRONTEIRAS_SEMENTES_POR_K = 30
FRONTEIRAS_BOOTSTRAP = True
FRONTEIRAS_NULO = "ng"

BESTEST_PERMUTACOES = 1000
BESTEST_CONTIGUAS = 1000

ALFAS = [round(0.1 * i, 1) for i in range(11)]

SINT_N_REGIOES = 40
SINT_N_GEMEAS = 6
SINT_RHOS = [0.0, 0.4, 0.8]
SINT_SEPARACAO = 1.0
SINT_N_INDICADORES = 15
SINT_REPETICOES = 20

N_PROCESSOS = 8

MORAN_SUBSTITUTOS = 500
MORAN_K = K_PADRAO

GAMAS = [0.25, 0.4, 0.6, 0.8, 1.0, 1.25, 1.6, 2.0, 2.5, 3.2, 4.0, 5.0]

KS_SENSIBILIDADE = [5, 7, 9, 12, 15, 20, 30]
GRUPOS_INDICADORES = {
    "tamanho": ["Declarantes", "taxa_declarantes"],
    "renda_patrimonio": ["Média rend. totais (R$)", "Média rend. tributáveis (R$)",
                         "Média patrimônio (R$)", "% Renda variável"],
    "imposto_deducoes": ["Média imposto devido (R$)", "Média desp. médicas (R$)",
                         "% Simplificada"],
    "demografia": ["Média de idade", "% Feminino", "% Cônjuges", "Média de dependentes"],
    "outros": ["% Atividade rural", "% Conta gov.br ouro/prata"],
}

SEMENTE_GLOBAL = 42
