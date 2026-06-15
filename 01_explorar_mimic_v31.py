import os
import sys
import argparse
import gzip
import time
import warnings
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

# Configuração
OUTPUT_DIR = "exploracao_mimic31"
os.makedirs(OUTPUT_DIR, exist_ok=True)

LOG_LINES = []

def log(msg=""):
    print(msg)
    LOG_LINES.append(msg)

def salvar_relatorio():
    caminho = os.path.join(OUTPUT_DIR, "relatorio_exploracao.txt")
    with open(caminho, "w", encoding="utf-8") as f:
        f.write("\n".join(LOG_LINES))
    print(f"\nRelatório salvo em: {caminho}")

ARQUIVOS_ESPERADOS = {
    "hosp": [
        "admissions.csv.gz",
        "patients.csv.gz",
        "diagnoses_icd.csv.gz",
        "labevents.csv.gz",
        "d_labitems.csv.gz",         # dicionário de itens laboratoriais
        "d_icd_diagnoses.csv.gz",    # dicionário CID
        "procedures_icd.csv.gz",
        "prescriptions.csv.gz",
        "omr.csv.gz",                # outpatient med records (peso, altura, PA)
    ],
    "icu": [
        "icustays.csv.gz",
        "chartevents.csv.gz",
        "d_items.csv.gz",            # dicionário de itens do chartevents
        "inputevents.csv.gz",
        "outputevents.csv.gz",
        "procedureevents.csv.gz",
    ],
}

# itemids cardiovasculares a verificar no d_labitems
LAB_ITEMS_ESPERADOS = {
    50912: "Creatinine",
    51006: "Urea Nitrogen",
    51222: "Hemoglobin",
    51301: "White Blood Cells",
    50983: "Sodium",
    50971: "Potassium",
    50931: "Glucose",
    50907: "Cholesterol, Total",
    50904: "Cholesterol, HDL",
    50902: "Cholesterol, LDL",
    50905: "Triglycerides",
    51002: "Troponin T",
    51003: "Troponin I",
    50963: "BNP",
    50995: "NT-proBNP",
    51248: "MCV",
    50960: "Magnesium",
}

CHART_ITEMS_ESPERADOS = {
    220045: "Heart Rate",
    220179: "Non Invasive Blood Pressure systolic",
    220180: "Non Invasive Blood Pressure diastolic",
    220210: "Respiratory Rate",
    223761: "Temperature Fahrenheit",
    220277: "SpO2",
    226512: "Admission Weight (Kg)",
    226730: "Height (cm)",
}

# CID-10 cardiovasculares
CID10_DCV_PATTERNS = {
    "Doença Isquêmica (I20–I25)":      r"^I2[0-5]",
    "IAM (I21–I22)":                   r"^I2[12]",
    "Insuficiência Cardíaca (I50)":    r"^I50",
    "Fibrilação Atrial (I48)":         r"^I48",
    "AVC Isquêmico (I63)":             r"^I63",
    "AVC Hemorrágico (I60–I62)":       r"^I6[012]",
    "AIT (G45)":                       r"^G45",
    "Hipertensão (I10–I15)":           r"^I1[0-5]",
    "Aterosclerose (I70)":             r"^I70",
    "Doença Arterial Periférica(I73)": r"^I73",
    "Cardiopatia Reumática (I05–I09)": r"^I0[5-9]",
}

def ler_csv_gz(caminho: str, **kwargs) -> pd.DataFrame:
    if caminho.endswith(".gz"):
        return pd.read_csv(caminho, compression="gzip", **kwargs)
    return pd.read_csv(caminho, **kwargs)


def tamanho_legivel(n_bytes: int) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if n_bytes < 1024:
            return f"{n_bytes:.1f} {unit}"
        n_bytes /= 1024
    return f"{n_bytes:.1f} TB"


# Verificação de arquivos
def verificar_arquivos(mimic_path: str) -> dict:
    encontrados = {}
    for modulo, arquivos in ARQUIVOS_ESPERADOS.items():
        pasta = os.path.join(mimic_path, modulo)
        log(f"\n  Módulo: {modulo}/ (pasta: {pasta})")

        if not os.path.isdir(pasta):
            log(f" Pasta não encontrada: {pasta}")
            continue

        # listar o que realmente há na pasta
        todos = sorted(os.listdir(pasta))
        log(f"  Arquivos presentes ({len(todos)} total):")
        for arq in todos:
            caminho_completo = os.path.join(pasta, arq)
            tam = os.path.getsize(caminho_completo)
            log(f"    {if arq in arquivos else ' '} {arq:<40} {tamanho_legivel(tam)}")
            if arq in arquivos:
                encontrados[f"{modulo}/{arq}"] = caminho_completo

    log(f"\n  Arquivos esperados encontrados: {len(encontrados)} / "
        f"{sum(len(v) for v in ARQUIVOS_ESPERADOS.values())}")
    return encontrados

def verificar_colunas(encontrados: dict):
    chaves_principais = [
        "hosp/admissions.csv.gz",
        "hosp/patients.csv.gz",
        "hosp/diagnoses_icd.csv.gz",
        "hosp/labevents.csv.gz",
        "hosp/omr.csv.gz",
        "icu/icustays.csv.gz",
        "icu/chartevents.csv.gz",
    ]

    for chave in chaves_principais:
        if chave not in encontrados:
            log(f"\n {chave} — não encontrado, pulando.")
            continue

        caminho = encontrados[chave]
        log(f"\n {chave}")
        try:
            df_head = ler_csv_gz(caminho, nrows=3)
            log(f"  Colunas ({len(df_head.columns)}): {list(df_head.columns)}")
            log(f"  Dtypes:\n{df_head.dtypes.to_string()}")
            log(f"  Amostra:\n{df_head.to_string()}")
        except Exception as e:
            log(f"  ERRO ao ler: {e}")


# Dicionário de itens laboratoriais
def verificar_labitems(encontrados: dict) -> pd.DataFrame:
    chave = "hosp/d_labitems.csv.gz"
    if chave not in encontrados:
        log("d_labitems.csv.gz não encontrado.")
        return pd.DataFrame()

    d_lab = ler_csv_gz(encontrados[chave])
    log(f"  Total de itens no dicionário: {len(d_lab)}")
    log(f"  Colunas: {list(d_lab.columns)}")

    col_id   = "itemid"   if "itemid"   in d_lab.columns else d_lab.columns[0]
    col_lbl  = "label"    if "label"    in d_lab.columns else d_lab.columns[1]
    col_fluid = "fluid"   if "fluid"    in d_lab.columns else None
    col_cat   = "category" if "category" in d_lab.columns else None

    log(f"\n  {'itemid':<10} {'Encontrado?':<14} {'Label no dicionário'}")
    log(f"  {'-'*10} {'-'*14} {'-'*35}")

    encontrados_lab = []
    for iid, nome_esperado in LAB_ITEMS_ESPERADOS.items():
        linha = d_lab[d_lab[col_id] == iid]
        if len(linha) == 0:
            log(f"  {iid:<10} {'NÃO ENCONTRADO':<14} (esperado: {nome_esperado})")
        else:
            label_real = linha[col_lbl].values[0]
            log(f"  {iid:<10} {'OK':<14} {label_real}")
            encontrados_lab.append(iid)

    log(f"\n  itemids encontrados: {len(encontrados_lab)} / {len(LAB_ITEMS_ESPERADOS)}")

    nao_encontrados = [iid for iid in LAB_ITEMS_ESPERADOS if iid not in encontrados_lab]
    if nao_encontrados:
        log("\n  Buscando por nome (para os não encontrados por id):")
        termos = ["troponin", "bnp", "proBNP", "cholesterol", "triglyceride",
                  "ldl", "hdl", "creatinine"]
        for termo in termos:
            mask = d_lab[col_lbl].str.lower().str.contains(termo.lower(), na=False)
            hits = d_lab[mask][[col_id, col_lbl]]
            if len(hits):
                log(f"\n    Busca '{termo}':")
                log(hits.to_string(index=False))

    # Salvar dicionário completo filtrado
    d_lab.to_csv(os.path.join(OUTPUT_DIR, "d_labitems_completo.csv"), index=False)
    log(f"\n  d_labitems completo salvo em: {OUTPUT_DIR}/d_labitems_completo.csv")
    return d_lab


# Dicionário de itens do chartevents (sinais vitais)
def verificar_chart_items(encontrados: dict):
    chave = "icu/d_items.csv.gz"
    if chave not in encontrados:
        log("d_items.csv.gz não encontrado.")
        return

    d_items = ler_csv_gz(encontrados[chave])
    log(f"  Total de itens: {len(d_items)}")
    log(f"  Colunas: {list(d_items.columns)}")

    col_id  = "itemid" if "itemid" in d_items.columns else d_items.columns[0]
    col_lbl = "label"  if "label"  in d_items.columns else d_items.columns[1]

    log(f"\n  {'itemid':<10} {'Encontrado?':<14} {'Label no dicionário'}")
    log(f"  {'-'*10} {'-'*14} {'-'*40}")

    for iid, nome_esperado in CHART_ITEMS_ESPERADOS.items():
        linha = d_items[d_items[col_id] == iid]
        if len(linha) == 0:
            log(f"  {iid:<10} {'NÃO ENCONTRADO':<14} (esperado: {nome_esperado})")
            # Busca por nome
            mask = d_items[col_lbl].str.lower().str.contains(
                nome_esperado.split()[0].lower(), na=False)
            alternativas = d_items[mask][[col_id, col_lbl]].head(5)
            if len(alternativas):
                log(f" Alternativas encontradas:")
                for _, row in alternativas.iterrows():
                    log(f"itemid={row[col_id]} = {row[col_lbl]}")
        else:
            log(f"  {iid:<10} {'OK':<14} {linha[col_lbl].values[0]}")

    d_items.to_csv(os.path.join(OUTPUT_DIR, "d_items_icu_completo.csv"), index=False)
    log(f"\n  d_items completo salvo em: {OUTPUT_DIR}/d_items_icu_completo.csv")


# Pacientes e admissões
def verificar_populacao(encontrados: dict):
    # Admissões
    chave_adm = "hosp/admissions.csv.gz"
    if chave_adm in encontrados:
        adm = ler_csv_gz(encontrados[chave_adm])
        log(f"\n  admissions.csv.gz")
        log(f"  Total de admissões (hadm_id únicos): {adm['hadm_id'].nunique()}")
        log(f"  Total de pacientes (subject_id únicos): {adm['subject_id'].nunique()}")
        log(f"  Colunas: {list(adm.columns)}")

        if "admission_type" in adm.columns:
            log(f"\n  Tipos de admissão:")
            log(adm["admission_type"].value_counts().to_string())

        if "hospital_expire_flag" in adm.columns:
            n_obitos = adm["hospital_expire_flag"].sum()
            log(f"\n  Óbitos hospitalares: {n_obitos} ({n_obitos/len(adm)*100:.1f}%)")

    # Pacientes
    chave_pat = "hosp/patients.csv.gz"
    if chave_pat in encontrados:
        pat = ler_csv_gz(encontrados[chave_pat])
        log(f"\n  patients.csv.gz")
        log(f"  Total de pacientes: {len(pat)}")
        log(f"  Colunas: {list(pat.columns)}")

        # Coluna de idade pode ser anchor_age ou age
        col_idade = "anchor_age" if "anchor_age" in pat.columns else (
                    "age"        if "age"        in pat.columns else None)
        col_sexo  = "gender"     if "gender"     in pat.columns else None

        if col_idade:
            adultos = pat[pat[col_idade] >= 18]
            log(f"\n  Distribuição de idade (≥18 anos, n={len(adultos)}):")
            log(f"    Mín: {adultos[col_idade].min():.0f} | "
                f"Mediana: {adultos[col_idade].median():.0f} | "
                f"Máx: {adultos[col_idade].max():.0f} | "
                f"Média: {adultos[col_idade].mean():.1f} ± {adultos[col_idade].std():.1f}")

        if col_sexo:
            log(f"\n  Distribuição de sexo:")
            log(pat[col_sexo].value_counts().to_string())

    # ICU stays
    chave_icu = "icu/icustays.csv.gz"
    if chave_icu in encontrados:
        icu = ler_csv_gz(encontrados[chave_icu])
        log(f"\n  icustays.csv.gz")
        log(f"  Total de estadias na UTI: {len(icu)}")
        log(f"  Colunas: {list(icu.columns)}")
        if "first_careunit" in icu.columns:
            log(f"\n  Unidades de cuidado:")
            log(icu["first_careunit"].value_counts().to_string())


# Diagnósticos cardiovasculares (CID-10)
def verificar_diagnosticos(encontrados: dict):
    chave = "hosp/diagnoses_icd.csv.gz"
    if chave not in encontrados:
        log("diagnoses_icd.csv.gz não encontrado.")
        return

    diag = ler_csv_gz(encontrados[chave])
    log(f"  Total de linhas em diagnoses_icd: {len(diag):,}")
    log(f"  Colunas: {list(diag.columns)}")

    col_code = "icd_code"    if "icd_code"    in diag.columns else diag.columns[2]
    col_ver  = "icd_version" if "icd_version" in diag.columns else None

    diag[col_code] = diag[col_code].astype(str).str.strip()

    # Filtrar apenas CID-10
    if col_ver:
        diag10 = diag[diag[col_ver] == 10].copy()
        diag9  = diag[diag[col_ver] == 9].copy()
        log(f"  CID-10: {len(diag10):,} registros | CID-9: {len(diag9):,} registros")
    else:
        diag10 = diag.copy()

    log(f"\n  {'Categoria':<40} {'Admissões únicas':>18} {'% do total':>12}")
    log(f"  {'-'*40} {'-'*18} {'-'*12}")

    total_hadm = diag["hadm_id"].nunique()
    todos_dcv_hadm = set()

    for descricao, padrao in CID10_DCV_PATTERNS.items():
        mask = diag10[col_code].str.match(padrao, na=False)
        hadm_set = set(diag10.loc[mask, "hadm_id"].unique())
        todos_dcv_hadm |= hadm_set
        pct = len(hadm_set) / total_hadm * 100
        log(f"  {descricao:<40} {len(hadm_set):>18,} {pct:>11.1f}%")

    log(f"\n  {'TOTAL DCV':<40} "
        f"{len(todos_dcv_hadm):>18,} "
        f"{len(todos_dcv_hadm)/total_hadm*100:>11.1f}%")
    log(f"  Total de hadm_id únicos na base: {total_hadm:,}")

    mask_dcv_total = diag10[col_code].str.match(
        r"^I|^G45", na=False)
    top_cid = (diag10.loc[mask_dcv_total, col_code]
               .value_counts()
               .head(20))
    log(top_cid.to_string())

    # Salvar
    pd.DataFrame({
        "categoria": list(CID10_DCV_PATTERNS.keys()),
    }).to_csv(os.path.join(OUTPUT_DIR, "diagnosticos_dcv_summary.csv"), index=False)


# Amostragem do labevents
def verificar_labevents_amostra(encontrados: dict):
    chave = "hosp/labevents.csv.gz"
    if chave not in encontrados:
        log("labevents.csv.gz não encontrado.")
        return

    caminho = encontrados[chave]
    tam = os.path.getsize(caminho)
    log(f"Tamanho do arquivo: {tamanho_legivel(tam)}")

    t0 = time.time()
    chunk = ler_csv_gz(caminho, nrows=500_000)
    log(f"  Tempo: {time.time()-t0:.1f}s")

    log(f"  Colunas: {list(chunk.columns)}")
    log(f"  Amostra:\n{chunk.head(5).to_string()}")

    col_iid = "itemid" if "itemid" in chunk.columns else chunk.columns[1]
    col_val = "valuenum" if "valuenum" in chunk.columns else None
    col_hadm = "hadm_id" if "hadm_id" in chunk.columns else None

    ids_presentes = set(chunk[col_iid].unique())
    for iid, nome in LAB_ITEMS_ESPERADOS.items():
        encontrado = if iid in ids_presentes
        log(f"    {encontrado} {iid} — {nome}")

    if col_val:
        log(f"\n  Estatísticas de valuenum (amostra):")
        log(f"    Nulos: {chunk[col_val].isna().sum():,} "
            f"({chunk[col_val].isna().mean()*100:.1f}%)")
        log(f"    Mín: {chunk[col_val].min():.4f} | "
            f"Máx: {chunk[col_val].max():.4f} | "
            f"Média: {chunk[col_val].mean():.4f}")

    colunas_tempo = [c for c in chunk.columns
                     if any(t in c.lower() for t in ["time","date"])]
    log(f"\n  Colunas de tempo encontradas: {colunas_tempo}")


# OMR (Outpatient Medical Records)
def verificar_omr(encontrados: dict):
    chave = "hosp/omr.csv.gz"
    if chave not in encontrados:
        log("omr.csv.gz não encontrado")
        return

    omr = ler_csv_gz(encontrados[chave])
    log(f"  Total de registros: {len(omr):,}")
    log(f"  Colunas: {list(omr.columns)}")

    if "result_name" in omr.columns:
        log(f"\n  Tipos de resultado disponíveis (result_name):")
        log(omr["result_name"].value_counts().head(30).to_string())

    # Verificar se contém pressão arterial, peso, altura
    campos_uteis = ["blood pressure", "weight", "height", "bmi", "eGFR"]
    if "result_name" in omr.columns:
        for campo in campos_uteis:
            mask = omr["result_name"].str.lower().str.contains(campo.lower(), na=False)
            n = mask.sum()
            log(f"    '{campo}': {n:,} registros")


# Amostra do chartevents
def verificar_chartevents_amostra(encontrados: dict):
    chave = "icu/chartevents.csv.gz"
    if chave not in encontrados:
        log("chartevents.csv.gz não encontrado.")
        return

    caminho = encontrados[chave]
    tam = os.path.getsize(caminho)
    log(f"  Tamanho: {tamanho_legivel(tam)}")

    t0 = time.time()
    chunk = ler_csv_gz(caminho, nrows=200_000)
    log(f"  Tempo: {time.time()-t0:.1f}s")
    log(f"  Colunas: {list(chunk.columns)}")
    log(f"  Amostra:\n{chunk.head(5).to_string()}")

    col_iid = "itemid" if "itemid" in chunk.columns else chunk.columns[2]
    ids_presentes = set(chunk[col_iid].unique())

    log(f"\n itemids de sinais vitais encontrados na amostra:")
    for iid, nome in CHART_ITEMS_ESPERADOS.items():
        encontrado = if iid in ids_presentes
        log(f"    {encontrado} {iid} — {nome}")

def main():
    parser = argparse.ArgumentParser(
        description="Exploração e diagnóstico do MIMIC-IV"
    )
    parser.add_argument(
        "--mimic",
        type=str,
        required=True,
        help="Caminho para a pasta raiz do MIMIC-IV",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.mimic):
        print(f"ERRO: pasta não encontrada: {args.mimic}")
        sys.exit(1)

    log(f"MIMIC-IV — Exploração e Diagnóstico")
    log(f"Pasta base: {args.mimic}")

    encontrados = verificar_arquivos(args.mimic)
    verificar_colunas(encontrados)
    verificar_labitems(encontrados)
    verificar_chart_items(encontrados)
    verificar_populacao(encontrados)
    verificar_diagnosticos(encontrados)
    verificar_labevents_amostra(encontrados)
    verificar_omr(encontrados)
    verificar_chartevents_amostra(encontrados)

    salvar_relatorio()


if __name__ == "__main__":
    main()
