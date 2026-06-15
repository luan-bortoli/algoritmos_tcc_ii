import os
import sys
import time
import argparse
import warnings
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model    import LogisticRegression
from sklearn.svm             import SVC
from sklearn.tree            import DecisionTreeClassifier, export_text, plot_tree
from sklearn.neighbors       import KNeighborsClassifier
from sklearn.preprocessing   import StandardScaler
from sklearn.impute          import SimpleImputer
from sklearn.pipeline        import Pipeline
from sklearn.model_selection import (StratifiedKFold, cross_validate,
                                     GridSearchCV, train_test_split)
from sklearn.metrics         import (accuracy_score, precision_score,
                                     recall_score, f1_score, roc_auc_score,
                                     roc_curve, confusion_matrix,
                                     ConfusionMatrixDisplay, classification_report)
from sklearn.inspection      import permutation_importance

warnings.filterwarnings("ignore")
np.random.seed(42)

# CONFIGURAÇÕES
OUTPUT_DIR   = "resultados_ml_mimic31"
K_FOLDS      = 10
TEST_SIZE    = 0.30
RANDOM_STATE = 42
CHUNK_SIZE   = 500_000

# Limites de amostragem para modelos lentos
SVM_MAX_HPO      = 10_000   # amostras para GridSearch do SVM
KNN_MAX_AMOSTRAS = 50_000   # amostras para treino+KFold do KNN

os.makedirs(OUTPUT_DIR, exist_ok=True)

LAB_ITEMS = {
    50912: "creatinina",
    51006: "ureia",
    51222: "hemoglobina",
    51301: "leucocitos",
    50983: "sodio",
    50971: "potassio",
    50931: "glicose",
    50907: "colesterol_total",
    50904: "hdl",
    50905: "ldl",
    51002: "troponina_i",
    51003: "troponina_t",
    50963: "nt_probnp",
    51248: "mch",
    50960: "magnesio",
}

CHART_ITEMS = {
    220045: "fc",
    220179: "pas",
    220180: "pad",
    220210: "fr",
    223761: "temperatura_f",
    220277: "spo2",
    226512: "peso_kg",
    226730: "altura_cm",
}

CID10_PATTERN = r"^I21|^I22|^I50"
CID9_PATTERN  = r"^410|^428"

FEATURES = [
    "idade", "sexo_m",
    "fc", "pas", "pad", "fr", "spo2",
    "troponina_i", "troponina_t", "nt_probnp",
    "creatinina", "ureia", "hemoglobina", "leucocitos",
    "sodio", "potassio", "glicose", "mch", "magnesio",
    "colesterol_total", "hdl", "ldl",
    "proporcao_hdl_ldl", "imc",
]

SCORING = {
    "accuracy":  "accuracy",
    "precision": "precision",
    "recall":    "recall",
    "f1":        "f1",
    "roc_auc":   "roc_auc",
}

CORES = {
    "Regressao Logistica": "#E63946",
    "Arvore de Decisao":   "#2A9D8F",
    "KNN":                 "#E9C46A",
    "SVM":                 "#457B9D",
}

# 1. CARREGAMENTO DO MIMIC-IV
def _ler_csv(caminho: str, **kwargs) -> pd.DataFrame:
    if not os.path.exists(caminho):
        alt = caminho.replace(".gz", "")
        if os.path.exists(alt):
            caminho = alt
        else:
            raise FileNotFoundError(f"Arquivo não encontrado: {caminho}")
    comp = "gzip" if caminho.endswith(".gz") else None
    return pd.read_csv(caminho, compression=comp, **kwargs)


def _ler_chunks(caminho, usecols, filtro_col, filtro_vals, chunk_size):
    comp = "gzip" if caminho.endswith(".gz") else None
    partes, total, t0 = [], 0, time.time()
    for i, chunk in enumerate(pd.read_csv(
        caminho, compression=comp,
        usecols=usecols, chunksize=chunk_size, low_memory=False
    )):
        filtrado = chunk[chunk[filtro_col].isin(filtro_vals)]
        partes.append(filtrado)
        total += len(chunk)
        if (i + 1) % 10 == 0:
            print(f"    … {total:,} linhas lidas, "
                  f"{sum(len(p) for p in partes):,} selecionadas "
                  f"({time.time()-t0:.0f}s)")
    resultado = pd.concat(partes, ignore_index=True) if partes else pd.DataFrame()
    print(f"Concluído: {total:,} lidas {len(resultado):,} selecionadas "
          f"({time.time()-t0:.0f}s)")
    return resultado


def carregar_admissoes(hosp):
    adm = _ler_csv(
        os.path.join(hosp, "admissions.csv.gz"),
        usecols=["subject_id", "hadm_id", "admittime", "dischtime",
                 "hospital_expire_flag", "admission_type"],
        parse_dates=["admittime", "dischtime"],
    )
    print(f"  {len(adm):,} admissões | {adm['subject_id'].nunique():,} pacientes")
    return adm


def carregar_pacientes(hosp):
    pat = _ler_csv(
        os.path.join(hosp, "patients.csv.gz"),
        usecols=["subject_id", "gender", "anchor_age", "anchor_year"],
    )
    pat.rename(columns={"anchor_age": "idade"}, inplace=True)
    pat["sexo_m"] = (pat["gender"].str.upper() == "M").astype(int)
    pat = pat[pat["idade"] >= 18]
    print(f"  {len(pat):,} pacientes adultos")
    return pat[["subject_id", "idade", "sexo_m"]]


def construir_labels(hosp, hadm_ids):
    diag = _ler_csv(
        os.path.join(hosp, "diagnoses_icd.csv.gz"),
        usecols=["hadm_id", "icd_code", "icd_version"],
    )
    diag["icd_code"] = diag["icd_code"].astype(str).str.strip()
    mask10 = (diag["icd_version"] == 10) & diag["icd_code"].str.match(CID10_PATTERN)
    mask9  = (diag["icd_version"] == 9)  & diag["icd_code"].str.match(CID9_PATTERN)
    dcv_hadm = set(diag.loc[mask10 | mask9, "hadm_id"].unique())
    labels = pd.Series({h: int(h in dcv_hadm) for h in hadm_ids}, name="label")
    n_pos = labels.sum()
    print(f"  DCV positivo: {n_pos:,} ({n_pos/len(labels)*100:.1f}%)")
    return labels


def carregar_labevents(hosp, hadm_ids, chunk_size):
    print("[4/8] Carregando labevents …")
    caminho = os.path.join(hosp, "labevents.csv.gz")
    lab = _ler_chunks(caminho, usecols=["hadm_id","itemid","valuenum","storetime"],
                      filtro_col="itemid", filtro_vals=set(LAB_ITEMS.keys()),
                      chunk_size=chunk_size)
    if lab.empty:
        return pd.DataFrame(columns=["hadm_id"])
    lab = lab.dropna(subset=["hadm_id","valuenum"])
    lab["hadm_id"] = lab["hadm_id"].astype(int)
    lab = lab[lab["hadm_id"].isin(hadm_ids)]
    lab["feature"] = lab["itemid"].map(LAB_ITEMS)
    limites = {
        "creatinina": (0.1,30), "ureia": (1,300), "hemoglobina": (2,25),
        "leucocitos": (0.1,100), "sodio": (100,180), "potassio": (1.5,10),
        "glicose": (20,2000), "colesterol_total": (50,600),
        "hdl": (5,200), "ldl": (10,500), "troponina_t": (0,100),
        "troponina_i": (0,200), "nt_probnp": (0,150000),
        "mch": (20,40), "magnesio": (0.5,5),
    }
    for feat, (lo, hi) in limites.items():
        m = lab["feature"] == feat
        lab.loc[m & ((lab["valuenum"] < lo) | (lab["valuenum"] > hi)), "valuenum"] = np.nan
    lab_pivot = (lab.groupby(["hadm_id","feature"])["valuenum"]
                 .median().unstack("feature").reset_index())
    print(f"  {len(lab_pivot):,} admissões com dados laboratoriais")
    return lab_pivot


def carregar_chartevents(icu, hadm_ids, chunk_size):
    caminho = os.path.join(icu, "chartevents.csv.gz")
    chart = _ler_chunks(caminho, usecols=["hadm_id","itemid","valuenum","charttime"],
                        filtro_col="itemid", filtro_vals=set(CHART_ITEMS.keys()),
                        chunk_size=chunk_size)
    if chart.empty:
        return pd.DataFrame(columns=["hadm_id"])
    chart = chart.dropna(subset=["hadm_id","valuenum"])
    chart["hadm_id"] = chart["hadm_id"].astype(int)
    chart = chart[chart["hadm_id"].isin(hadm_ids)]
    chart["feature"] = chart["itemid"].map(CHART_ITEMS)
    limites_chart = {
        "fc": (20,300), "pas": (40,280), "pad": (10,200),
        "fr": (4,70), "temperatura_f": (86,115),
        "spo2": (50,100), "peso_kg": (20,400), "altura_cm": (100,250),
    }
    for feat, (lo, hi) in limites_chart.items():
        m = chart["feature"] == feat
        chart.loc[m & ((chart["valuenum"] < lo) | (chart["valuenum"] > hi)), "valuenum"] = np.nan
    chart_pivot = (chart.groupby(["hadm_id","feature"])["valuenum"]
                   .median().unstack("feature").reset_index())
    if "temperatura_f" in chart_pivot.columns:
        chart_pivot["temperatura_c"] = (chart_pivot["temperatura_f"] - 32) * 5 / 9
        chart_pivot.drop(columns=["temperatura_f"], inplace=True)
    print(f"  {len(chart_pivot):,} admissões com sinais vitais")
    return chart_pivot


def carregar_omr(hosp, subject_ids):
    caminho = os.path.join(hosp, "omr.csv.gz")
    if not os.path.exists(caminho):
        print("omr.csv.gz não encontrado")
        return pd.DataFrame()
    try:
        omr = _ler_csv(caminho)
    except Exception as e:
        print(f"Erro ao ler OMR: {e}")
        return pd.DataFrame()
    if "result_name" not in omr.columns:
        return pd.DataFrame()
    campos = {"weight": "peso_kg_omr", "height": "altura_cm_omr",
              "bmi": "imc_omr", "blood pressure": "pa_omr"}
    frames = []
    for termo, col_saida in campos.items():
        mask = omr["result_name"].str.lower().str.contains(termo, na=False)
        sub = omr[mask][["subject_id","result_value"]].copy()
        sub.rename(columns={"result_value": col_saida}, inplace=True)
        sub[col_saida] = pd.to_numeric(sub[col_saida], errors="coerce")
        frames.append(sub.dropna())
    if not frames:
        return pd.DataFrame()
    omr_agg = None
    for frame in frames:
        col = [c for c in frame.columns if c != "subject_id"][0]
        agg = frame.groupby("subject_id")[col].median().reset_index()
        omr_agg = agg if omr_agg is None else omr_agg.merge(agg, on="subject_id", how="outer")
    print(f"  {len(omr_agg):,} pacientes com dados OMR")
    return omr_agg


def montar_dataset(mimic_path, chunk_size):
    hosp = os.path.join(mimic_path, "hosp")
    icu  = os.path.join(mimic_path, "icu")
    # Admissões base
    adm = carregar_admissoes(hosp)
    hadm_ids    = set(adm["hadm_id"].unique())
    subject_ids = set(adm["subject_id"].unique())
    # Pacientes
    pat    = carregar_pacientes(hosp)
    # Labels
    labels = construir_labels(hosp, hadm_ids)
    # Dados laboratoriais
    lab    = carregar_labevents(hosp, hadm_ids, chunk_size)
    # Sinais vitais
    chart  = carregar_chartevents(icu, hadm_ids, chunk_size)
    # OMR
    omr    = carregar_omr(hosp, subject_ids)

    #Junção de tabelas
    df = adm[["subject_id","hadm_id"]].copy()
    df = df.merge(pat, on="subject_id", how="left")
    df["label"] = df["hadm_id"].map(labels)
    if not lab.empty:
        df = df.merge(lab, on="hadm_id", how="left")
    if not chart.empty:
        df = df.merge(chart, on="hadm_id", how="left")
    if not omr.empty and "subject_id" in omr.columns:
        df = df.merge(omr, on="subject_id", how="left")
        for col_chart, col_omr in [("peso_kg","peso_kg_omr"),("altura_cm","altura_cm_omr")]:
            if col_chart in df.columns and col_omr in df.columns:
                df[col_chart].fillna(df[col_omr], inplace=True)

    if "hdl" in df.columns and "ldl" in df.columns:
        df["proporcao_hdl_ldl"] = df["hdl"] / df["ldl"].replace(0, np.nan)
    if "peso_kg" in df.columns and "altura_cm" in df.columns:
        alt_m = df["altura_cm"] / 100
        df["imc"] = df["peso_kg"] / (alt_m ** 2)
        df.loc[(df["imc"] < 10) | (df["imc"] > 80), "imc"] = np.nan

    # Limpesa final e montagem do dataset limpo
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    print(f"Dataset final: {len(df):,} admissões | "
          f"DCV+: {df['label'].sum():,} ({df['label'].mean()*100:.1f}%) | "
          f"Colunas: {len(df.columns)}")

    cols_salvar = ["hadm_id","subject_id","label"] + \
                  [c for c in FEATURES if c in df.columns]
    df[cols_salvar].to_csv(
        os.path.join(OUTPUT_DIR, "dataset_consolidado.csv.gz"),
        index=False, compression="gzip")
    print(f"Dataset salvo em: {OUTPUT_DIR}/dataset_consolidado.csv.gz")
    return df

# 2. SELEÇÃO DE FEATURES
def selecionar_features(df):
    cols = [c for c in FEATURES if c in df.columns]
    ausentes = [c for c in FEATURES if c not in df.columns]
    if ausentes:
        print(f"\n Features não disponíveis: {ausentes}")
    X = df[cols].copy()
    y = df["label"].values
    print(f"\n  Features selecionadas ({len(cols)}): {cols}")
    return X, y, cols

def salvar(fig, nome_arquivo):
    caminho = os.path.join(OUTPUT_DIR, nome_arquivo)
    fig.savefig(caminho, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Salvo: {caminho}")

def plot_distribuicao_classes(y):
    vals, counts = np.unique(y, return_counts=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(["Sem DCV","Com DCV"], counts,
                  color=["#457B9D","#E63946"], edgecolor="white")
    for bar, cnt in zip(bars, counts):
        ax.text(bar.get_x()+bar.get_width()/2,
                bar.get_height()+counts.max()*0.015,
                f"{cnt:,}\n({cnt/counts.sum()*100:.1f}%)",
                ha="center", fontsize=10, fontweight="bold")
    ax.set_title("Distribuição das Classes", fontsize=13)
    ax.set_ylabel("Admissões")
    ax.set_ylim(0, counts.max()*1.2)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    salvar(fig, "distribuicao_classes.png")

def plot_missing_heatmap(X):
    fig, ax = plt.subplots(figsize=(12, 5))
    missing_pct = X.isna().mean().sort_values(ascending=False) * 100
    missing_pct = missing_pct[missing_pct > 0]
    if missing_pct.empty:
        plt.close(fig)
        return
    ax.bar(missing_pct.index, missing_pct.values, color="#E63946", alpha=0.8)
    ax.set_title("Porcentagem de Dados Ausentes por Feature")
    ax.set_ylabel("% Ausente")
    ax.set_xticklabels(missing_pct.index, rotation=45, ha="right")
    ax.axhline(20, color="orange", ls="--", lw=1, label="Limiar 20%")
    ax.axhline(50, color="red",    ls="--", lw=1, label="Limiar 50%")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    salvar(fig, "dados_ausentes.png")


# 4. DEFINIÇÃO DOS MODELOS
def construir_pipeline_modelo(clf, usar_scaler=True):
    etapas = [("imputer", SimpleImputer(strategy="median"))]
    if usar_scaler:
        etapas.append(("scaler", StandardScaler()))
    etapas.append(("clf", clf))
    return Pipeline(etapas)


def definir_modelos():
    return {
        "Regressao Logistica": (
            construir_pipeline_modelo(
                LogisticRegression(
                    max_iter=3000, solver="saga",
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                )
            ),
            {"clf__C": [0.01, 0.1, 1, 10], "clf__penalty": ["l2"]},
        ),
        "Arvore de Decisao": (
            construir_pipeline_modelo(
                DecisionTreeClassifier(criterion="gini", random_state=RANDOM_STATE),
                usar_scaler=False,
            ),
            {
                "clf__max_depth":        [5, 10, 15, None],
                "clf__min_samples_leaf": [5, 20, 50],
                "clf__class_weight":     [None, "balanced"],
            },
        ),
        "KNN": (
            construir_pipeline_modelo(
                KNeighborsClassifier(metric="euclidean", n_jobs=-1)
            ),
            {
                "clf__n_neighbors": [5, 11, 21, 31, 51],
                "clf__weights":     ["uniform", "distance"],
            },
        ),
        "SVM": (
            construir_pipeline_modelo(
                SVC(
                    kernel="rbf", probability=True,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    cache_size=1000,
                )
            ),
            {"clf__C": [0.1, 1, 10], "clf__gamma": ["scale", 0.01]},
        ),
    }


# 5. AMOSTRAGEM ESTRATIFICADA (para KNN e SVM)
def amostrar_estratificado(x, y, n_amostras, seed=RANDOM_STATE):
    rng = np.random.default_rng(seed)
    idx = []
    for classe in np.unique(y):
        idx_classe = np.where(y == classe)[0]
        n_classe   = int(n_amostras * len(idx_classe) / len(y))
        n_classe   = max(n_classe, 100)
        escolhidos = rng.choice(idx_classe,
                                size=min(n_classe, len(idx_classe)),
                                replace=False)
        idx.extend(escolhidos.tolist())
    return np.array(idx)


# 6. TREINAMENTO, AVALIAÇÃO E SALVAMENTO
def treinar_avaliar(nome, pipeline, grade, X_train, y_train, X_test, y_test):
    print(f"\n{'═'*65}")
    print(f"  [{nome}]")
    print(f"{'═'*65}")

    cv_inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    # Amostragem para HPO
    if nome == "SVM" and len(y_train) > SVM_MAX_HPO:
        print(f"   SVM: usando amostra estratificada de {SVM_MAX_HPO:,} "
              f"amostras para GridSearchCV.")
        idx_hpo = amostrar_estratificado(X_train, y_train, SVM_MAX_HPO)
        X_hpo, y_hpo = X_train[idx_hpo], y_train[idx_hpo]
        print(f"Amostra HPO: {len(y_hpo):,} (DCV+: {y_hpo.mean()*100:.1f}%)")
    elif nome == "KNN" and len(y_train) > KNN_MAX_AMOSTRAS:
        print(f"   KNN: limitando treino+HPO a {KNN_MAX_AMOSTRAS:,} amostras "
              f"(escala O(n) na predição).")
        idx_hpo = amostrar_estratificado(X_train, y_train, KNN_MAX_AMOSTRAS)
        X_hpo, y_hpo = X_train[idx_hpo], y_train[idx_hpo]
        print(f"Amostra KNN: {len(y_hpo):,} (DCV+: {y_hpo.mean()*100:.1f}%)")
    else:
        X_hpo, y_hpo = X_train, y_train

    # GridSearchCV
    n_combinacoes = 1
    for v in grade.values():
        n_combinacoes *= len(v)
    print(f"GridSearchCV 5-fold | {n_combinacoes} combinações "
          f"= {n_combinacoes*5} fits …")
    t0 = time.time()
    gs = GridSearchCV(
        pipeline, grade,
        cv=cv_inner, scoring="roc_auc",
        n_jobs=-1, refit=True, verbose=1,
    )
    gs.fit(X_hpo, y_hpo)
    print(f"Melhores parâmetros : {gs.best_params_}")
    print(f"AUC-ROC (HPO)       : {gs.best_score_:.4f} "
          f"(em {time.time()-t0:.0f}s)")

    melhor = gs.best_estimator_

    n_splits_efetivo = 5 
    if nome == "SVM" else K_FOLDS
    cv_outer = StratifiedKFold(n_splits=n_splits_efetivo,
                               shuffle=True, random_state=RANDOM_STATE)

    X_cv = X_hpo if nome == "KNN" else X_train
    y_cv = y_hpo if nome == "KNN" else y_train

    print(f"Validação cruzada {n_splits_efetivo}-Fold "
          f"({len(y_cv):,} amostras) …")
    t0 = time.time()
    cv_res = cross_validate(
        melhor, X_cv, y_cv,
        cv=cv_outer, scoring=SCORING,
        return_train_score=False, n_jobs=-1,
    )
    print(f"K-Fold concluído em {time.time()-t0:.0f}s")

    print(f"\n  Métricas {n_splits_efetivo}-Fold (treino):")
    for m in SCORING:
        v = cv_res[f"test_{m}"]
        print(f"    {m:<12}: {v.mean():.4f} ± {v.std():.4f}")

    X_fit = X_hpo if nome == "KNN" else X_train
    y_fit = y_hpo if nome == "KNN" else y_train
    melhor.fit(X_fit, y_fit)

    y_pred = melhor.predict(X_test)
    y_prob = melhor.predict_proba(X_test)[:, 1]

    metricas_teste = {
        "test_accuracy":  accuracy_score(y_test, y_pred),
        "test_precision": precision_score(y_test, y_pred, zero_division=0),
        "test_recall":    recall_score(y_test, y_pred, zero_division=0),
        "test_f1":        f1_score(y_test, y_pred, zero_division=0),
        "test_roc_auc":   roc_auc_score(y_test, y_prob),
    }

    print(f"\n  Métricas no conjunto de teste:")
    for k, v in metricas_teste.items():
        print(f"    {k:<20}: {v:.4f}")

    resultado = {
        "nome":            nome,
        "pipeline":        melhor,
        "melhores_params": gs.best_params_,
        "n_splits":        n_splits_efetivo,
        "cv_accuracy":     cv_res["test_accuracy"].mean(),
        "cv_precision":    cv_res["test_precision"].mean(),
        "cv_recall":       cv_res["test_recall"].mean(),
        "cv_f1":           cv_res["test_f1"].mean(),
        "cv_roc_auc":      cv_res["test_roc_auc"].mean(),
        "cv_f1_std":       cv_res["test_f1"].std(),
        "cv_auc_std":      cv_res["test_roc_auc"].std(),
        "cv_raw":          cv_res,
        **metricas_teste,
        "y_pred": y_pred,
        "y_prob": y_prob,
    }

    salvar_relatorio_individual(resultado, y_test)

    return resultado


# 7. RELATÓRIO INDIVIDUAL DE CADA MODELO
def salvar_relatorio_individual(res, y_test):
    nome_slug = res["nome"].lower().replace(" ", "_").replace("ã", "a").replace("ó", "o")
    print(f"\n  [Salvando relatório individual: {res['nome']}]")

    caminho_txt = os.path.join(OUTPUT_DIR, f"relatorio_{nome_slug}.txt")
    with open(caminho_txt, "w", encoding="utf-8") as f:
        f.write(f"{'='*60}\n")
        f.write(f"  RELATÓRIO — {res['nome']}\n")
        f.write(f"{'='*60}\n\n")

        f.write(f"Melhores hiperparâmetros: {res['melhores_params']}\n\n")

        f.write(f"Métricas {res['n_splits']}-Fold Cross-Validation\n")
        for m in ["cv_accuracy","cv_precision","cv_recall","cv_f1","cv_roc_auc"]:
            label = m.replace("cv_","").upper()
            f.write(f"  {label:<12}: {res[m]:.4f}  "
                    f"(±{res.get(m+'_std', 0):.4f})\n")

        f.write(f"\nMétricas Holdout\n")
        for m in ["test_accuracy","test_precision","test_recall","test_f1","test_roc_auc"]:
            label = m.replace("test_","").upper()
            f.write(f"  {label:<12}: {res[m]:.4f}\n")

        f.write(f"\nClassification Report\n")
        f.write(classification_report(
            y_test, res["y_pred"],
            target_names=["Sem DCV","Com DCV"]
        ))

    print(f"Salvo: {caminho_txt}")

    caminho_csv = os.path.join(OUTPUT_DIR, f"metricas_{nome_slug}.csv")
    pd.DataFrame([{
        "Modelo":        res["nome"],
        "Acc_CV":        round(res["cv_accuracy"], 4),
        "Prec_CV":       round(res["cv_precision"], 4),
        "Rec_CV":        round(res["cv_recall"], 4),
        "F1_CV":         round(res["cv_f1"], 4),
        "AUC_CV":        round(res["cv_roc_auc"], 4),
        "F1_CV_std":     round(res["cv_f1_std"], 4),
        "AUC_CV_std":    round(res["cv_auc_std"], 4),
        "Acc_Teste":     round(res["test_accuracy"], 4),
        "Prec_Teste":    round(res["test_precision"], 4),
        "Rec_Teste":     round(res["test_recall"], 4),
        "F1_Teste":      round(res["test_f1"], 4),
        "AUC_Teste":     round(res["test_roc_auc"], 4),
        "Params":        str(res["melhores_params"]),
    }]).to_csv(caminho_csv, index=False, sep=";", decimal=",")
    print(f"Salvo: {caminho_csv}")

    # Curva ROC individual
    fig, ax = plt.subplots(figsize=(7, 6))
    fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
    auc = res["test_roc_auc"]
    ax.plot(fpr, tpr, lw=2.5, color=CORES.get(res["nome"], "#333"),
            label=f'AUC = {auc:.3f}')
    ax.plot([0,1],[0,1], "k--", lw=1, label="Aleatório")
    ax.set_xlabel("Taxa de Falsos Positivos")
    ax.set_ylabel("Sensibilidade (Recall)")
    ax.set_title(f"Curva ROC — {res['nome']}", fontsize=13)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    salvar(fig, f"roc_{nome_slug}.png")

    # Matriz de confusão individual
    fig, ax = plt.subplots(figsize=(5, 4))
    cm = confusion_matrix(y_test, res["y_pred"])
    ConfusionMatrixDisplay(cm, display_labels=["Sem DCV","Com DCV"]).plot(
        ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Matriz de Confusão — {res['nome']}", fontsize=11)
    fig.tight_layout()
    salvar(fig, f"cm_{nome_slug}.png")

    print(f" Relatório individual de [{res['nome']}] salvo com sucesso.")


# 8. Visualizações finais em conjunto com outros modelos
def plot_matrizes_confusao(resultados, y_test):
    n = len(resultados)
    cols = min(n, 2)
    rows = (n + 1) // 2
    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows))
    axes = axes.flat if n > 1 else [axes]
    fig.suptitle("Matrizes de Confusão — Conjunto de Teste", fontsize=13)
    for ax, res in zip(axes, resultados):
        cm = confusion_matrix(y_test, res["y_pred"])
        ConfusionMatrixDisplay(cm, display_labels=["Sem DCV","Com DCV"]).plot(
            ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(res["nome"], fontsize=11)
    fig.tight_layout()
    salvar(fig, "matrizes_confusao.png")


def plot_roc_curves(resultados, y_test):
    fig, ax = plt.subplots(figsize=(8, 7))
    for res in resultados:
        fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
        auc = res["test_roc_auc"]
        ax.plot(fpr, tpr, lw=2.5, color=CORES.get(res["nome"], "#333"),
                label=f'{res["nome"]} (AUC = {auc:.3f})')
    ax.plot([0,1],[0,1], "k--", lw=1, label="Aleatório (AUC = 0.500)")
    ax.set_xlabel("Taxa de Falsos Positivos (1 – Especificidade)", fontsize=11)
    ax.set_ylabel("Sensibilidade (Recall / TPR)", fontsize=11)
    ax.set_title("Curvas ROC — Comparação dos Modelos", fontsize=13)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    salvar(fig, "curvas_roc.png")


def plot_comparacao_metricas(resultados):
    metricas = ["test_accuracy","test_precision","test_recall","test_f1","test_roc_auc"]
    labels   = ["Acurácia","Precisão","Recall","F1-Score","AUC-ROC"]
    nomes    = [r["nome"] for r in resultados]
    data     = {l: [r[m] for r in resultados] for l, m in zip(labels, metricas)}

    fig, ax = plt.subplots(figsize=(13, 6))
    x = np.arange(len(nomes))
    w = 0.15
    for i, (lbl, vals) in enumerate(data.items()):
        bars = ax.bar(x + i*w, vals, w, label=lbl)
        for b in bars:
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.005,
                    f"{b.get_height():.3f}", ha="center", fontsize=7)
    ax.set_xticks(x + w*2)
    ax.set_xticklabels(nomes, rotation=15, ha="right")
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Valor")
    ax.set_title("Comparação de Métricas — Conjunto de Teste", fontsize=13)
    ax.legend(loc="upper right", ncol=3)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    salvar(fig, "comparacao_metricas.png")


def plot_importancia_permutacao(resultados, X_test, y_test, feature_names):
    n = len(resultados)
    cols = min(n, 2)
    rows = (n + 1) // 2
    fig, axes = plt.subplots(rows, cols, figsize=(8*cols, 7*rows))
    axes = axes.flat if n > 1 else [axes]
    fig.suptitle("Importância de Features por Permutação (AUC-ROC)", fontsize=13)
    for ax, res in zip(axes, resultados):
        perm = permutation_importance(
            res["pipeline"], X_test, y_test,
            n_repeats=30, scoring="roc_auc",
            random_state=RANDOM_STATE, n_jobs=-1,
        )
        idx = np.argsort(perm.importances_mean)[::-1][:15]
        ax.barh(
            [feature_names[i] for i in idx[::-1]],
            perm.importances_mean[idx[::-1]],
            xerr=perm.importances_std[idx[::-1]],
            color=CORES.get(res["nome"], "#aaa"), alpha=0.85,
        )
        ax.set_title(res["nome"])
        ax.set_xlabel("Redução na AUC-ROC")
        ax.axvline(0, color="black", lw=0.8)
        ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    salvar(fig, "importancia_permutacao.png")


def plot_lr_coeficientes(res, feature_names):
    clf  = res["pipeline"].named_steps["clf"]
    coef = clf.coef_[0]
    df_c = pd.DataFrame({
        "feature":     feature_names,
        "coeficiente": coef,
        "odds_ratio":  np.exp(np.clip(coef, -10, 10)),
    }).sort_values("coeficiente")
    df_c.to_csv(os.path.join(OUTPUT_DIR, "lr_coeficientes.csv"), index=False)
    top = pd.concat([df_c.tail(10), df_c.head(10)])
    fig, ax = plt.subplots(figsize=(9, 7))
    cores_bar = ["#E63946" if v > 0 else "#457B9D" for v in top["coeficiente"]]
    ax.barh(top["feature"], top["coeficiente"], color=cores_bar, alpha=0.85)
    ax.axvline(0, color="black", lw=1)
    ax.set_xlabel("Coeficiente β (log-odds)")
    ax.set_title("Regressao Logistica — Coeficientes\n"
                 "(vermelho = aumenta risco DCV, azul = reduz)", fontsize=12)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    salvar(fig, "lr_coeficientes.png")


def plot_arvore_decisao(res, feature_names):
    clf = res["pipeline"].named_steps["clf"]
    fig, ax = plt.subplots(figsize=(22, 10))
    plot_tree(clf, feature_names=feature_names,
              class_names=["Sem DCV","Com DCV"],
              filled=True, rounded=True, max_depth=4,
              fontsize=8, ax=ax)
    ax.set_title("Arvore de Decisao — 4 primeiras camadas", fontsize=13)
    salvar(fig, "arvore_decisao.png")

    regras = export_text(clf, feature_names=feature_names, max_depth=6)
    caminho_txt = os.path.join(OUTPUT_DIR, "arvore_regras_decisao.txt")
    with open(caminho_txt, "w") as f:
        f.write(regras)
    print(f"  Regras da árvore salvas em: {caminho_txt}")
    imp = pd.DataFrame({
        "feature": feature_names,
        "gini_importance": clf.feature_importances_,
    }).sort_values("gini_importance", ascending=False).head(15)
    imp.to_csv(os.path.join(OUTPUT_DIR, "dt_gini_importance.csv"), index=False)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(imp["feature"][::-1], imp["gini_importance"][::-1],
            color="#2A9D8F", alpha=0.85)
    ax.set_xlabel("Importância Gini")
    ax.set_title("Arvore de Decisao — Feature Importance (Índice Gini)", fontsize=12)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    salvar(fig, "dt_gini_importance.png")


def plot_knn_curva_k(X_train, y_train, X_test, y_test):
    if len(y_train) > KNN_MAX_AMOSTRAS:
        idx = amostrar_estratificado(X_train, y_train, KNN_MAX_AMOSTRAS)
        X_tr, y_tr = X_train[idx], y_train[idx]
    else:
        X_tr, y_tr = X_train, y_train

    imp = SimpleImputer(strategy="median")
    sc  = StandardScaler()
    Xtr = sc.fit_transform(imp.fit_transform(X_tr))
    Xte = sc.transform(imp.transform(X_test))

    k_range = range(1, 32)
    f1s, aucs = [], []
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k, metric="euclidean")
        knn.fit(Xtr, y_tr)
        yp = knn.predict(Xte)
        yb = knn.predict_proba(Xte)[:, 1]
        f1s.append(f1_score(y_test, yp, zero_division=0))
        aucs.append(roc_auc_score(y_test, yb))

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()
    ax1.plot(list(k_range), f1s,  "o-",  color="#E63946", lw=2, label="F1-Score")
    ax2.plot(list(k_range), aucs, "s--", color="#457B9D", lw=2, label="AUC-ROC")
    ax1.set_xlabel("Valor de K")
    ax1.set_ylabel("F1-Score",  color="#E63946")
    ax2.set_ylabel("AUC-ROC",  color="#457B9D")
    ax1.set_title("KNN — Influência do Hiperparâmetro K", fontsize=13)
    lines1, l1 = ax1.get_legend_handles_labels()
    lines2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1+lines2, l1+l2, loc="lower right")
    ax1.grid(alpha=0.3)
    fig.tight_layout()
    salvar(fig, "knn_escolha_k.png")


# 9. RELATÓRIO GERAL FINAL
def gerar_relatorio_geral(resultados, y_test):
    print("\n" + "="*65)
    print("  RELATÓRIO GERAL FINAL")
    print("="*65)

    linhas = []
    for r in resultados:
        linhas.append({
            "Modelo":        r["nome"],
            "Acc_CV":        round(r["cv_accuracy"], 4),
            "Prec_CV":       round(r["cv_precision"], 4),
            "Rec_CV":        round(r["cv_recall"], 4),
            "F1_CV":         round(r["cv_f1"], 4),
            "AUC_CV":        round(r["cv_roc_auc"], 4),
            "F1_CV_std":     round(r["cv_f1_std"], 4),
            "AUC_CV_std":    round(r["cv_auc_std"], 4),
            "Acc_Teste":     round(r["test_accuracy"], 4),
            "Prec_Teste":    round(r["test_precision"], 4),
            "Rec_Teste":     round(r["test_recall"], 4),
            "F1_Teste":      round(r["test_f1"], 4),
            "AUC_Teste":     round(r["test_roc_auc"], 4),
            "Params":        str(r["melhores_params"]),
        })

    df = pd.DataFrame(linhas)

    # Salvar CSV
    caminho_csv = os.path.join(OUTPUT_DIR, "relatorio_comparativo_geral.csv")
    df.to_csv(caminho_csv, index=False, sep=";", decimal=",")

    # Melhor modelo por AUC
    melhor = df.loc[df["AUC_Teste"].idxmax(), "Modelo"]
    print(f"\n  ★ Melhor modelo (AUC-ROC): {melhor}")
    print(f"  Relatório salvo em: {caminho_csv}")

    # Salvar classification reports de todos
    caminho_txt = os.path.join(OUTPUT_DIR, "classification_reports_todos.txt")
    with open(caminho_txt, "w", encoding="utf-8") as f:
        for r in resultados:
            f.write(f"\n{'='*55}\n{r['nome']}\n{'='*55}\n")
            f.write(classification_report(
                y_test, r["y_pred"],
                target_names=["Sem DCV","Com DCV"]
            ))
    print(f"  Classification reports salvos em: {caminho_txt}")

    return df

def main():
    parser = argparse.ArgumentParser(description="Pipeline ML DCV — MIMIC-IV v3.1 v2")
    parser.add_argument("--mimic", type=str, required=True,
                        help="Caminho para a pasta raiz do MIMIC-IV v3.1")
    parser.add_argument("--chunk", type=int, default=CHUNK_SIZE,
                        help=f"Tamanho do chunk (default: {CHUNK_SIZE})")
    parser.add_argument("--dataset-cache", type=str, default=None,
                        help="Caminho para dataset_consolidado.csv.gz já gerado ")
    args = parser.parse_args()
    print("=" * 65)

    # Carregamento
    if args.dataset_cache and os.path.exists(args.dataset_cache):
        print(f"\n[!] Usando dataset cache: {args.dataset_cache}")
        df = pd.read_csv(args.dataset_cache)
    else:
        if not os.path.isdir(args.mimic):
            print(f"ERRO: pasta não encontrada: {args.mimic}")
            sys.exit(1)
        df = montar_dataset(args.mimic, chunk_size=args.chunk)

    # Features e target
    X, y, feature_names = selecionar_features(df)

    # Visualizações iniciais
    print("\nGerando visualizações preliminares")
    plot_distribuicao_classes(y)
    plot_missing_heatmap(X)

    X_arr = X.values
    X_train, X_test, y_train, y_test = train_test_split(
        X_arr, y,
        test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE,
    )
    print(f"\n  Treino : {len(y_train):,} | Teste: {len(y_test):,}")
    print(f"  Prev. DCV — Treino: {y_train.mean()*100:.1f}% | "
          f"Teste: {y_test.mean()*100:.1f}%")

    modelos   = definir_modelos()
    resultados = []

    print("\n" + "="*65)
    print("  ORDEM DE EXECUÇÃO DOS MODELOS")
    for i, nome in enumerate(modelos, 1):
        print(f"  {i}. {nome}")
    print("="*65)

    for nome, (pipeline, grade) in modelos.items():
        res = treinar_avaliar(nome, pipeline, grade,
                              X_train, y_train, X_test, y_test)
        resultados.append(res)
        print(f"\n [{nome}] concluído e salvo. "
              f"({len(resultados)}/{len(modelos)} modelos)")

    print("\n[Gerando visualizações comparativas finais]")
    plot_matrizes_confusao(resultados, y_test)
    plot_roc_curves(resultados, y_test)
    plot_comparacao_metricas(resultados)
    plot_importancia_permutacao(resultados, X_test, y_test, feature_names)
    plot_knn_curva_k(X_train, y_train, X_test, y_test)

    for res in resultados:
        if res["nome"] == "Regressao Logistica":
            plot_lr_coeficientes(res, feature_names)
        if res["nome"] == "Arvore de Decisao":
            plot_arvore_decisao(res, feature_names)

    # Relatório geral final
    print("\nGerando relatório geral final")
    gerar_relatorio_geral(resultados, y_test)

    print(f"\nPipeline concluído. Todos os artefatos em: ./{OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
