import sys
import re
import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import RidgeCV, LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from scipy.stats import mannwhitneyu, wilcoxon, pearsonr, ks_2samp
from statsmodels.stats.multitest import multipletests
from sklearn.metrics import mean_absolute_error, mean_squared_error
import statsmodels.formula.api as smf

warnings.filterwarnings('ignore')

# ATC3 -> readable drug class name mapping
ATC3_NAMES = {
    'A02B': 'Antacids/PPI', 'A10B': 'Antidiabetics/GLP-1', 'A11C': 'Vitamin A/D',
    'A11D': 'Vitamin B1', 'A11G': 'Vitamin C', 'A12A': 'Calcium', 'A12C': 'Other Minerals',
    'B01A': 'Antithrombotics', 'B02B': 'Vitamin K/Hemostatics', 'B03A': 'Iron Preps',
    'B03B': 'Vitamin B12/Folic', 'C07A': 'Beta Blockers', 'C08C': 'Ca-Channel Blockers',
    'C09A': 'ACE Inhibitors', 'C09C': 'ARBs Plain', 'C09D': 'ARBs Combo',
    'C10A': 'Statins', 'C10B': 'Lipid Combo', 'G02C': 'Uterine Stimulants',
    'G03C': 'Estrogens', 'G03D': 'Progestogens', 'G03F': 'Progest+Estrogen (HRT)',
    'G04C': 'BPH Drugs', 'M04A': 'Antigout', 'M05B': 'Bone Drugs',
    'N03A': 'Antiepileptics', 'N05C': 'Hypnotics/Sedatives', 'N06A': 'Antidepressants',
    'N06B': 'ADHD/Stimulants', 'R03A': 'Adrenergics Inhalants', 'R06A': 'Antihistamines',
    'OTHE': 'Other', 'UNKN': 'Unknown',
}
def atc3_label(code):
    name = ATC3_NAMES.get(code, code)
    return f"{code} ({name})" if name != code else code

# --- CONFIGURATION ---
LEJEPA_PATH   = "/net/mraid20/export/genie/LabData/Analyses/gilsa/embeddings/embeddings_with_date.pkl"
TARGETS_PATH  = "/home/gilsa/PycharmProjects/DEXA/targets_for_downstream_full.csv"
GLP1_PATH     = "/net/mraid20/export/genie/LabData/Analyses/nastya/GLP1/all_glp1_meds_logged.csv"

OUTPUT_DIR = "age_prediction_analysis"
if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
PARTITIONS_OUT = os.path.join(OUTPUT_DIR, "age_prediction_rate_partitions.csv")

FOLDS = 10
AGE_PARTITION_QUANTILES = 4
MIN_TIME_GAP_YEARS = 1.0
RATE_COL = 'rate_lejepa'
RATE_COL_RTM = 'rate_lejepa_rtm_corrected'
MODEL_COL_RTM = 'bin_rate_lejepa_rtm'
TOP_N_PLOTS = 8
MIN_POPULAR_DRUG_USERS = 20
MIN_PAIRED_SUBJECTS = 10
CIPRALEX_PATTERN = r'cipralex|escital'
TOP_HIST_ATC = 5

def is_atc3_code(code):
    s = str(code).strip().upper()
    return len(s) == 4 and s[0].isalpha() and s[1:].isalnum()

# =============================================================================
# CORE PIPELINE: PREDICTIONS & RATES
# =============================================================================

def clean_format(df):
    if isinstance(df, pd.Series): df = df.to_frame()
    df = df.reset_index()
    if 'date' in df.columns and 'Date' not in df.columns: df = df.rename(columns={'date': 'Date'})
    if 'RegistrationCode' in df.columns:
        df['RegistrationCode'] = df['RegistrationCode'].astype(str).apply(lambda x: f"10K_{x}" if not x.startswith("10K_") else x)
    if 'research_stage' in df.columns:
        df['research_stage'] = df['research_stage'].astype(str).apply(lambda x: 'baseline' if x == '00_00_visit' else x)
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'].astype(str).str.replace('_', '-'), errors='coerce')
    return df

def generate_predictions_pipeline():
    print("\n=== STEP 1: ML Prediction (RidgeCV) ===")
    df_lejepa = clean_format(pd.read_pickle(LEJEPA_PATH))
    meta_cols = [c for c in ['RegistrationCode', 'research_stage', 'Date'] if c in df_lejepa.columns]
    emb_cols = [c for c in df_lejepa.columns if c not in meta_cols]
    df_lejepa = df_lejepa.rename(columns={c: f"lej_{i}" for i, c in enumerate(emb_cols)})
    
    df_targets = clean_format(pd.read_csv(TARGETS_PATH))
    df_targets['age'] = df_targets.get('age', df_targets.get('Age'))
        
    df_train = pd.merge(df_lejepa, df_targets[['RegistrationCode', 'research_stage', 'age']], on=['RegistrationCode', 'research_stage'], how='inner')
    df_train = df_train.dropna(subset=['age', 'Date']).drop_duplicates(subset=['RegistrationCode', 'research_stage'])

    X = df_train[[f"lej_{i}" for i in range(len(emb_cols))]].values.astype(np.float32)
    y = df_train['age'].values
    preds = np.zeros_like(y, dtype=np.float64)

    for train_ix, test_ix in GroupKFold(n_splits=FOLDS).split(X, y, df_train['RegistrationCode'].values):
        pipe = Pipeline([('imputer', SimpleImputer(strategy='mean')), ('scaler', StandardScaler()), ('ridge', RidgeCV(alphas=np.logspace(-3, 4, 100)))])
        y_mean, y_std = y[train_ix].mean(), y[train_ix].std()
        pipe.fit(X[train_ix], (y[train_ix] - y_mean) / y_std)
        preds[test_ix] = pipe.predict(X[test_ix]) * y_std + y_mean

    df_train['age_pred_lejepa'] = preds
    
    r_val, _ = pearsonr(y, preds)
    mae = mean_absolute_error(y, preds)
    print(f'  > Age prediction: r={r_val:.3f}, MAE={mae:.2f}')
    
    return df_train.rename(columns={'age': 'age_true'})[meta_cols + ['age_true', 'age_pred_lejepa']]

def calculate_rates_and_partition(df_preds):
    print("\n=== STEP 2: Calculating Age Residuals & RTM-Corrected Rates ===")
    X_age = PolynomialFeatures(degree=2).fit_transform(df_preds[['age_true']])
    df_preds['expected_pred_age'] = LinearRegression().fit(X_age, df_preds['age_pred_lejepa']).predict(X_age)
    df_preds['age_residual'] = df_preds['age_pred_lejepa'] - df_preds['expected_pred_age']
    
    df_preds = df_preds.sort_values(by=['RegistrationCode', 'age_true'])
    rate_data = []
    
    for reg, group in df_preds.groupby('RegistrationCode'):
        if len(group) >= 2:
            first, last = group.iloc[0], group.iloc[-1]
            t_diff = last['age_true'] - first['age_true']
            if t_diff >= MIN_TIME_GAP_YEARS:
                rate_data.append({
                    'RegistrationCode': reg, 
                    RATE_COL: (last['age_residual'] - first['age_residual']) / t_diff,
                    'rate_start_date': first['Date'],
                    'rate_end_date': last['Date']
                })
                
    df_rates = pd.DataFrame(rate_data)
    
    # RTM correction
    baseline_residuals = df_preds.sort_values('age_true').drop_duplicates('RegistrationCode', keep='first')[['RegistrationCode', 'age_residual']].rename(columns={'age_residual': 'baseline_residual'})
    if not df_rates.empty:
        df_rates = df_rates.merge(baseline_residuals, on='RegistrationCode', how='left')
        
        mask = df_rates[['baseline_residual', RATE_COL]].notna().all(axis=1)
        X_bl = df_rates.loc[mask, 'baseline_residual'].values.reshape(-1, 1)
        y_rate = df_rates.loc[mask, RATE_COL].values
        reg = LinearRegression().fit(X_bl, y_rate)
        df_rates.loc[mask, RATE_COL_RTM] = y_rate - reg.predict(X_bl)
        
        valid = df_rates[RATE_COL_RTM].notna()
        if valid.sum() > 0:
            df_rates.loc[valid, MODEL_COL_RTM] = pd.qcut(
                df_rates.loc[valid, RATE_COL_RTM].rank(method='first'),
                q=AGE_PARTITION_QUANTILES, labels=['Q1', 'Q2', 'Q3', 'Q4'])
    
    return df_preds.merge(df_rates, on='RegistrationCode', how='left')

# =============================================================================
# EXTREME PHENOTYPES (Q1 vs Q4)
# =============================================================================

def plot_phenotype_extremes(df_data, group_col, df_targets, analysis_name, sex_filter=None):
    print(f"\n=== STEP 3: Phenotype Analysis ({analysis_name} Extremes) ===")
    
    df_gender = df_targets[['RegistrationCode', 'gender']].drop_duplicates()
    df_gender['Gender'] = df_gender['gender'].map({0: 'Female', 1: 'Male'})

    if sex_filter in ['Female', 'Male']:
        sex_code = 0 if sex_filter == 'Female' else 1
        allowed_regs = set(df_gender[df_gender['gender'] == sex_code]['RegistrationCode'])
        df_data = df_data[df_data['RegistrationCode'].isin(allowed_regs)].copy()
        df_targets = df_targets[df_targets['RegistrationCode'].isin(allowed_regs)].copy()
        df_gender = df_gender[df_gender['gender'] == sex_code].copy()
    
    num_cols = [c for c in df_targets.select_dtypes(include=np.number).columns if c not in ['age', 'Age', 'gender', 'index', 'level_0']]
    
    is_rate_analysis = analysis_name.startswith("Aging Rate")

    if is_rate_analysis:
        sort_col = 'Date' if 'Date' in df_targets.columns else 'age'
        df_targets = df_targets.sort_values(['RegistrationCode', sort_col])
        first = df_targets.groupby('RegistrationCode')[num_cols + ['age']].first()
        last = df_targets.groupby('RegistrationCode')[num_cols + ['age']].last()
        tdiff = last['age'] - first['age']
        valid = tdiff[tdiff >= MIN_TIME_GAP_YEARS].index
        pheno_vals = (last.loc[valid, num_cols] - first.loc[valid, num_cols]).div(tdiff.loc[valid], axis=0).reset_index()
        y_label = "Annualized Rate of Change"
    else:
        pheno_vals = df_targets[['RegistrationCode', 'research_stage'] + num_cols]
        y_label = "Raw Baseline Value"

    if is_rate_analysis:
        df_merged = df_data.merge(pheno_vals, on=['RegistrationCode']).merge(df_gender, on='RegistrationCode')
    else:
        df_merged = df_data.merge(pheno_vals, on=['RegistrationCode', 'research_stage']).merge(df_gender, on='RegistrationCode')
        
    df_extremes = df_merged[df_merged[group_col].isin(['Q1', 'Q4'])]

    results = []
    for pheno in num_cols:
        data = df_extremes[[group_col, pheno]].dropna()
        q1, q4 = data[data[group_col] == 'Q1'][pheno], data[data[group_col] == 'Q4'][pheno]
        if len(q1) >= 50 and len(q4) >= 50 and data[pheno].std() > 1e-6:
            _, p_val = mannwhitneyu(q1, q4)
            pooled_std = np.sqrt((q1.std()**2 + q4.std()**2) / 2)
            cohens_d = (q4.mean() - q1.mean()) / pooled_std if pooled_std > 1e-6 else 0
            row = {'Phenotype': pheno, 'Median_Diff': q4.median() - q1.median(), 'Mean_Diff': q4.mean() - q1.mean(), 'Cohens_d': cohens_d, 'N_Q1': len(q1), 'N_Q4': len(q4), 'P_Value': p_val}
            results.append(row)
            
    if not results: return
    df_res = pd.DataFrame(results).sort_values('P_Value')
    _, df_res['Adjusted_P_Value'], _, _ = multipletests(df_res['P_Value'], alpha=0.05, method='fdr_bh')
    
    res_out = os.path.join(OUTPUT_DIR, f"{analysis_name.replace(' ', '_').lower()}_phenotype_results.csv")
    df_res.to_csv(res_out, index=False)
    
    top_phenos = df_res.head(TOP_N_PLOTS)['Phenotype'].tolist()
    n_cols = 4
    n_rows = (len(top_phenos) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
    axes = axes.flatten()
    single_sex_plot = sex_filter in ['Female', 'Male']

    for i, pheno in enumerate(top_phenos):
        ax, subset = axes[i], df_extremes[[group_col, 'Gender', pheno]].dropna()
        q_val = df_res[df_res['Phenotype'] == pheno]['Adjusted_P_Value'].iloc[0]
        cohens_d_val = df_res[df_res['Phenotype'] == pheno]['Cohens_d'].iloc[0]
        ax.set_ylim(subset[pheno].quantile(0.02), subset[pheno].quantile(0.98))

        one_color = '#ff9da7' if sex_filter == 'Female' else ('#4e79a7' if sex_filter == 'Male' else '#7f8c8d')
        sns.boxplot(data=subset, x=group_col, y=pheno, order=['Q1', 'Q4'], color=one_color, ax=ax, showfliers=False, width=0.6)
        sns.stripplot(data=subset, x=group_col, y=pheno, order=['Q1', 'Q4'], color=one_color, alpha=0.12, jitter=True, size=2.5, ax=ax)
        
        ax.plot(range(2), subset.groupby(group_col)[pheno].median().reindex(['Q1', 'Q4']).values, 'k--o', linewidth=2.5, zorder=3)
        ax.set_title(f"{pheno.replace('_', ' ').title()}\nadj p={q_val:.1e} | d={cohens_d_val:.2f}",
                     color='darkred' if q_val < 0.05 else 'black', fontweight='bold' if q_val < 0.05 else 'normal', fontsize=9)
        ax.set_ylabel(y_label); ax.set_xlabel('')
        
    for j in range(i + 1, len(axes)): axes[j].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{analysis_name.replace(' ', '_').lower()}_phenotypes.png"), dpi=300)
    plt.close()


# =============================================================================
# MEDICATIONS: LMM & PAIRED
# =============================================================================

def load_medications():
    """Loads medications and injects GLP-1s, standardizes ATC3 codes."""
    try:
        from LabData.DataLoaders.Medications10KLoader import Medications10KLoader
        meds_loader = Medications10KLoader(gen_cache=True)
        meds_df = meds_loader.get_data().df.reset_index()
        meds_df['Date'] = pd.to_datetime(meds_df['Date'], errors='coerce', utc=True).dt.tz_localize(None)
        start_events = meds_df[(meds_df['Start'] == True) & meds_df['Date'].notna()].copy()

        meds_meta = getattr(meds_loader, '_columns_metadata', pd.DataFrame())
        if isinstance(meds_meta, pd.DataFrame) and not meds_meta.empty and 'column_name' in meds_meta.columns:
            atc_source_col = 'ATC' if 'ATC' in meds_meta.columns else ('ATC4' if 'ATC4' in meds_meta.columns else None)
            if atc_source_col is not None:
                atc_map = meds_meta[['column_name', atc_source_col]].copy()
                atc_map.columns = ['medication', 'atc_raw']
                atc_map['medication'] = atc_map['medication'].astype(str).str.strip()
                atc_map['atc_raw'] = atc_map['atc_raw'].astype(str).str.strip().str.upper()
                atc_map = atc_map.dropna(subset=['medication', 'atc_raw']).drop_duplicates('medication')
                atc_map = atc_map[~atc_map['atc_raw'].isin(['', 'EMPTY', 'NAN'])]
                start_events['medication'] = start_events['medication'].astype(str).str.strip()
                start_events = start_events.merge(atc_map, on='medication', how='left')
                start_events['atc3'] = start_events['atc_raw'].str[:4]
                start_events.drop(columns=['atc_raw'], inplace=True)
    except Exception as e:
        print(f"  > Could not load Medications10KLoader: {e}")
        start_events = pd.DataFrame(columns=['RegistrationCode', 'Date', 'medication', 'Start'])
    
    try:
        glp1_df = clean_format(pd.read_csv(GLP1_PATH)).dropna(subset=['Date'])
        glp1_df['Date'] = pd.to_datetime(glp1_df['Date'], errors='coerce', utc=True).dt.tz_localize(None)
        glp1_starts = glp1_df.groupby('RegistrationCode')['Date'].min().reset_index()
        glp1_starts['medication'] = 'GLP-1 Agonists'
        glp1_starts['atc3'] = 'A10B'
        glp1_starts['Start'] = True
        start_events = pd.concat([start_events, glp1_starts], ignore_index=True)
    except Exception as e:
        pass

    if 'Date' in start_events.columns:
        start_events['Date'] = pd.to_datetime(start_events['Date'], errors='coerce', utc=True).dt.tz_localize(None)
        start_events = start_events.dropna(subset=['Date'])

    if start_events.empty: return None

    atc_col = next((c for c in ['atc', 'ATC', 'atc_code', 'ATC_code'] if c in start_events.columns), None)
    if atc_col is not None and 'atc3' not in start_events.columns:
        start_events['atc3'] = start_events[atc_col].astype(str).str.strip().str.upper().str[:4]
    start_events['atc3'] = start_events['atc3'].astype(str).str.strip()
    return start_events[start_events['atc3'].str.len() == 4].copy()

def run_lmm_medication_analysis(df_final, start_events):
    print("\n=== STEP 4A: Linear Mixed-Effects Model (LMM) for Medications ===")
    
    lmm_results = []
    visits = df_final[['RegistrationCode', 'Date', 'age_true', 'age_residual']].dropna().copy()
    visits['Date'] = pd.to_datetime(visits['Date'])
    
    popular_drugs = start_events.groupby('atc3')['RegistrationCode'].nunique()
    popular_drugs = popular_drugs[popular_drugs >= MIN_POPULAR_DRUG_USERS].index.tolist()
    
    # Optional CIPRALEX override
    start_events['medication_norm'] = start_events['medication'].astype(str)
    cip_events = start_events[start_events['medication_norm'].str.contains(CIPRALEX_PATTERN, case=False, na=False)]
    
    analyzable_drugs = []
    for code in popular_drugs:
        first_start = start_events[start_events['atc3'] == code].groupby('RegistrationCode')['Date'].min()
        analyzable_drugs.append({'drug': code, 'label': atc3_label(code), 'first_start': first_start})
    if not cip_events.empty:
        analyzable_drugs.append({'drug': 'CIPRALEX', 'label': 'CIPRALEX (Escitalopram)', 'first_start': cip_events.groupby('RegistrationCode')['Date'].min()})

    mean_age = visits['age_true'].mean()
    
    for group in analyzable_drugs:
        drug = group['drug']
        first_start = group['first_start'].rename('drug_start_date')
        
        # Merge prescription start dates onto the longitudinal visit data
        df_model = visits.merge(first_start, on='RegistrationCode', how='left')
        
        # Time-Varying Covariates
        df_model['is_active'] = np.where(df_model['Date'] >= df_model['drug_start_date'], 1, 0)
        df_model['years_on_drug'] = np.where(
            df_model['is_active'] == 1,
            (df_model['Date'] - df_model['drug_start_date']).dt.days / 365.25,
            0
        )
        df_model['age_centered'] = df_model['age_true'] - mean_age
        
        n_exposed_scans = df_model['is_active'].sum()
        if n_exposed_scans < 20: 
            continue
            
        try:
            # Model 1: Main binary effect
            md = smf.mixedlm("age_residual ~ age_centered + is_active", df_model, groups=df_model["RegistrationCode"]).fit()
            
            # Model 2: Dose Response (optional context)
            md_dose = smf.mixedlm("age_residual ~ age_centered + years_on_drug", df_model, groups=df_model["RegistrationCode"]).fit()
            
            lmm_results.append({
                'drug': drug,
                'label': group['label'],
                'N_total_scans': len(df_model),
                'N_exposed_scans': n_exposed_scans,
                'lmm_effect': md.params['is_active'],
                'lmm_p_value': md.pvalues['is_active'],
                'dose_effect_per_year': md_dose.params['years_on_drug'],
                'dose_p_value': md_dose.pvalues['years_on_drug']
            })
        except Exception as e:
            print(f"  > LMM failed for {drug}: {e}")
            
    lmm_df = pd.DataFrame(lmm_results)
    if not lmm_df.empty:
        lmm_df = lmm_df.sort_values('lmm_p_value')
        _, lmm_df['lmm_adjusted_p'], _, _ = multipletests(lmm_df['lmm_p_value'], alpha=0.05, method='fdr_bh')
        lmm_df.to_csv(os.path.join(OUTPUT_DIR, "medications_atc3_lmm_results.csv"), index=False)
        print(f"  > Saved LMM results for {len(lmm_df)} drugs.")
        
    return lmm_df, analyzable_drugs

def run_paired_medication_analysis(df_final, df_targets, analyzable_drugs):
    print("\n=== STEP 4B: Paired Crossover Analysis (Strict Before/After) ===")
    visits = df_final[['RegistrationCode', 'Date', 'age_residual']].dropna().copy()
    visits['Date'] = pd.to_datetime(visits['Date'])
    gender_df = df_targets[['RegistrationCode', 'gender']].drop_duplicates() if 'gender' in df_targets.columns else None
    
    paired_results = []
    
    for group in analyzable_drugs:
        drug = group['drug']
        first_start = group['first_start']
        drug_visits = visits[visits['RegistrationCode'].isin(first_start.index)].copy()
        
        # Dynamic Gender Shield
        demo = "All"
        if gender_df is not None:
            genders = drug_visits.merge(gender_df, on='RegistrationCode')['gender']
            female_ratio = (genders == 0).mean() if len(genders) > 0 else 0.5
            if female_ratio >= 0.90: demo, allowed = "Female", 0
            elif female_ratio <= 0.10: demo, allowed = "Male", 1
            else: allowed = None
            if allowed is not None: drug_visits = drug_visits[drug_visits['RegistrationCode'].isin(gender_df[gender_df['gender'] == allowed]['RegistrationCode'])]
        
        drug_visits['start_date'] = drug_visits['RegistrationCode'].map(first_start)
        drug_visits = drug_visits.dropna(subset=['start_date'])
        drug_visits['period'] = np.where(drug_visits['Date'] < drug_visits['start_date'], 'Before', 'After')
        
        subj_means = drug_visits.groupby(['RegistrationCode', 'period'])['age_residual'].mean().unstack()
        if 'Before' not in subj_means.columns or 'After' not in subj_means.columns: continue
            
        subj_means = subj_means.dropna(subset=['Before', 'After'])
        
        if len(subj_means) >= MIN_PAIRED_SUBJECTS:
            delta = subj_means['After'] - subj_means['Before']
            _, pval = wilcoxon(subj_means['Before'], subj_means['After'])
            paired_results.append({
                'drug': drug, 'demo': demo, 'N_paired': len(subj_means),
                'mean_before': subj_means['Before'].mean(), 'mean_after': subj_means['After'].mean(), 
                'mean_delta': delta.mean(), 'p_value': pval,
            })

    paired_df = pd.DataFrame(paired_results)
    if paired_df.empty: return pd.DataFrame(), []
    
    paired_df = paired_df.sort_values('p_value')
    _, paired_df['adjusted_p_value'], _, _ = multipletests(paired_df['p_value'], alpha=0.05, method='fdr_bh')
    paired_df.to_csv(os.path.join(OUTPUT_DIR, "medications_atc3_paired_before_after.csv"), index=False)
    
    sig_paired_drugs = paired_df[(paired_df['adjusted_p_value'] < 0.05) | (paired_df['p_value'] < 0.10)]['drug'].tolist()
    sig_paired_drugs = list(dict.fromkeys([d for d in sig_paired_drugs if is_atc3_code(d)]))[:TOP_HIST_ATC]
    
    # Plot top 6 paired crossovers
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for i, (_, row) in enumerate(paired_df.head(6).iterrows()):
        ax, drug = axes.flatten()[i], row['drug']
        first_start = next(g['first_start'] for g in analyzable_drugs if g['drug'] == drug)
        dvis = visits[visits['RegistrationCode'].isin(first_start.index)].copy()
        dvis['start_date'] = dvis['RegistrationCode'].map(first_start)
        dvis['period'] = np.where(dvis['Date'] < dvis['start_date'], 'Before', 'After')
        dvis = dvis[dvis.groupby('RegistrationCode')['period'].transform('nunique') == 2]
        
        sns.boxplot(data=dvis, x='period', y='age_residual', order=['Before', 'After'], palette={'Before':'#bdc3c7', 'After':'#3498db'}, showfliers=False, ax=ax, width=0.5)
        for _, p_row in dvis.groupby(['RegistrationCode', 'period'])['age_residual'].mean().unstack().iterrows(): 
            ax.plot([0, 1], [p_row['Before'], p_row['After']], color='black', alpha=0.15, linewidth=1)
        ax.axhline(0, color='red', linestyle='--', alpha=0.5)
        ax.set_title(f"[{row['demo']}] {atc3_label(drug)}\nN={row['N_paired']} | Δ {row['mean_delta']:+.2f} yrs | adj p={row['adjusted_p_value']:.2f}", fontsize=9, fontweight='bold' if row['adjusted_p_value'] < 0.05 else 'normal')
        ax.set_ylabel('Age Residual'); ax.set_xlabel('')
        
    for j in range(i+1, 6): axes.flatten()[j].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "medications_atc3_top_paired_crossovers.png"), dpi=300)
    plt.close()
    
    return paired_df, sig_paired_drugs


def run_significant_drug_phenotype_change_analysis(df_targets, start_events, significant_drugs, df_final):
    print("\n=== STEP 4C: Phenotype Changes for Significant Paired Drugs ===")
    if not significant_drugs: return

    targets = df_targets.copy()
    
    # --- Restored Failsafe: Map Dates from df_final if missing from targets ---
    if 'Date' not in targets.columns:
        if 'Date' in df_final.columns and 'research_stage' in targets.columns:
            date_map = df_final[['RegistrationCode', 'research_stage', 'Date']].dropna(subset=['Date']).drop_duplicates(['RegistrationCode', 'research_stage'])
            targets = targets.merge(date_map, on=['RegistrationCode', 'research_stage'], how='left')
        else:
            print("  > Skipping phenotype-change analysis: targets file has no Date column and no mappable date source.")
            return

    targets['Date'] = pd.to_datetime(targets['Date'], errors='coerce')
    targets = targets.dropna(subset=['RegistrationCode', 'Date'])

    num_cols = [c for c in targets.select_dtypes(include=np.number).columns if c not in ['age', 'Age', 'gender', 'index', 'level_0']]
    drug_codes = [d for d in significant_drugs if is_atc3_code(d)]

    results = []
    for drug in drug_codes:
        first_start = start_events[start_events['atc3'] == drug].groupby('RegistrationCode')['Date'].min()
        sub = targets[targets['RegistrationCode'].isin(first_start.index)].copy()
        sub['start_date'] = sub['RegistrationCode'].map(first_start)
        sub = sub.dropna(subset=['start_date'])
        sub['period'] = np.where(sub['Date'] < sub['start_date'], 'Before', 'After')

        per_subj = sub.groupby(['RegistrationCode', 'period'])[num_cols].mean().unstack('period')
        if ('Before' not in per_subj.columns.get_level_values(1)) or ('After' not in per_subj.columns.get_level_values(1)): continue

        subj_sex = sub.groupby('RegistrationCode')['gender'].first() if 'gender' in sub.columns else pd.Series(index=per_subj.index, dtype='float64')

        for sex_code, sex_label in [(0, 'Female'), (1, 'Male')]:
            sex_idx = subj_sex[subj_sex == sex_code].index
            sex_frame = per_subj.loc[per_subj.index.intersection(sex_idx)]
            sex_rows = []
            for ph in num_cols:
                b = sex_frame[(ph, 'Before')].dropna()
                a = sex_frame[(ph, 'After')].dropna()
                common_idx = b.index.intersection(a.index)
                if len(common_idx) < 5: continue
                b, a = b.loc[common_idx], a.loc[common_idx]
                d = a - b
                if np.nanstd(d.values) < 1e-8: p_val = np.nan
                else: _, p_val = wilcoxon(b.values, a.values)
                d_std = np.nanstd(d.values)
                cohens_d = float(np.mean(d.values)) / d_std if d_std > 1e-8 else 0.0

                sex_rows.append({
                    'drug': drug, 'drug_label': atc3_label(drug), 'sex': sex_label, 'phenotype': ph,
                    'N_paired': len(common_idx), 'cohens_d': cohens_d, 'p_value': p_val,
                })

            if sex_rows:
                sex_df = pd.DataFrame(sex_rows).sort_values('p_value', na_position='last')
                valid = sex_df['p_value'].notna()
                sex_df['adjusted_p_value'] = np.nan
                if valid.sum() > 0: _, sex_df.loc[valid, 'adjusted_p_value'], _, _ = multipletests(sex_df.loc[valid, 'p_value'], alpha=0.05, method='fdr_bh')
                results.extend(sex_df.to_dict('records'))

    if results:
        out_df = pd.DataFrame(results).sort_values(['drug', 'sex', 'p_value'], na_position='last')
        out_csv = os.path.join(OUTPUT_DIR, 'medications_significant_hits_phenotype_changes_by_sex.csv')
        out_df.to_csv(out_csv, index=False)
        plot_significant_drug_phenotype_forest(out_csv)

def plot_significant_drug_phenotype_forest(pheno_csv_path):
    df = pd.read_csv(pheno_csv_path)
    sig_df = df[df['adjusted_p_value'] < 0.10].copy()
    if sig_df.empty: return
    top_drugs = sig_df['drug'].unique().tolist()
    
    for drug in top_drugs:
        plot_df = df[df['drug'] == drug].copy()
        sig_phenos = plot_df[plot_df['adjusted_p_value'] < 0.10]['phenotype'].unique()
        plot_df = plot_df[plot_df['phenotype'].isin(sig_phenos)]
        if plot_df.empty: continue
            
        plot_df['abs_effect'] = plot_df['cohens_d'].abs()
        pheno_order = plot_df.groupby('phenotype')['abs_effect'].mean().sort_values(ascending=False).index
        fig, ax = plt.subplots(figsize=(10, max(5, len(pheno_order) * 0.6)))
        ax.set_yticks(np.arange(len(pheno_order)))
        ax.set_yticklabels([p.replace('_', ' ').title() for p in pheno_order])
        
        for sex, color, offset in [('Female', '#ff9da7', -0.15), ('Male', '#4e79a7', 0.15)]:
            sex_data = plot_df[plot_df['sex'] == sex].set_index('phenotype')
            for i, pheno in enumerate(pheno_order):
                if pheno in sex_data.index:
                    row = sex_data.loc[pheno]
                    strict_sig = row['adjusted_p_value'] < 0.05
                    ax.plot(row['cohens_d'], i + offset, marker='o', color=color, markersize=9 if strict_sig else 6, markerfacecolor=color if strict_sig else 'white', markeredgecolor=color, markeredgewidth=1.5, zorder=3)
                    align = 'left' if row['cohens_d'] > 0 else 'right'
                    ax.text(row['cohens_d'] + (0.05 if row['cohens_d'] > 0 else -0.05), i + offset, f"n={int(row['N_paired'])}", va='center', ha=align, fontsize=8, color='dimgray')

        ax.axvline(0, color='black', linestyle='--', alpha=0.5, zorder=1)
        ax.set_title(f"Phenotypic Changes After Starting {plot_df['drug_label'].iloc[0]}", fontweight='bold')
        ax.set_xlabel("Effect Size (Cohen's d: Standardized Δ from Baseline)")
        ax.invert_yaxis(); ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"phenotype_changes_{drug}.png"), dpi=300)
        plt.close()

# =============================================================================
# COMBINED FOREST PLOT & DISTRIBUTIONS
# =============================================================================

def plot_medication_forest(lmm_df, paired_df):
    """Forest plot comparing LMM Baseline Shift vs. Paired Crossover Shift."""
    dfs = []
    if lmm_df is not None and not lmm_df.empty:
        l = lmm_df.copy()
        l['analysis'] = 'LMM (Full Cohort)'
        l['effect'] = l['lmm_effect']
        l['atc3'] = l['drug']
        l['p'] = l['lmm_p_value']
        l['adj_p'] = l['lmm_adjusted_p']
        dfs.append(l[['atc3', 'analysis', 'effect', 'N_exposed_scans', 'p', 'adj_p']].rename(columns={'N_exposed_scans': 'N'}))
        
    if paired_df is not None and not paired_df.empty:
        p = paired_df.copy()
        p['analysis'] = 'Paired (Incident Users)'
        p['effect'] = p['mean_delta']
        p['atc3'] = p['drug']
        p['p'] = p['p_value']
        p['adj_p'] = p.get('adjusted_p_value', p['p_value'])
        dfs.append(p[['atc3', 'analysis', 'effect', 'N_paired', 'p', 'adj_p']].rename(columns={'N_paired': 'N'}))
        
    if not dfs: return
    
    df = pd.concat(dfs, ignore_index=True)
    df['label'] = df['atc3'].apply(atc3_label)
    
    # Take top drugs by LMM p-value
    top_drugs = df[df['analysis'] == 'LMM (Full Cohort)'].nsmallest(15, 'p')['atc3'].tolist()
    if not top_drugs: top_drugs = df['atc3'].unique()[:15]
    
    df = df[df['atc3'].isin(top_drugs)].copy()
    
    fig, ax = plt.subplots(figsize=(10, max(6, len(top_drugs) * 0.6)))
    colors = {'LMM (Full Cohort)': '#8e44ad', 'Paired (Incident Users)': '#3498db'}
    y_positions = {}
    y = 0
    for drug in top_drugs:
        rows = df[df['atc3'] == drug]
        for _, row in rows.iterrows():
            y_positions[(drug, row['analysis'])] = y
            color = colors[row['analysis']]
            sig = row['adj_p'] < 0.05
            marker = 's' if 'Paired' in row['analysis'] else 'o'
            ax.plot(row['effect'], y, marker=marker, color=color, markersize=10 if sig else 7, markeredgecolor='black' if sig else color, markeredgewidth=1.5 if sig else 0, zorder=3)
            ax.text(row['effect'] + 0.02, y, f"p={row['p']:.1e}{' *' if sig else ''}", fontsize=7, va='center')
            y += 1
        y += 0.5 
    
    ax.axvline(0, color='black', linestyle='--', alpha=0.5)
    yticks = [np.mean([y_positions.get((d, a)) for a in colors.keys() if (d, a) in y_positions]) for d in top_drugs]
    ax.set_yticks(yticks); ax.set_yticklabels([atc3_label(d) for d in top_drugs], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel('Effect on Biological Age Residual (years)', fontsize=11)
    ax.set_title('Medication Effects: LMM vs Paired Crossover\n(■ Paired, ● LMM; bold outline = adjusted p<0.05)', fontsize=12, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#8e44ad', markersize=10, label='LMM (Full Cohort)'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#3498db', markersize=10, label='Paired (Incident Users)')
    ], loc='lower right', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "medications_forest_plot.png"), dpi=300)
    plt.close()


def plot_rate_distribution_with_drugs(df_final, start_events, highlight_atc3):
    print("\n=== STEP 5: Plotting Overall Rate Distribution with Drug Overlays ===")
    if RATE_COL_RTM not in df_final.columns or df_final[RATE_COL_RTM].isna().all(): return

    df_rates = df_final[['RegistrationCode', RATE_COL_RTM]].dropna().drop_duplicates('RegistrationCode')
    pop_vals = df_rates[RATE_COL_RTM].values
    pop_mean = float(np.mean(pop_vals))
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#f1c40f']

    drug_data = []
    for idx, drug_identifier in enumerate([d for d in highlight_atc3 if is_atc3_code(d)][:TOP_HIST_ATC]):
        matching_events = start_events[start_events['atc3'] == drug_identifier]
        if matching_events.empty: continue
        drug_rates = df_rates[df_rates['RegistrationCode'].isin(matching_events['RegistrationCode'])]
        if len(drug_rates) >= 10:
            vals = drug_rates[RATE_COL_RTM].values
            ks_stat, ks_p = ks_2samp(vals, pop_vals)
            drug_data.append({'label': atc3_label(drug_identifier), 'vals': vals, 'mean': float(np.mean(vals)), 'n': len(vals), 'ks_stat': ks_stat, 'ks_p': ks_p, 'color': colors[idx % len(colors)]})

    fig, ax = plt.subplots(figsize=(12, 7))
    sns.histplot(pop_vals, bins=60, color='#95a5a6', edgecolor='white', stat='density', alpha=0.35, ax=ax)
    sns.kdeplot(pop_vals, color='black', linewidth=2.2, ax=ax, label=f'Population (n={len(pop_vals)})')
    ax.axvline(pop_mean, color='black', linestyle='-', linewidth=1.8, alpha=0.9)

    for d in drug_data:
        ax.axvline(d['mean'], color=d['color'], linestyle='--', linewidth=1.5, alpha=0.9)
        ax.plot([], [], color=d['color'], linestyle='--', linewidth=1.5, label=f"{d['label']} (n={d['n']}) | mean={d['mean']:+.3f} | KS={d['ks_stat']:.3f}, p={d['ks_p']:.1e}")

    ax.set_title('RTM-Corrected Aging Pace Distribution', fontsize=13, fontweight='bold')
    ax.set_xlabel('Aging Pace (Δ bio yrs / chron yr)'); ax.set_ylabel('Density')
    ax.legend(fontsize=8, framealpha=0.95); ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "aging_rate_distribution_with_drugs.png"), dpi=300)
    plt.close()


def plot_baseline_gap_distribution_with_drugs(df_gap, start_events, highlight_drugs):
    print("\n=== STEP 6: Plotting Baseline Age Gap Distribution with Drug Overlays ===")
    base = df_gap[['RegistrationCode', 'age_residual']].dropna().drop_duplicates('RegistrationCode')
    if base.empty or start_events is None or start_events.empty: return

    atc_only = [d for d in highlight_drugs if is_atc3_code(d)][:TOP_HIST_ATC]
    if not atc_only: return
    
    fig, axes = plt.subplots(len(atc_only) + 1, 1, figsize=(10, 3.5 * (len(atc_only) + 1)))
    pop_vals = base['age_residual'].values
    pop_mean = pop_vals.mean()
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#f1c40f']
    
    drug_rows = []
    for i, drug in enumerate(atc_only):
        regs = start_events[start_events['atc3'] == drug]['RegistrationCode'].dropna().unique()
        vals = base[base['RegistrationCode'].isin(regs)]['age_residual'].dropna().values
        if len(vals) >= 10: drug_rows.append({'label': atc3_label(drug), 'vals': vals, 'n': len(vals), 'color': colors[i % len(colors)]})

    ax0 = axes[0]
    sns.histplot(pop_vals, bins=60, color='#95a5a6', edgecolor='white', stat='density', alpha=0.6, ax=ax0)
    ax0.axvline(pop_mean, color='black', linestyle='-', linewidth=2, label=f'Pop mean: {pop_mean:+.2f}')
    for d in drug_rows: ax0.axvline(np.mean(d['vals']), color=d['color'], linestyle='--', linewidth=2, label=f"{d['label']} (n={d['n']})")
    ax0.set_title('Baseline Age Gap Distribution'); ax0.legend(fontsize=8); ax0.grid(True, alpha=0.2)

    for i, d in enumerate(drug_rows):
        ax = axes[i + 1]
        sns.histplot(d['vals'], bins=30, color=d['color'], edgecolor='white', stat='density', alpha=0.7, ax=ax)
        sns.kdeplot(pop_vals, color='#95a5a6', linewidth=2, alpha=0.6, ax=ax)
        ks_stat, ks_p = ks_2samp(d['vals'], pop_vals)
        ax.axvline(np.mean(d['vals']), color=d['color'], linestyle='--', linewidth=2)
        ax.set_title(f"{d['label']} | n={d['n']} | KS={ks_stat:.3f}, p={ks_p:.1e}")
        ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'baseline_gap_distribution_with_drugs.png'), dpi=300)
    plt.close()


def main():
    # 1. Base Predictions & Pace
    df_preds = generate_predictions_pipeline()
    df_final = calculate_rates_and_partition(df_preds)
    df_final.to_csv(PARTITIONS_OUT, index=False)
    
    df_targets = clean_format(pd.read_csv(TARGETS_PATH))
    
    # 2. Extreme Phenotypes (Baseline Gap & RTM-Corrected Pace)
    df_gap = df_final.sort_values('age_true').drop_duplicates('RegistrationCode', keep='first').copy()
    df_gap['Gap_Quartile'] = pd.qcut(df_gap['age_residual'], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
    plot_phenotype_extremes(df_gap, 'Gap_Quartile', df_targets, "Baseline Gap Female", sex_filter='Female')
    plot_phenotype_extremes(df_gap, 'Gap_Quartile', df_targets, "Baseline Gap Male", sex_filter='Male')
    
    df_rate_rtm = df_final.dropna(subset=[MODEL_COL_RTM]).drop_duplicates('RegistrationCode', keep='first')
    if len(df_rate_rtm) > 0:
        plot_phenotype_extremes(df_rate_rtm, MODEL_COL_RTM, df_targets, "Aging Rate Female", sex_filter='Female')
        plot_phenotype_extremes(df_rate_rtm, MODEL_COL_RTM, df_targets, "Aging Rate Male", sex_filter='Male')
    
    # 3. Medications (LMM & Paired Timing)
    start_events = load_medications()
    if start_events is not None:
        # A. Discovery Engine (LMM)
        lmm_df, analyzable_drugs = run_lmm_medication_analysis(df_final, start_events)
        
        # B. Causal Proof Engine (Paired Wilcoxon)
        paired_df, sig_drugs = run_paired_medication_analysis(df_final, df_targets, analyzable_drugs)
        
        # C. Phenotype Changes for significant paired hits
        run_significant_drug_phenotype_change_analysis(df_targets, start_events, sig_drugs, df_final)
        
        # 4. Plots (Forest & Distributions)
        plot_medication_forest(lmm_df, paired_df)
        
        highlight = [d for d in (sig_drugs or []) if is_atc3_code(d)][:TOP_HIST_ATC]
        if not highlight and not lmm_df.empty: highlight = lmm_df['drug'].head(TOP_HIST_ATC).tolist()
            
        plot_rate_distribution_with_drugs(df_final, start_events, highlight)
        plot_baseline_gap_distribution_with_drugs(df_gap, start_events, highlight)
        
    print("\n" + "="*60)
    print("PIPELINE COMPLETE — Output files:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        print(f"  {f:55s} ({os.path.getsize(os.path.join(OUTPUT_DIR, f)) / 1024:.1f} KB)")
    print("="*60)

if __name__ == "__main__":
    main()