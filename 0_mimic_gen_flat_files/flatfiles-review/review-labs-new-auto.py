import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -------- USER INPUT --------
emory_file = "/hpc/home/yy450/link_dctrl_yy450/Sepy2.0/0_mimic_gen_flat_files/flatfiles-review/combine-labs/emory-labs-joined.csv"
mimic_file = "/hpc/home/yy450/link_dctrl_yy450/Sepy2.0/0_mimic_gen_flat_files/flatfiles-review/combine-labs/mimic-labs-joined.csv"

emory_grouping_file = "/hpc/home/yy450/link_dctrl_yy450/Sepy2.0/0_mimic_gen_flat_files/flatfiles-review/combine-labs/emory-labs-grouping-joined.csv"
mimic_grouping_file = "/hpc/home/yy450/link_dctrl_yy450/Sepy2.0/mimic_groupings/mimic_grouping_labs_new.csv"

output_root = "/hpc/home/yy450/link_dctrl_yy450/Sepy2.0/0_mimic_gen_flat_files/flatfiles-review/labs"
# ----------------------------

columns_to_analyze = ['alanine_aminotransferase_(alt)', 'albumin', 'alkaline_phosphatase', 'ammonia', 'amylase', 'anion_gap', 'aspartate_aminotransferase_(ast)', 'b-type_natriuretic_peptide_(bnp)', 'base_excess', 'bicarb_(hco3)', 'bilirubin_direct', 'bilirubin_indirect', 'bilirubin_total', 'blood_urea_nitrogen_(bun)', 'c_diff', 'calcium', 'calcium_ionized', 'carboxy_hgb', 'chloride', 'cortisol', 'covid', 'creatinine', 'crp_high_sens', 'd_dimer', 'erythrocyte_sedimentation_rate_(esr)', 'fibrinogen', 'gfr', 'glucose', 'haptoglobin', 'hematocrit', 'hemoglobin', 'hemoglobin_a1c', 'inr', 'lactate_dehydrogenase', 'lactic_acid', 'lipase', 'lymphocyte', 'magnesium', 'met_hgb', 'neutrophils', 'osmolarity', 'parathyroid_level', 'partial_pressure_of_carbon_dioxide_(paco2)', 'partial_pressure_of_oxygen_(pao2)', 'partial_prothrombin_time_(ptt)', 'ph', 'phosphorus', 'platelets', 'potassium', 'progesterone', 'protein', 'prothrombin_time_(pt)', 'saturation_of_oxygen_(sao2)', 'sodium', 'thrombin_time', 'thyroid_stimulating_hormone_(tsh)', 'transferrin', 'troponin', 'white_blood_cell_count']

# -------- Load Data --------
df_emory = pd.read_csv(emory_file)
df_mimic = pd.read_csv(mimic_file)
df_emory_grouping = pd.read_csv(emory_grouping_file)
df_mimic_grouping = pd.read_csv(mimic_grouping_file)

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 2000)
pd.set_option('display.max_colwidth', None)

print(f"✅ Loaded Emory: {df_emory.shape}, MIMIC: {df_mimic.shape}")
print(f"📋 Emory Columns: {list(df_emory.columns)}")
print(f"📋 MIMIC Columns: {list(df_mimic.columns)}")

# =============================
# 🔁 主循环：逐个运行原分析逻辑
# =============================
for idx, colname in enumerate(columns_to_analyze, start=1):
    print(f"\n🚀 Running analysis for {idx}. {colname}")

    # 创建输出目录
    out_dir = os.path.join(output_root, f"{idx}-{colname}")
    os.makedirs(out_dir, exist_ok=True)
    log_file = os.path.join(out_dir, f"{colname}.txt")

    # 记录 stdout
    original_stdout = sys.stdout
    sys.stdout = open(log_file, "w")

    try:
        # ======== 原始分析逻辑开始 ========

        # (保持你原始代码内容完全一致，只将 filter_column 改为 colname)
        filter_column = [colname]
        emory_col = "lab_result"
        mimic_col = "lab_result"

        df_emory_grouping_filtered = df_emory_grouping[
            (df_emory_grouping["super_table_col_name"].isin(filter_column)) &
            (df_emory_grouping["import"] == "Yes")
        ]
        df_mimic_grouping_filtered = df_mimic_grouping[
            (df_mimic_grouping["super_table_col_name"].isin(filter_column)) &
            (df_mimic_grouping["import"] == "Yes")
        ]

        print(f"🔍 Emory: super_table_col_name {filter_column} comes from the following [component, component_id, proc_cat_name, proc_desc]: \n {df_emory_grouping_filtered[['component', 'component_id', 'proc_cat_name', 'proc_desc']]}")
        print(f"🔍 MIMIC: super_table_col_name {filter_column} comes from the following [component, component_id, proc_cat_name, proc_desc]: \n {df_mimic_grouping_filtered[['component', 'component_id', 'proc_cat_name', 'proc_desc']]}")

        print()
        
        df_emory_filtered = df_emory[
            (df_emory["super_table_col_name"].isin(filter_column)) &
            (df_emory["import"] == "Yes")
        ]
        print(f"🔍 Emory rows after filtering by super_table_col_name {filter_column}: {df_emory_filtered.shape[0]} rows")
        
        df_mimic_filtered = df_mimic[
            (df_mimic["super_table_col_name"].isin(filter_column)) &
            (df_mimic["import"] == "Yes")
        ]
        print(f"🔍 MIMIC rows after filtering by super_table_col_name {filter_column}: {df_mimic_filtered.shape[0]} rows")

        if emory_col not in df_emory.columns:
            raise ValueError(f"'{emory_col}' not found in Emory file.")
        if mimic_col not in df_mimic.columns:
            raise ValueError(f"'{mimic_col}' not found in MIMIC file.")

        e_col = pd.to_numeric(df_emory_filtered[emory_col], errors='coerce')
        m_col = pd.to_numeric(df_mimic_filtered[mimic_col], errors='coerce')

        print("\n🧼 Null Counts:")
        print(f"Emory - {emory_col}: {e_col.isnull().sum()} nulls out of {len(e_col)}")
        print(f"MIMIC - {mimic_col}: {m_col.isnull().sum()} nulls out of {len(m_col)}")

        pd.options.display.float_format = '{:.4f}'.format

        print("\n📊 Summary Statistics:")
        print("\nEmory:")
        print(e_col.describe(include='all'))
        print("\nMIMIC:")
        print(m_col.describe(include='all'))

        def plot_column_dist(col, source_name, ax, value_range=None):
            if pd.api.types.is_numeric_dtype(col):
                col_to_plot = col.dropna()
                if value_range:
                    col_to_plot = col_to_plot[(col_to_plot >= value_range[0]) & (col_to_plot <= value_range[1])]
                sns.histplot(col_to_plot, kde=True, bins=50, ax=ax)
                ax.set_title(f"{source_name} - Histogram")
                ax.set_xlabel(col.name)
            else:
                col.value_counts().plot(kind='bar', ax=ax)
                ax.set_title(f"{source_name} - Bar Chart")
                ax.set_ylabel("Count")
                ax.set_xlabel(col.name)

        def smart_value_range(col):
            desc = col.describe(percentiles=[0.01, 0.25, 0.5, 0.75, 0.99])
            q1, q3 = desc['25%'], desc['75%']
            iqr = q3 - q1
            lower_iqr, upper_iqr = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            lower_q, upper_q = desc['1%'], desc['99%']
            lower = max(lower_iqr, lower_q)
            upper = min(upper_iqr, upper_q)
            return lower, upper

        e_value_range = smart_value_range(e_col)
        m_value_range = smart_value_range(m_col)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        plot_column_dist(e_col, "Emory", axes[0], value_range=e_value_range)
        plot_column_dist(m_col, "MIMIC", axes[1], value_range=m_value_range)
        plt.tight_layout()
        fig.savefig(os.path.join(out_dir, f"{colname}_hist.png"))
        plt.close(fig)

        if not pd.api.types.is_numeric_dtype(e_col) or not pd.api.types.is_numeric_dtype(m_col):
            print("\n⚠️ Skipping side-by-side KDE plot: one or both columns are not numeric.")
        else:
            plt.figure(figsize=(10, 5))
            e_kde = e_col.dropna()
            m_kde = m_col.dropna()
            if e_value_range or m_value_range:
                e_kde = e_kde[(e_kde >= e_value_range[0]) & (e_kde <= e_value_range[1])]
                m_kde = m_kde[(m_kde >= m_value_range[0]) & (m_kde <= m_value_range[1])]
            sns.kdeplot(e_kde, label='Emory', fill=True)
            sns.kdeplot(m_kde, label='MIMIC', fill=True)
            plt.title(f"Emory ({emory_col}) vs MIMIC ({mimic_col}) - KDE Overlay")
            plt.xlabel("Value")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"{colname}_kde.png"))
            plt.close()

        # ======== 原始分析逻辑结束 ========

    except Exception as e:
        print(f"❌ Error processing {colname}: {e}")

    # 恢复 stdout
    sys.stdout.close()
    sys.stdout = original_stdout

print("\n✅ All 48 analyses completed and saved.")
