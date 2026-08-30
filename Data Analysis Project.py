"""
=====================================================================
FULL FAST-FASHION MATERIAL COMPOSITION PROJECT — COMBINED SCRIPT
=====================================================================
Sections:
  0. Setup / config
  1. Effect size (eta-squared) for materials/composition ANOVAs
  2. Levene's test + Welch's ANOVA (robust to unequal variance)
  3. Multiple regression: price ~ material % + store
  4. Two-way ANOVA: store x region on polyester %
  5. Clean ASOS
  6. Clean Shein
  7. Clean harmonized H&M + Uniqlo (xlsx, GB/Aus regions)
  8. Clean H&M (handm.csv, real prices, region = US)
  9. Material composition tables/graphs by store
 10. One-way ANOVA — materials per product, by store
 11. One-way ANOVA — % composition (polyester / recycled polyester / cotton), by store
 12. Material composition vs price (correlation + scatter)
 13. Region breakdown per store (H&M: US/GB/Aus, Uniqlo: GB/Aus, ASOS: GB, Shein: India)
=====================================================================
"""

import pandas as pd
import numpy as np
import re
import ast
import json
import os
from collections import Counter
from scipy import stats
import matplotlib.pyplot as plt

# =====================================================================
# 0. CONFIG — automatically use the folder this script is saved in
# =====================================================================

try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    os.chdir(SCRIPT_DIR)
except NameError:
    pass

RAW_ASOS_PATH = 'products_asos.csv'
RAW_SHEIN_PATH = 'shein_sample.csv'
RAW_HARMONIZED_PATH = '6_JSONL_component_normalized_public_clean.csv.xlsx'
RAW_HM_PATH = 'handm.csv'

ANOVA_STORE_FILES = {'H&M': 'hm_clean.csv', 'ASOS': 'asos_clean.csv', 'Shein': 'shein_clean.csv'}
TARGETS = {'polyester': 'polyester', 'recycled polyester': 'recycled_polyester', 'cotton': 'cotton'}

print("Working directory:", os.getcwd())
print("Files here:", os.listdir())
print()


# =====================================================================
# HELPER FUNCTIONS
# =====================================================================

def extract_composition_materials(text):
    if pd.isna(text):
        return []
    if not str(text).startswith('COMPOSITION'):
        return []
    comp_section = str(text).split(' , ')[0]
    comp_section = comp_section.replace('COMPOSITION', '', 1).strip()
    materials = []
    for line in comp_section.split('\n'):
        line = line.strip()
        if not line:
            continue
        if ':' in line:
            _, _, line = line.partition(':')
        for part in line.split(','):
            part = part.strip()
            m = re.match(r'^([A-Za-z\u00C0-\u024F\u2122\u00AE\'\-\s]+?)\s*\d+(?:\.\d+)?%', part)
            if m:
                name = re.sub(r'\s+', ' ', m.group(1).strip())
                if name:
                    materials.append(name)
    return materials


def extract_materials_pct_first(text):
    """Parses ASOS/Shein-style text: percentage BEFORE the material name."""
    if not text:
        return []
    materials = []
    for _pct, name in re.findall(r'(\d+(?:\.\d+)?)%\s*([A-Za-z][A-Za-z\s\-]*?)(?=[,\.]|$)', text):
        name_clean = re.sub(r'\s+', ' ', name.strip())
        if name_clean:
            materials.append(name_clean)
    return materials


def extract_recycled_bases(text):
    if pd.isna(text):
        return set()
    bases = set()
    for m in re.finditer(r'Recycled\s+([A-Za-z\u00C0-\u024F\'\-]+)', str(text), re.IGNORECASE):
        bases.add(m.group(1).strip().lower())
    return bases


def classify_material(name, recycled_bases):
    base = name.strip().lower()
    is_recycled = base in recycled_bases or base.startswith('recycled')
    base_clean = re.sub(r'^recycled\s+', '', base)

    if base_clean == 'polyester':
        return 'Recycled Polyester' if is_recycled else 'Polyester (Virgin)'
    if base_clean in ('polyamide', 'nylon'):
        return 'Recycled Polyamide/Nylon' if is_recycled else 'Polyamide/Nylon (Virgin)'
    if base_clean == 'cotton':
        return 'Recycled Cotton' if is_recycled else 'Cotton (Virgin)'
    if base_clean == 'wool':
        return 'Recycled Wool' if is_recycled else 'Wool (Virgin)'

    if is_recycled:
        return f'Recycled {base_clean.title()}'
    return name.strip().title()


def classify_row(materials_list, recycled_bases):
    return [classify_material(mat, recycled_bases) for mat in materials_list]


def extract_pct(text, include_terms, exclude_terms=None):
    """Highest % mention of a material matching ALL include_terms and NONE of exclude_terms."""
    if pd.isna(text):
        return np.nan
    matches = re.findall(r'(\d+(?:\.\d+)?)\s*%\s*([A-Za-z][A-Za-z\s\-]*?)(?=[,\./]|$)', str(text))
    best = []
    for pct, name in matches:
        name_l = name.strip().lower()
        if all(t in name_l for t in include_terms):
            if exclude_terms and any(t in name_l for t in exclude_terms):
                continue
            best.append(float(pct))
    return max(best) if best else np.nan


def normalize_region(r):
    r = str(r).strip().lower()
    if r in ('gb', 'uk') or 'britain' in r or 'united kingdom' in r:
        return 'GB'
    if r in ('au', 'aus') or 'australia' in r:
        return 'Aus'
    return r


def parse_description(text):
    if pd.isna(text):
        return {}
    try:
        items = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return {}
    result = {}
    for item in items:
        if isinstance(item, dict):
            result.update(item)
    return result


def parse_components(json_text):
    """Parses the 'components_structured' JSON column from the harmonized dataset."""
    try:
        components = json.loads(json_text)
    except (json.JSONDecodeError, TypeError):
        return []
    if isinstance(components, dict):
        components = [components]
    materials = []
    for comp in components:
        if not isinstance(comp, dict):
            continue
        for mat in comp.get('materials', []):
            materials.append((mat.get('material'), mat.get('pct'), mat.get('recycled_pct')))
    return materials


def classify_materials_struct(materials_parsed):
    classified = []
    for material, _pct, recycled_pct in materials_parsed:
        if not material:
            continue
        mat_lower = material.strip().lower()
        is_recycled = recycled_pct is not None and recycled_pct > 0
        if mat_lower == 'polyester':
            label = 'Recycled Polyester' if is_recycled else 'Polyester (Virgin)'
        elif mat_lower in ('polyamide', 'nylon'):
            label = 'Recycled Polyamide/Nylon' if is_recycled else 'Polyamide/Nylon (Virgin)'
        elif mat_lower == 'cotton':
            label = 'Recycled Cotton' if is_recycled else 'Cotton (Virgin)'
        elif mat_lower == 'wool':
            label = 'Recycled Wool' if is_recycled else 'Wool (Virgin)'
        else:
            label = material.strip().title()
            if is_recycled:
                label = f'Recycled {label}'
        classified.append(label)
    return classified


def total_polyester_pct(materials_parsed):
    """Sum of ALL polyester % (recycled + virgin combined) per product."""
    total = sum(pct for mat, pct, _rpct in materials_parsed
                if mat and mat.strip().lower() == 'polyester' and pct is not None)
    return total if total > 0 else None


def recycled_polyester_pct_struct(materials_parsed):
    total, recycled = 0.0, 0.0
    for material, pct, recycled_pct in materials_parsed:
        if material and material.strip().lower() == 'polyester' and pct is not None:
            total += pct
            if recycled_pct is not None:
                recycled += pct * (recycled_pct / 100.0)
    return round(100 * recycled / total, 1) if total else None


def recycled_polyester_pct_hm(text):
    """For H&M's raw COMPOSITION-style text: sums explicit 'X% Recycled polyester' mentions."""
    if pd.isna(text):
        return np.nan
    matches = re.findall(r'(\d+(?:\.\d+)?)%\s*Recycled\s+polyester\b', text, re.IGNORECASE)
    return sum(float(m) for m in matches) if matches else np.nan


def save_and_report(df, base_name):
    """Standard save + material count report used across all stores."""
    df.to_csv(f'{base_name}_clean.csv', index=False)
    organized = df.sort_values(['primaryMaterial', 'productName']).reset_index(drop=True) \
        if 'primaryMaterial' in df.columns and 'productName' in df.columns else df
    organized.to_csv(f'{base_name}_by_material.csv', index=False)

    if 'allMaterials' in df.columns:
        all_materials = []
        for s in df['allMaterials'].dropna():
            for m in str(s).split(';'):
                m = m.strip()
                if m:
                    all_materials.append(m)
        counts = Counter(all_materials)
        counts_df = pd.DataFrame(counts.most_common(), columns=['material', 'productCount'])
        counts_df.to_csv(f'{base_name}_material_counts.csv', index=False)
        print(f"\n--- {base_name}: top materials ---")
        print(counts_df.head(15).to_string(index=False))

    print(f"\nSaved: {base_name}_clean.csv, {base_name}_by_material.csv, {base_name}_material_counts.csv")


def get_top_materials(store_df, top_n=10):
    counter = Counter()
    for s in store_df['allMaterials'].dropna():
        s = str(s).strip()
        if not s:
            continue
        for m in s.split(';'):
            m = m.strip()
            if m:
                counter[m] += 1
    total = len(store_df)
    unparsed = int((store_df['primaryMaterial'] == 'Unknown').sum()) if 'primaryMaterial' in store_df.columns else 0
    top = counter.most_common(top_n)
    table = pd.DataFrame([{'material': m, 'productCount': c, 'pctOfProducts': round(100 * c / total, 1)} for m, c in top])
    return total, unparsed, table


def plot_store_materials(store_name, total, unparsed, table):
    if table.empty:
        return
    colors = ['#1D9E75' if m.lower().startswith('recycled') else '#888780' for m in table['material']]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(table['material'][::-1], table['pctOfProducts'][::-1], color=colors[::-1])
    ax.set_xlabel('% of products containing this material')
    ax.set_title(f'{store_name} — top materials\n({total:,} products, {unparsed:,} unparsed)')
    ax.set_xlim(0, 100)
    for i, v in enumerate(table['pctOfProducts'][::-1]):
        ax.text(v + 1, i, f'{v}%', va='center', fontsize=9)
    plt.tight_layout()
    safe = store_name.replace('&', 'and').replace(' ', '_')
    plt.savefig(f'{safe}_material_composition.png', dpi=150)
    print(f"Saved: {safe}_material_composition.png")


def eta_squared(groups):
    """Eta-squared: proportion of total variance explained by group membership.
    Rule of thumb: 0.01 small, 0.06 medium, 0.14 large."""
    all_vals = np.concatenate(groups)
    grand_mean = np.mean(all_vals)
    ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups)
    ss_total = sum((all_vals - grand_mean) ** 2)
    return ss_between / ss_total if ss_total > 0 else np.nan


def welch_anova(groups):
    """Welch's ANOVA — robust alternative to one-way ANOVA when variances
    are unequal across groups (doesn't assume homogeneity of variance)."""
    k = len(groups)
    ni = np.array([len(g) for g in groups])
    mi = np.array([np.mean(g) for g in groups])
    vi = np.array([np.var(g, ddof=1) for g in groups])

    wi = ni / vi
    grand_mean = np.sum(wi * mi) / np.sum(wi)

    numerator = np.sum(wi * (mi - grand_mean) ** 2) / (k - 1)
    denom_term = np.sum((1 - wi / np.sum(wi)) ** 2 / (ni - 1))
    denominator = 1 + (2 * (k - 2) / (k ** 2 - 1)) * denom_term

    f_stat = numerator / denominator
    df1 = k - 1
    df2 = (k ** 2 - 1) / (3 * denom_term)
    p_value = stats.f.sf(f_stat, df1, df2)

    return f_stat, p_value, df1, df2


# =====================================================================
# 1. EFFECT SIZE (eta-squared) for materials/composition ANOVAs
# =====================================================================
print("\n" + "=" * 90)
print("SECTION 1: EFFECT SIZE (eta-squared) FOR EACH ANOVA")
print("=" * 90)
print("Rule of thumb: 0.01 = small, 0.06 = medium, 0.14 = large effect")

followup_frames = []
for store_name, path in ANOVA_STORE_FILES.items():
    if os.path.isfile(path):
        d = pd.read_csv(path)
        d['store'] = store_name
        d['price'] = pd.to_numeric(d['price'], errors='coerce')
        d['materialCount'] = d['allMaterials'].fillna('').apply(lambda s: len([m for m in str(s).split(';') if m.strip()]))
        followup_frames.append(d[['store', 'materials', 'price', 'materialCount']])
    else:
        print(f"NOT FOUND: '{path}' — skipping {store_name}")

if not followup_frames:
    print("No store files found — cannot run Sections 1-3. Run the cleaning sections below first.")
else:
    followup_combined = pd.concat(followup_frames, ignore_index=True)
    followup_combined['polyester_pct'] = followup_combined['materials'].apply(
        lambda t: extract_pct(t, ['polyester'], ['recycled']))
    followup_combined['recycled_polyester_pct'] = followup_combined['materials'].apply(
        lambda t: extract_pct(t, ['recycled', 'polyester']))
    followup_combined['cotton_pct'] = followup_combined['materials'].apply(
        lambda t: extract_pct(t, ['cotton'], ['recycled']))

    MEASURES = {
        'materialCount': 'Materials per product',
        'polyester_pct': 'Polyester %',
        'recycled_polyester_pct': 'Recycled Polyester %',
        'cotton_pct': 'Cotton %',
    }

    for col, label in MEASURES.items():
        sub = followup_combined.dropna(subset=[col])
        groups = [g[col].values for _store_name, g in sub.groupby('store') if len(g) >= 2]
        if len(groups) < 2:
            continue
        f_stat, p_value = stats.f_oneway(*groups)
        eta2 = eta_squared(groups)
        size_label = 'small' if eta2 < 0.06 else ('medium' if eta2 < 0.14 else 'large')
        print(f"\n{label}:")
        print(f"  F={f_stat:.2f}, p={p_value:.6f}, eta-squared={eta2:.4f} ({size_label} effect)")


    # =================================================================
    # 2. LEVENE'S TEST + WELCH'S ANOVA
    # =================================================================
    print("\n" + "=" * 90)
    print("SECTION 2: LEVENE'S TEST (equal variance check) + WELCH'S ANOVA")
    print("=" * 90)

    for col, label in MEASURES.items():
        sub = followup_combined.dropna(subset=[col])
        groups = [g[col].values for _store_name, g in sub.groupby('store') if len(g) >= 2]
        if len(groups) < 2:
            continue

        levene_stat, levene_p = stats.levene(*groups)
        print(f"\n{label}:")
        print(f"  Levene's test: stat={levene_stat:.4f}, p={levene_p:.6f}", end='  ')
        if levene_p < 0.05:
            print("-> Variances are UNEQUAL across stores (standard ANOVA assumption violated)")
        else:
            print("-> Variances are roughly equal (standard ANOVA assumption OK)")

        f_stat, p_value, df1, df2 = welch_anova(groups)
        print(f"  Welch's ANOVA (robust to unequal variance): F={f_stat:.2f}, df1={df1}, df2={df2:.1f}, p={p_value:.6f}",
              "(significant)" if p_value < 0.05 else "(not significant)")


    # =================================================================
    # 3. MULTIPLE REGRESSION: price ~ material % + store
    # =================================================================
    print("\n" + "=" * 90)
    print("SECTION 3: MULTIPLE REGRESSION: price ~ recycled% + polyester% + cotton% + store")
    print("=" * 90)

    reg_df = followup_combined.dropna(subset=['price']).copy()
    for col in ['recycled_polyester_pct', 'polyester_pct', 'cotton_pct']:
        reg_df[col] = reg_df[col].fillna(0)

    try:
        import statsmodels.formula.api as smf

        reg_model = smf.ols(
            'price ~ recycled_polyester_pct + polyester_pct + cotton_pct + C(store)',
            data=reg_df
        ).fit()

        print(reg_model.summary())

        print("\n--- Plain-English interpretation guide ---")
        print("Positive coef = higher % of that material is associated with HIGHER price, controlling for store")
        print("Negative coef = higher % of that material is associated with LOWER price, controlling for store")
        print("P>|t| < 0.05 means that material's effect on price is statistically significant")
        print("C(store)[T.X] rows show each store's price difference vs. the baseline store,")
        print("holding material composition constant.")

    except ImportError:
        print("statsmodels not installed. Install via Thonny's Tools > Manage packages, then re-run.")
        print("Falling back to plain correlations (does NOT control for store):")
        for col in ['recycled_polyester_pct', 'polyester_pct', 'cotton_pct']:
            r, p = stats.pearsonr(reg_df[col], reg_df['price'])
            print(f"  {col}: r={r:.4f}, p={p:.6f}")


# =====================================================================
# 4. TWO-WAY ANOVA: store x region on polyester %
# =====================================================================
print("\n" + "=" * 90)
print("SECTION 4: TWO-WAY ANOVA: store x region on polyester %")
print("=" * 90)

two_way_frames = []

if os.path.isfile('hm_uniqlo_clean.csv'):
    hu_2way = pd.read_csv('hm_uniqlo_clean.csv')
    print("\nDiagnostic — polyesterPctTotal parse success by store (hm_uniqlo_clean.csv):")
    print(hu_2way.groupby('brandName')['polyesterPctTotal'].apply(lambda s: s.notna().sum()))
    hu_2way = hu_2way.rename(columns={'brandName': 'store'})
    two_way_frames.append(hu_2way[['store', 'region', 'polyesterPctTotal']].rename(
        columns={'polyesterPctTotal': 'polyester_pct'}))
else:
    print("NOT FOUND: 'hm_uniqlo_clean.csv'")

if os.path.isfile('hm_clean.csv'):
    hm_2way = pd.read_csv('hm_clean.csv')
    hm_2way['region'] = 'US'
    hm_2way['polyester_pct'] = hm_2way['materials'].apply(lambda t: extract_pct(t, ['polyester'], ['recycled']))
    hm_2way['store'] = 'hm'
    two_way_frames.append(hm_2way[['store', 'region', 'polyester_pct']])
else:
    print("NOT FOUND: 'hm_clean.csv'")

if two_way_frames:
    two_way_df = pd.concat(two_way_frames, ignore_index=True).dropna(subset=['polyester_pct'])
    print(f"\nData used: {len(two_way_df)} products")
    print(two_way_df.groupby(['store', 'region']).size())

    n_stores = two_way_df['store'].nunique()
    n_regions = two_way_df['region'].nunique()

    if n_stores < 2 or n_regions < 2:
        print(f"\nCannot run a two-way ANOVA: need at least 2 stores AND 2 regions with data.")
        print(f"Currently have {n_stores} store(s) and {n_regions} region(s) after dropping unparsed rows.")
    else:
        try:
            import statsmodels.formula.api as smf
            from statsmodels.stats.anova import anova_lm

            twoway_model = smf.ols('polyester_pct ~ C(store) * C(region)', data=two_way_df).fit()
            anova_table = anova_lm(twoway_model, typ=2)
            print("\n--- Two-way ANOVA table ---")
            print(anova_table)

            print("\n--- Interpretation guide ---")
            print("C(store): does polyester % differ by store, averaged across regions?")
            print("C(region): does polyester % differ by region, averaged across stores?")
            print("C(store):C(region): does the store effect DEPEND on region (interaction)?")
            print("PR(>F) < 0.05 means that effect is statistically significant.")

            fig, ax = plt.subplots(figsize=(8, 5))
            for store in two_way_df['store'].unique():
                sub = two_way_df[two_way_df['store'] == store]
                means = sub.groupby('region')['polyester_pct'].mean()
                ax.plot(means.index, means.values, marker='o', label=store)
            ax.set_xlabel('Region')
            ax.set_ylabel('Mean Polyester %')
            ax.set_title('Store x Region interaction: Polyester %')
            ax.legend()
            plt.tight_layout()
            plt.savefig('twoway_anova_interaction_plot.png', dpi=150)
            print("\nSaved: twoway_anova_interaction_plot.png")

        except ImportError:
            print("statsmodels not installed. Install via Thonny's Tools > Manage packages, then re-run.")
        except ValueError as e:
            print(f"\nANOVA failed: {e}")
else:
    print("Not enough region-tagged data available to run the two-way ANOVA.")


# =====================================================================
# 5. CLEAN ASOS
# =====================================================================
print("\n" + "=" * 90)
print("SECTION 5: CLEAN ASOS")
print("=" * 90)

if os.path.isfile(RAW_ASOS_PATH):
    asos_raw = pd.read_csv(RAW_ASOS_PATH)
    asos_raw = asos_raw.dropna(subset=['sku', 'name', 'description']).copy()
    before = len(asos_raw)
    asos_raw = asos_raw.drop_duplicates(subset=['sku']).reset_index(drop=True)
    print(f"Rows before dedup: {before}  Rows after dedup: {len(asos_raw)}")

    def extract_asos_brand(url):
        if pd.isna(url):
            return 'Unknown'
        m = re.search(r'asos\.com/([^/]+)/', url)
        return m.group(1).replace('-', ' ').title() if m else 'Unknown'

    asos_raw['brandName'] = asos_raw['url'].apply(extract_asos_brand)

    asos_raw['description_parsed'] = asos_raw['description'].apply(parse_description)
    asos_raw['productDetails'] = asos_raw['description_parsed'].apply(lambda d: d.get('Product Details', ''))
    asos_raw['aboutMe'] = asos_raw['description_parsed'].apply(lambda d: d.get('About Me', ''))

    FOOTWEAR_KEYWORDS = ['shoe', 'trainer', 'boot', 'sandal', 'sneaker', 'heel', 'flip-flop',
                         'flip flop', 'loafer', 'slipper', 'espadrille', 'clog', 'mule']
    is_footwear = asos_raw['name'].str.lower().str.contains('|'.join(FOOTWEAR_KEYWORDS), na=False) | \
                  asos_raw['url'].str.lower().str.contains('|'.join(FOOTWEAR_KEYWORDS), na=False)

    before = len(asos_raw)
    asos_clean = asos_raw[~is_footwear].copy().reset_index(drop=True)
    print(f"Rows before footwear removal: {before}  Rows after: {len(asos_clean)}")

    asos_clean['materials_list'] = asos_clean['aboutMe'].apply(extract_materials_pct_first)
    asos_clean['recycled_bases'] = asos_clean['aboutMe'].apply(extract_recycled_bases)
    asos_clean['materials_list_classified'] = asos_clean.apply(
        lambda row: classify_row(row['materials_list'], row['recycled_bases']), axis=1)

    asos_clean['primaryMaterial'] = asos_clean['materials_list_classified'].apply(lambda lst: lst[0] if lst else 'Unknown')
    asos_clean['allMaterials'] = asos_clean['materials_list_classified'].apply(lambda lst: '; '.join(sorted(set(lst))) if lst else '')
    asos_clean['containsRecycledPolyester'] = asos_clean['materials_list_classified'].apply(lambda lst: 'Recycled Polyester' in lst)
    asos_clean['containsVirginPolyester'] = asos_clean['materials_list_classified'].apply(lambda lst: 'Polyester (Virgin)' in lst)

    asos_final = asos_clean.rename(columns={'sku': 'productId', 'name': 'productName', 'color': 'colorName'}).copy()
    asos_final['materials'] = asos_final['aboutMe']
    output_cols = ['productId', 'productName', 'brandName', 'url', 'price', 'colorName',
                   'productDetails', 'materials', 'primaryMaterial', 'allMaterials',
                   'containsRecycledPolyester', 'containsVirginPolyester']
    asos_final = asos_final[output_cols]

    save_and_report(asos_final, 'asos')
else:
    print(f"NOT FOUND: '{RAW_ASOS_PATH}' — skipping Section 5.")


# =====================================================================
# 6. CLEAN SHEIN
# =====================================================================
print("\n" + "=" * 90)
print("SECTION 6: CLEAN SHEIN")
print("=" * 90)

if os.path.isfile(RAW_SHEIN_PATH):
    shein_raw = pd.read_csv(RAW_SHEIN_PATH, sep=';', encoding='latin1', on_bad_lines='skip', engine='python')
    shein_raw = shein_raw.dropna(subset=['sku', 'name', 'description']).copy()
    before = len(shein_raw)
    shein_raw = shein_raw.drop_duplicates(subset=['sku']).reset_index(drop=True)
    print(f"Rows before dedup: {before}  Rows after dedup: {len(shein_raw)}")

    shein_raw['description_parsed'] = shein_raw['description'].apply(parse_description)
    shein_raw['productType'] = shein_raw['description_parsed'].apply(lambda d: d.get('Type', ''))
    shein_raw['composition'] = shein_raw['description_parsed'].apply(lambda d: d.get('Composition', ''))
    shein_raw['colorName'] = shein_raw['description_parsed'].apply(lambda d: d.get('Color', ''))

    NON_CLOTHING_KEYWORDS = ['shoe', 'boot', 'sandal', 'sneaker', 'heel', 'flip-flop', 'flip flop',
                              'loafer', 'slipper', 'espadrille', 'clog', 'mule', 'wedge', 'skate shoes',
                              'bag', 'backpack', 'wallet', 'purse', 'pillowcase', 'jewelry', 'jewellery',
                              'necklace', 'earring', 'ring', 'bracelet', 'sunglasses', 'hair',
                              'phone case', 'decoration']
    pattern = '|'.join(NON_CLOTHING_KEYWORDS)
    is_non_clothing = shein_raw['name'].str.lower().str.contains(pattern, na=False) | \
                       shein_raw['productType'].str.lower().str.contains(pattern, na=False)

    before = len(shein_raw)
    shein_clean = shein_raw[~is_non_clothing].copy().reset_index(drop=True)
    print(f"Rows before non-clothing removal: {before}  Rows after: {len(shein_clean)}")

    shein_clean['materials_list'] = shein_clean['composition'].apply(extract_materials_pct_first)
    shein_clean['recycled_bases'] = shein_clean['composition'].apply(extract_recycled_bases)
    shein_clean['materials_list_classified'] = shein_clean.apply(
        lambda row: classify_row(row['materials_list'], row['recycled_bases']), axis=1)

    shein_clean['primaryMaterial'] = shein_clean['materials_list_classified'].apply(lambda lst: lst[0] if lst else 'Unknown')
    shein_clean['allMaterials'] = shein_clean['materials_list_classified'].apply(lambda lst: '; '.join(sorted(set(lst))) if lst else '')
    shein_clean['containsRecycledPolyester'] = shein_clean['materials_list_classified'].apply(lambda lst: 'Recycled Polyester' in lst)
    shein_clean['containsVirginPolyester'] = shein_clean['materials_list_classified'].apply(lambda lst: 'Polyester (Virgin)' in lst)

    shein_final = shein_clean.rename(columns={'sku': 'productId', 'name': 'productName', 'brand': 'brandName',
                                               'productType': 'mainCatCode', 'composition': 'materials'}).copy()
    output_cols = ['productId', 'productName', 'brandName', 'url', 'price', 'colorName',
                   'mainCatCode', 'materials', 'primaryMaterial', 'allMaterials',
                   'containsRecycledPolyester', 'containsVirginPolyester']
    shein_final = shein_final[output_cols]

    save_and_report(shein_final, 'shein')
else:
    print(f"NOT FOUND: '{RAW_SHEIN_PATH}' — skipping Section 6.")


# =====================================================================
# 7. CLEAN HARMONIZED H&M + UNIQLO (GB / Aus regions)
# =====================================================================
print("\n" + "=" * 90)
print("SECTION 7: CLEAN HARMONIZED H&M + UNIQLO")
print("=" * 90)

if os.path.isfile(RAW_HARMONIZED_PATH):
    harm_raw = pd.read_excel(RAW_HARMONIZED_PATH)
    harm_raw = harm_raw.dropna(subset=['components_structured']).copy()

    FOOTWEAR_KEYWORDS = ['shoe', 'boot', 'sandal', 'sneaker', 'heel', 'slipper', 'loafer', 'trainer']
    is_footwear = harm_raw['product_name'].str.lower().str.contains('|'.join(FOOTWEAR_KEYWORDS), na=False) | \
                  harm_raw['raw_category'].str.lower().str.contains('|'.join(FOOTWEAR_KEYWORDS), na=False)
    before = len(harm_raw)
    harm_clean = harm_raw[~is_footwear].copy().reset_index(drop=True)
    print(f"Rows before footwear removal: {before}  Rows after: {len(harm_clean)}")

    harm_clean['materials_parsed'] = harm_clean['components_structured'].apply(parse_components)
    harm_clean['materials_list_classified'] = harm_clean['materials_parsed'].apply(classify_materials_struct)
    harm_clean['primaryMaterial'] = harm_clean['materials_list_classified'].apply(lambda lst: lst[0] if lst else 'Unknown')
    harm_clean['allMaterials'] = harm_clean['materials_list_classified'].apply(lambda lst: '; '.join(sorted(set(lst))) if lst else '')
    harm_clean['containsRecycledPolyester'] = harm_clean['materials_list_classified'].apply(lambda lst: 'Recycled Polyester' in lst)
    harm_clean['containsVirginPolyester'] = harm_clean['materials_list_classified'].apply(lambda lst: 'Polyester (Virgin)' in lst)
    harm_clean['recycledPolyesterPct'] = harm_clean['materials_parsed'].apply(recycled_polyester_pct_struct)
    harm_clean['polyesterPctTotal'] = harm_clean['materials_parsed'].apply(total_polyester_pct)
    harm_clean['region'] = harm_clean['region'].apply(normalize_region)

    harm_final = harm_clean.rename(columns={
        'parent_product_id': 'productId', 'product_name': 'productName', 'brand': 'brandName',
        'raw_category': 'mainCatCode', 'raw_material_text': 'materials',
    }).copy()
    output_cols = ['productId', 'productName', 'brandName', 'region', 'gender_section', 'url',
                   'mainCatCode', 'parent_category', 'variant_colour', 'materials',
                   'primaryMaterial', 'allMaterials', 'containsRecycledPolyester',
                   'containsVirginPolyester', 'recycledPolyesterPct', 'polyesterPctTotal']
    harm_final = harm_final[output_cols]

    harm_final.to_csv('hm_uniqlo_clean.csv', index=False)
    print("Saved: hm_uniqlo_clean.csv")

    brand_summary = harm_final.groupby('brandName').agg(
        products=('productId', 'count'),
        recycled_polyester_products=('containsRecycledPolyester', 'sum'),
    ).reset_index()
    brand_summary['pct_recycled_polyester'] = (100 * brand_summary['recycled_polyester_products'] / brand_summary['products']).round(2)
    print(brand_summary.to_string(index=False))
else:
    print(f"NOT FOUND: '{RAW_HARMONIZED_PATH}' — skipping Section 7. (Requires openpyxl installed.)")


# =====================================================================
# 8. CLEAN H&M (handm.csv, real prices, tagged region = US)
# =====================================================================
print("\n" + "=" * 90)
print("SECTION 8: CLEAN H&M")
print("=" * 90)

if os.path.isfile(RAW_HM_PATH):
    hm_raw = pd.read_csv(RAW_HM_PATH)
    if 'Unnamed: 0' in hm_raw.columns:
        hm_raw.drop('Unnamed: 0', axis=1, inplace=True)

    shoe_cats = [c for c in hm_raw['mainCatCode'].dropna().unique() if 'shoe' in c.lower()]
    shoe_cats.append('ladies_nightwear_slippers')
    before = len(hm_raw)
    hm_clean = hm_raw[~hm_raw['mainCatCode'].isin(shoe_cats)].copy().reset_index(drop=True)
    print(f"Rows before: {before}  Rows after footwear removal: {len(hm_clean)}")

    hm_clean['materials_list'] = hm_clean['materials'].apply(extract_composition_materials)
    hm_clean['recycled_bases'] = hm_clean['materials'].apply(extract_recycled_bases)
    hm_clean['materials_list_classified'] = hm_clean.apply(
        lambda row: classify_row(row['materials_list'], row['recycled_bases']), axis=1)

    hm_clean['primaryMaterial'] = hm_clean['materials_list_classified'].apply(lambda lst: lst[0] if lst else 'Unknown')
    hm_clean['allMaterials'] = hm_clean['materials_list_classified'].apply(lambda lst: '; '.join(sorted(set(lst))) if lst else '')
    hm_clean['containsRecycledPolyester'] = hm_clean['materials_list_classified'].apply(lambda lst: 'Recycled Polyester' in lst)
    hm_clean['containsVirginPolyester'] = hm_clean['materials_list_classified'].apply(lambda lst: 'Polyester (Virgin)' in lst)
    hm_clean['recycledPolyesterPct'] = hm_clean['materials'].apply(recycled_polyester_pct_hm)
    hm_clean['region'] = 'US'
    hm_clean['brandName'] = 'hm'

    output_cols = ['productId', 'productName', 'brandName', 'region', 'url', 'price', 'stockState',
                   'colorName', 'mainCatCode', 'details', 'materials', 'primaryMaterial', 'allMaterials',
                   'containsRecycledPolyester', 'containsVirginPolyester', 'recycledPolyesterPct']
    hm_final = hm_clean[output_cols].copy()

    save_and_report(hm_final, 'hm')
    print(f"Products with price: {hm_final['price'].notna().sum()}")
else:
    print(f"NOT FOUND: '{RAW_HM_PATH}' — skipping Section 8.")


# =====================================================================
# 9. MATERIAL COMPOSITION BY STORE — TABLES + GRAPHS
# =====================================================================
print("\n" + "=" * 90)
print("SECTION 9: MATERIAL COMPOSITION BY STORE (tables + graphs)")
print("=" * 90)

STORE_FILES = {
    'H&M': 'hm_clean.csv',
    'Uniqlo': None,  # pulled from hm_uniqlo_clean.csv below
    'ASOS': 'asos_clean.csv',
    'Shein': 'shein_clean.csv',
}

for store_name, path in STORE_FILES.items():
    if store_name == 'Uniqlo':
        if os.path.isfile('hm_uniqlo_clean.csv'):
            hu = pd.read_csv('hm_uniqlo_clean.csv')
            store_df = hu[hu['brandName'] == 'uniqlo']
        else:
            print("NOT FOUND: 'hm_uniqlo_clean.csv' — skipping Uniqlo")
            continue
    else:
        if not path or not os.path.isfile(path):
            print(f"NOT FOUND: '{path}' — skipping {store_name}")
            continue
        store_df = pd.read_csv(path)

    total, unparsed, table = get_top_materials(store_df)
    print(f"\n=== {store_name} === Total: {total}  Unparsed: {unparsed}")
    print(table.to_string(index=False))
    plot_store_materials(store_name, total, unparsed, table)
    table.to_csv(f'{store_name.replace("&", "and").replace(" ", "_")}_top_materials.csv', index=False)


# =====================================================================
# 10. ONE-WAY ANOVA — materials per product (complexity), by store
# =====================================================================
print("\n" + "=" * 90)
print("SECTION 10: ANOVA — materials per product, by store")
print("=" * 90)

anova_frames = []
for store_name, path in ANOVA_STORE_FILES.items():
    if os.path.isfile(path):
        d = pd.read_csv(path)
        d['materialCount'] = d['allMaterials'].fillna('').apply(lambda s: len([m for m in str(s).split(';') if m.strip()]))
        d['store'] = store_name
        anova_frames.append(d[['store', 'materialCount']])
    else:
        print(f"NOT FOUND: '{path}' — skipping {store_name} in ANOVA")

if anova_frames:
    anova_combined = pd.concat(anova_frames, ignore_index=True)
    anova_valid = anova_combined[anova_combined['materialCount'] > 0]

    summary = anova_valid.groupby('store')['materialCount'].agg(['count', 'mean', 'std', 'min', 'max'])
    print(summary.round(2).to_string())

    groups = [g['materialCount'].values for _store_name, g in anova_valid.groupby('store')]
    if len(groups) >= 2:
        f_stat, p_value = stats.f_oneway(*groups)
        print(f"\nF-statistic: {f_stat:.4f}, p-value: {p_value:.6f}",
              "(significant)" if p_value < 0.05 else "(not significant)")

        try:
            from statsmodels.stats.multicomp import pairwise_tukeyhsd
            tukey = pairwise_tukeyhsd(endog=anova_valid['materialCount'], groups=anova_valid['store'], alpha=0.05)
            print(tukey)
        except ImportError:
            print("(statsmodels not installed — skipping Tukey HSD)")

    fig, ax = plt.subplots(figsize=(8, 5))
    store_order = anova_valid['store'].unique()
    box_data = [anova_valid[anova_valid['store'] == s]['materialCount'].values for s in store_order]
    ax.boxplot(box_data)
    ax.set_xticks(range(1, len(store_order) + 1))
    ax.set_xticklabels(store_order)
    ax.set_ylabel('Number of distinct materials per product')
    ax.set_title('Material composition complexity by store')
    plt.tight_layout()
    plt.savefig('anova_materials_per_product_boxplot.png', dpi=150)
    print("Saved: anova_materials_per_product_boxplot.png")
else:
    print("No data available for this ANOVA.")


# =====================================================================
# 11. ONE-WAY ANOVA — % composition (polyester / recycled polyester / cotton), by store
# =====================================================================
print("\n" + "=" * 90)
print("SECTION 11: ANOVA — material % composition, by store")
print("=" * 90)

pct_frames = []
for store_name, path in ANOVA_STORE_FILES.items():
    if os.path.isfile(path):
        d = pd.read_csv(path)
        d['store'] = store_name
        pct_frames.append(d[['store', 'materials']])
    else:
        print(f"NOT FOUND: '{path}' — skipping {store_name}")

if pct_frames:
    pct_combined = pd.concat(pct_frames, ignore_index=True)

    for target_str, safe_name in TARGETS.items():
        print(f"\n--- {target_str} ---")
        pct_combined[f'{safe_name}_pct'] = pct_combined['materials'].apply(
            lambda t, ts=target_str: extract_pct(t, [ts])
        )
        valid = pct_combined.dropna(subset=[f'{safe_name}_pct'])
        print(f"Products containing '{target_str}': {len(valid)}")
        if valid.empty:
            continue

        summary = valid.groupby('store')[f'{safe_name}_pct'].agg(['count', 'mean', 'std'])
        print(summary.round(2).to_string())

        groups = [g[f'{safe_name}_pct'].values for _store_name, g in valid.groupby('store') if len(g) >= 2]
        if len(groups) >= 2:
            f_stat, p_value = stats.f_oneway(*groups)
            print(f"ANOVA: F={f_stat:.4f}, p={p_value:.6f}", "(significant)" if p_value < 0.05 else "(not significant)")

        fig, ax = plt.subplots(figsize=(8, 5))
        store_order = valid['store'].unique()
        box_data = [valid[valid['store'] == s][f'{safe_name}_pct'].values for s in store_order]
        ax.boxplot(box_data)
        ax.set_xticks(range(1, len(store_order) + 1))
        ax.set_xticklabels(store_order)
        ax.set_ylabel(f'{target_str.title()} composition (%)')
        ax.set_title(f'{target_str.title()} composition % by store')
        plt.tight_layout()
        plt.savefig(f'anova_{safe_name}_pct_boxplot.png', dpi=150)
        print(f"Saved: anova_{safe_name}_pct_boxplot.png")
else:
    print("No data available for this section.")


# =====================================================================
# 12. MATERIAL COMPOSITION vs PRICE
# =====================================================================
print("\n" + "=" * 90)
print("SECTION 12: MATERIAL COMPOSITION vs PRICE")
print("=" * 90)

price_frames = []
for store_name, path in ANOVA_STORE_FILES.items():
    if os.path.isfile(path):
        d = pd.read_csv(path)
        d['store'] = store_name
        d['price'] = pd.to_numeric(d['price'], errors='coerce')
        price_frames.append(d[['store', 'materials', 'price']])
    else:
        print(f"NOT FOUND: '{path}' — skipping {store_name}")

if price_frames:
    price_combined = pd.concat(price_frames, ignore_index=True)

    for target_str, safe_name in TARGETS.items():
        price_combined[f'{safe_name}_pct'] = price_combined['materials'].apply(
            lambda t, ts=target_str: extract_pct(t, [ts])
        )
        sub = price_combined.dropna(subset=[f'{safe_name}_pct', 'price'])
        print(f"\n--- {target_str} % vs price --- (n={len(sub)})")
        if len(sub) < 3:
            print("Not enough data — skipping.")
            continue

        pearson_r, pearson_p = stats.pearsonr(sub[f'{safe_name}_pct'], sub['price'])
        spearman_r, spearman_p = stats.spearmanr(sub[f'{safe_name}_pct'], sub['price'])
        print(f"Pearson r={pearson_r:.4f}, p={pearson_p:.6f} | Spearman r={spearman_r:.4f}, p={spearman_p:.6f}")

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(sub[f'{safe_name}_pct'], sub['price'], alpha=0.4, s=15, color='#378ADD')
        z = np.polyfit(sub[f'{safe_name}_pct'], sub['price'], 1)
        trend_x = np.linspace(sub[f'{safe_name}_pct'].min(), sub[f'{safe_name}_pct'].max(), 100)
        ax.plot(trend_x, np.polyval(z, trend_x), color='#D85A30', linewidth=2)
        ax.set_xlabel(f'{target_str.title()} (%)')
        ax.set_ylabel('Price')
        ax.set_title(f'{target_str.title()} % vs price\nr={pearson_r:.3f}, p={pearson_p:.4f}')
        plt.tight_layout()
        plt.savefig(f'price_vs_{safe_name}.png', dpi=150)
        print(f"Saved: price_vs_{safe_name}.png")
else:
    print("No data available for this section.")


# =====================================================================
# 13. REGION BREAKDOWN PER STORE (H&M: US/GB/Aus, Uniqlo: GB/Aus, ASOS: GB, Shein: India)
# =====================================================================
print("\n" + "=" * 90)
print("SECTION 13: REGION BREAKDOWN PER STORE")
print("=" * 90)

# ---- H&M ----
print("\n--- H&M regions ---")
hm_region_frames = []
if os.path.isfile('hm_clean.csv'):
    us = pd.read_csv('hm_clean.csv')
    us['region'] = 'US'
    us['price'] = pd.to_numeric(us['price'], errors='coerce')
    hm_region_frames.append(us[['region', 'materials', 'price']])
if os.path.isfile('hm_uniqlo_clean.csv'):
    raw = pd.read_csv('hm_uniqlo_clean.csv')
    hm_only = raw[raw['brandName'] == 'hm'].copy()
    hm_only['price'] = np.nan
    hm_region_frames.append(hm_only[['region', 'materials', 'price']])

if hm_region_frames:
    hm_region_df = pd.concat(hm_region_frames, ignore_index=True)
    print(hm_region_df['region'].value_counts())
    for target_str, safe_name in TARGETS.items():
        hm_region_df[f'{safe_name}_pct'] = hm_region_df['materials'].apply(
            lambda t, ts=target_str: extract_pct(t, [ts])
        )
        for region in ['US', 'GB', 'Aus']:
            sub = hm_region_df[hm_region_df['region'] == region].dropna(subset=[f'{safe_name}_pct'])
            if sub.empty:
                continue
            print(f"H&M {region} — {target_str}: n={len(sub)}, mean={sub[f'{safe_name}_pct'].mean():.2f}%")
else:
    print("No H&M files found for region breakdown.")

# ---- Uniqlo ----
print("\n--- Uniqlo regions ---")
if os.path.isfile('hm_uniqlo_clean.csv'):
    raw = pd.read_csv('hm_uniqlo_clean.csv')
    uq_df = raw[raw['brandName'] == 'uniqlo'].copy()
    print(uq_df['region'].value_counts())
    for target_str, safe_name in TARGETS.items():
        uq_df[f'{safe_name}_pct'] = uq_df['materials'].apply(
            lambda t, ts=target_str: extract_pct(t, [ts])
        )
        for region in ['GB', 'Aus']:
            sub = uq_df[uq_df['region'] == region].dropna(subset=[f'{safe_name}_pct'])
            if sub.empty:
                continue
            print(f"Uniqlo {region} — {target_str}: n={len(sub)}, mean={sub[f'{safe_name}_pct'].mean():.2f}%")
else:
    print("hm_uniqlo_clean.csv not found — skipping Uniqlo region breakdown.")

# ---- ASOS (GB) ----
print("\n--- ASOS GB ---")
if os.path.isfile('asos_clean.csv'):
    asos_region_df = pd.read_csv('asos_clean.csv')
    asos_region_df['region'] = 'GB'
    for target_str, safe_name in TARGETS.items():
        asos_region_df[f'{safe_name}_pct'] = asos_region_df['materials'].apply(
            lambda t, ts=target_str: extract_pct(t, [ts])
        )
        sub = asos_region_df.dropna(subset=[f'{safe_name}_pct'])
        if not sub.empty:
            print(f"ASOS GB — {target_str}: n={len(sub)}, mean={sub[f'{safe_name}_pct'].mean():.2f}%")
else:
    print("asos_clean.csv not found — skipping ASOS region breakdown.")

# ---- Shein (India) ----
print("\n--- Shein India ---")
if os.path.isfile('shein_clean.csv'):
    shein_region_df = pd.read_csv('shein_clean.csv')
    shein_region_df['region'] = 'India'
    for target_str, safe_name in TARGETS.items():
        shein_region_df[f'{safe_name}_pct'] = shein_region_df['materials'].apply(
            lambda t, ts=target_str: extract_pct(t, [ts])
        )
        sub = shein_region_df.dropna(subset=[f'{safe_name}_pct'])
        if not sub.empty:
            print(f"Shein India — {target_str}: n={len(sub)}, mean={sub[f'{safe_name}_pct'].mean():.2f}%")
else:
    print("shein_clean.csv not found — skipping Shein region breakdown.")

print("\n\n" + "=" * 90)
print("ALL SECTIONS COMPLETE. Check the working directory for saved CSVs and PNGs.")
print("=" * 90)

plt.show()
