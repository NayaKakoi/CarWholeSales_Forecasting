import os
import glob
import re
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

INPUT_DIR = "."
OUTPUT_FILE = "Data_Bersih_Gaikindo_ALL_YEARS.csv"

MONTH_MAP = {
    'jan': '01', 'feb': '02', 'mar': '03', 'mrt': '03',
    'apr': '04', 'may': '05', 'mei': '05', 'jun': '06',
    'jul': '07', 'aug': '08', 'agu': '08', 'agt': '08',
    'sep': '09', 'oct': '10', 'okt': '10', 'nov': '11', 'nop': '11',
    'dec': '12', 'des': '12',
}

MONTH_ORDER = ['jan', 'feb', 'mar', 'apr', 'may', 'jun',
               'jul', 'aug', 'sep', 'oct', 'nov', 'dec']

MONTH_ALIASES = {
    'mrt': 'mar', 'mei': 'may', 'agu': 'aug', 'agt': 'aug',
    'okt': 'oct', 'nop': 'nov', 'des': 'dec',
}

KNOWN_BRANDS = [
    'toyota', 'daihatsu', 'honda', 'mitsubishi', 'suzuki', 'nissan',
    'wuling', 'hyundai', 'mazda', 'isuzu', 'hino', 'kia', 'datsun',
    'bmw', 'mercedes', 'lexus', 'chevrolet', 'vw', 'volkswagen',
    'audi', 'peugeot', 'renault', 'dfsk', 'fiat', 'mini', 'jaguar',
    'ford', 'proton', 'chrysler', 'jeep', 'land rover', 'mg', 'tata', 'chery',
    'byd', 'nio', 'geely', 'changan', 'great wall', 'gwm', 'omoda',
    'neta', 'zeekr', 'jetour', 'tank', 'haval', 'foton', 'morris', 'seres', 'baic'
]

JUNK_PATTERNS = re.compile(
    r'\b(total|jumlah|cumul|subtotal|segment\s+share|market\s+share'
    r'|pct|percent|%|keterangan|note|sumber|source|rank|no\.|nomor|forall|for\s*all|gvw|ton)\b',
    re.IGNORECASE
)

def extract_year_from_filename(filename):
    match = re.search(r'(20\d{2})', str(filename))
    return match.group(1) if match else None

def clean_number(val):
    if pd.isna(val):
        return 0
    if isinstance(val, (int, float)):
        return max(0, int(round(val)))
    
    val_str = str(val).strip().lower()
    if val_str in ['-', '', 'na', 'nan', 'null', 'none', '#n/a', '#ref!', '#value!']:
        return 0
    
    val_str = re.sub(r'[^\d.,]', '', val_str)
    if not val_str:
        return 0
        
    if '.' in val_str and ',' in val_str:
        if val_str.rfind('.') > val_str.rfind(','):
            val_str = val_str.replace(',', '')
        else:
            val_str = val_str.replace('.', '').replace(',', '.')
    elif '.' in val_str:
        parts = val_str.split('.')
        if len(parts[-1]) > 2:
            val_str = val_str.replace('.', '')
    elif ',' in val_str:
        parts = val_str.split(',')
        if len(parts[-1]) > 2:
            val_str = val_str.replace(',', '')
        else:
            val_str = val_str.replace(',', '.')
    try:
        return max(0, int(float(val_str)))
    except Exception:
        return 0

def clean_brand(brand_name):
    b_upper = ' '.join(str(brand_name).split()).upper()
    if 'TOYOTA' in b_upper: return 'Toyota'
    if 'HONDA' in b_upper: return 'Honda'
    if 'DAIHATSU' in b_upper: return 'Daihatsu'
    if 'MITSUBISHI' in b_upper: return 'Mitsubishi'
    if 'SUZUKI' in b_upper: return 'Suzuki'
    if 'WULING' in b_upper or 'W ULING' in b_upper: return 'Wuling'
    if 'NISSAN' in b_upper: return 'Nissan'
    if 'HYUNDAI' in b_upper: return 'Hyundai'
    if 'MAZDA' in b_upper: return 'Mazda'
    if 'ISUZU' in b_upper: return 'Isuzu'
    if 'HINO' in b_upper: return 'Hino'
    if 'KIA' in b_upper: return 'KIA'
    if 'MERCEDES' in b_upper or 'BENZ' in b_upper: return 'Mercedes-Benz'
    if 'BMW' in b_upper: return 'BMW'
    if 'BYD' in b_upper: return 'BYD'
    if 'CHERY' in b_upper: return 'Chery'
    if 'GEELY' in b_upper: return 'Geely'
    if 'GREAT WALL' in b_upper or 'GWM' in b_upper: return 'GWM'
    if 'HAVAL' in b_upper: return 'Haval'
    if 'NETA' in b_upper: return 'Neta'
    if 'OMODA' in b_upper: return 'Omoda'
    if 'JETOUR' in b_upper: return 'Jetour'
    if 'MORRIS' in b_upper or ' MG ' in f" {b_upper} " or b_upper == 'MG': return 'Morris Garage'
    if 'DFSK' in b_upper: return 'DFSK'
    if 'VOLKSW' in b_upper or ' VW ' in f" {b_upper} " or b_upper == 'VW': return 'Volkswagen'
    if 'RENAULT' in b_upper: return 'Renault'
    if 'LAND ROVER' in b_upper: return 'Land Rover'
    if 'CHEVROLET' in b_upper: return 'Chevrolet'
    if 'PEUGEOT' in b_upper: return 'Peugeot'
    if 'AUDI' in b_upper: return 'Audi'
    if 'DATSUN' in b_upper: return 'Datsun'
    if 'FORD' in b_upper: return 'Ford'
    if 'ALFA ROMEO' in b_upper: return 'Alfa Romeo'

    return b_upper.title().replace(' Motors', '').replace(' Motor', '').strip()

def classify_fuel_type(model_name):
    m = str(model_name).upper()
    bev_keywords = ['BEV', 'AIR EV', 'BINGUO', 'ATTO', 'SEAL ', 'DOLPHIN', 'IONIQ 6', 'IONIQ6', 'LEAF', 'ZS EV', 'MG4', 'NETA', 'ZEEKR', 'NEXO', 'EV6', 'GWM ORA', 'TANK EV', 'HAVAL JOLION EV', ' EV ', 'ELECTRIC']
    hev_keywords = ['HEV', 'HYBRID', 'PHEV', 'CROSS HYBRID', 'ZENIX HEV', 'YARIS CROSS HEV', 'INNOVA ZENIX', 'ALPHARD HEV', 'VELLFIRE', 'IONIQ 5', 'IONIQ5', 'TUCSON HEV', 'SANTA FE HEV', 'CRETA HEV']
    diesel_keywords = ['DIESEL', 'DEXCEL', 'PAJERO SPORT', 'FORTUNER 2.4', 'FORTUNER 2.8', 'HILUX', 'TRITON', 'PANTHER', 'TRAGA', 'TRUCK', 'BUS', 'PICK UP', 'PICKUP', 'DUMP', 'RANGER', 'EVEREST', 'TERRA', 'NAVARA', 'D-MAX', 'MU-X', 'ISUZU ELF', 'HINO', 'FUSO']

    if any(k in m for k in bev_keywords): return 'Elektrik (BEV)'
    if any(k in m for k in hev_keywords): return 'Hybrid (HEV)'
    if any(k in m for k in diesel_keywords): return 'Diesel'
    return 'Bensin (ICE)'

def _normalize_month_key(cell_str):
    if pd.isna(cell_str): return None
    s = str(cell_str).strip().lower()
    prefix = s[:3]
    prefix = MONTH_ALIASES.get(prefix, prefix)
    return prefix if prefix in MONTH_ORDER else None

def _find_header_row(df_raw, search_limit=25):
    best_row, best_count = -1, 1 
    for i in range(min(search_limit, len(df_raw))):
        row_vals = df_raw.iloc[i]
        found = set()
        for v in row_vals:
            mk = _normalize_month_key(v)
            if mk: found.add(mk)
        if len(found) > best_count:
            best_count = len(found)
            best_row = i
    return best_row

def _extract_month_columns(header_row_vals):
    month_col_map = {}
    for col_idx, val in enumerate(header_row_vals):
        mk = _normalize_month_key(val)
        if mk:
            month_col_map.setdefault(mk, []).append(col_idx)
    return month_col_map

def _find_brand_col(df_raw, start_row, end_col, sample_rows=40):
    best_col, best_score = 0, -1
    for col in range(end_col):
        # Memastikan sampel diproses dengan aman tanpa as_type yang bisa gagal
        sample = df_raw.iloc[start_row:start_row + sample_rows, col]
        score = sample.apply(
            lambda x: sum(1 for kb in KNOWN_BRANDS if kb in str(x).lower()) 
            if pd.notna(x) else 0
        ).sum()
        
        if score > best_score:
            best_score = score
            best_col = col
    return best_col

def _is_junk_brand(brand_val):
    val_nospace = str(brand_val).lower().replace(' ', '').replace('\n', '')
    junk_keywords = ['forall', 'gvw', 'ton', 'total', 'jumlah', 'segment']
    return any(junk in val_nospace for junk in junk_keywords) or val_nospace.startswith('cc')

def process_gaikindo_excel(uploaded_file):
    filename = uploaded_file.name if hasattr(uploaded_file, 'name') else os.path.basename(str(uploaded_file))
    year = extract_year_from_filename(filename)
    if not year:
        return None, "Nama file harus mengandung tahun (misal: Gaikindo_2025.xlsx)."

    all_rows = []
    try:
        sheet_dict = pd.read_excel(uploaded_file, engine='openpyxl', header=None, sheet_name=None)
    except Exception as e:
        return None, f"Gagal membuka file Excel: {str(e)}"

    mem = {'brand_col': None, 'model_col': None, 'month_cols': None}

    for sheet_name, df_raw in sheet_dict.items():
        if df_raw is None or df_raw.empty:
            continue

        header_idx = _find_header_row(df_raw)
        if header_idx != -1:
            raw_header = df_raw.iloc[header_idx]
            month_col_map = _extract_month_columns(raw_header)
            if not month_col_map: continue

            all_month_cols_flat = sorted(set(c for cols in month_col_map.values() for c in cols))
            first_month_col = all_month_cols_flat[0]

            brand_col = _find_brand_col(df_raw, header_idx + 1, first_month_col)
            model_col = brand_col + 1 if (brand_col + 1) < first_month_col else brand_col

            mem.update({'brand_col': brand_col, 'model_col': model_col, 'month_cols': month_col_map})
            data_start = header_idx + 1
        else:
            if not mem['month_cols']: continue
            brand_col, model_col, month_col_map = mem['brand_col'], mem['model_col'], mem['month_cols']
            data_start = 0

        ordered_month_cols = [(mk, month_col_map[mk][0]) for mk in MONTH_ORDER if mk in month_col_map]
        if not ordered_month_cols: continue

        max_needed_col = max(brand_col, model_col, max(c for _, c in ordered_month_cols))
        current_brand = "Unknown"

        for row_idx in range(data_start, len(df_raw)):
            row = df_raw.iloc[row_idx]
            if max_needed_col >= len(row): continue

            # Proses Brand
            raw_brand = row.iloc[brand_col]
            brand_val = str(raw_brand).strip() if pd.notna(raw_brand) else ""
            
            if brand_val.lower() not in ('nan', 'none', '', 'nat'):
                if not re.match(r'^[\d\s,.\-]+$', brand_val) and not _is_junk_brand(brand_val):
                    current_brand = clean_brand(brand_val)

            if current_brand == "Unknown": continue

            # Proses Model
            raw_model = row.iloc[model_col]
            model_val = str(raw_model).strip() if pd.notna(raw_model) else ""
            
            if model_val.lower() in ('nan', 'none', '', 'nat') or re.match(r'^[\d\s,.\-]+$', model_val):
                continue
                
            combined_check = f"{current_brand} {model_val}".lower()
            if JUNK_PATTERNS.search(combined_check):
                continue

            fuel_type = classify_fuel_type(model_val)

            # Ekstrak Nilai Bulanan
            for month_key, col_idx in ordered_month_cols:
                val = clean_number(row.iloc[col_idx])
                if val >= 0:
                    date_str = f"{year}-{MONTH_MAP[month_key]}-01"
                    all_rows.append([date_str, current_brand, model_val, fuel_type, val])

    if not all_rows:
        return None, "Gagal mengekstrak data dari file ini. Pastikan format tabel memiliki header (Jan-Des)."

    df_master = pd.DataFrame(all_rows, columns=['Tanggal', 'Brand', 'Model', 'Segment_Fuel', 'Aktual'])
    df_master['Tanggal'] = pd.to_datetime(df_master['Tanggal'], errors='coerce')
    df_master.dropna(subset=['Tanggal'], inplace=True) # Hapus baris jika tanggal gagal di-parsing

    df_agg = df_master.groupby(['Tanggal', 'Brand', 'Model', 'Segment_Fuel'], as_index=False)['Aktual'].sum()
    return df_agg, None

def main():
    logging.info("Memulai Ekstraksi GAIKINDO (Mode Batch)...")
    excel_files = glob.glob(os.path.join(INPUT_DIR, "*GAIKINDO*.xlsx"))

    if not excel_files:
        logging.error("Tidak ada file Excel GAIKINDO ditemukan.")
        return

    master_frames = []
    for fp in sorted(excel_files):
        df_ext, err = process_gaikindo_excel(fp)
        if df_ext is not None and not df_ext.empty:
            master_frames.append(df_ext)
        else:
            logging.error(f"  ✘ {fp}: {err}")

    if not master_frames:
        logging.error("Tidak ada data berhasil diekstrak sama sekali.")
        return

    df_final = pd.concat(master_frames, ignore_index=True)
    df_final_agg = df_final.groupby(['Tanggal', 'Brand', 'Model', 'Segment_Fuel'], as_index=False)['Aktual'].sum()
    df_final_agg.to_csv(OUTPUT_FILE, index=False)
    
    detected_brands = sorted(df_final_agg['Brand'].unique())
    logging.info("=" * 55)
    logging.info(f"✅ BERHASIL! Diekspor ke: {OUTPUT_FILE}")
    logging.info(f"📊 Total baris: {len(df_final_agg)}")
    logging.info(f"🏷️ Merek Terdeteksi ({len(detected_brands)} Brand): {detected_brands}")
    logging.info("=" * 55)

if __name__ == "__main__":
    main()