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
    'jan': '01',
    'feb': '02',
    'mar': '03', 'mrt': '03',
    'apr': '04',
    'may': '05', 'mei': '05',
    'jun': '06',
    'jul': '07',
    'aug': '08', 'agu': '08', 'agt': '08',
    'sep': '09',
    'oct': '10', 'okt': '10',
    'nov': '11', 'nop': '11',
    'dec': '12', 'des': '12',
}

MONTH_ORDER = ['jan', 'feb', 'mar', 'apr', 'may', 'jun',
               'jul', 'aug', 'sep', 'oct', 'nov', 'dec']

MONTH_ALIASES = {
    'mrt': 'mar',
    'mei': 'may',
    'agu': 'aug', 'agt': 'aug',
    'okt': 'oct',
    'nop': 'nov',
    'des': 'dec',
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


def extract_year_from_filename(filename):
    match = re.search(r'(20\d{2})', filename)
    return match.group(1) if match else None


def clean_number(val):
    """Konversi berbagai format angka ke integer. Return 0 jika tidak valid."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return 0
    if isinstance(val, (int, float)):
        v = int(round(val))
        return v if v >= 0 else 0
    val = str(val).strip()
    if val.lower() in ['-', '', 'na', 'nan', 'null', 'none', '#n/a', '#ref!', '#value!']:
        return 0
    val = re.sub(r'[^\d.,]', '', val)
    if not val:
        return 0
    if '.' in val and ',' in val:
        if val.rfind('.') > val.rfind(','):
            val = val.replace(',', '')
        else:
            val = val.replace('.', '').replace(',', '.')
    elif '.' in val:
        parts = val.split('.')
        if len(parts[-1]) > 2:
            val = val.replace('.', '')
    elif ',' in val:
        parts = val.split(',')
        if len(parts[-1]) > 2:
            val = val.replace(',', '')
        else:
            val = val.replace(',', '.')
    try:
        return max(0, int(float(val)))
    except Exception:
        return 0


def clean_brand(brand_name):
    b_upper = ' '.join(str(brand_name).split()).upper()

    if 'TOYOTA' in b_upper:      return 'Toyota'
    if 'HONDA' in b_upper:       return 'Honda'
    if 'DAIHATSU' in b_upper:    return 'Daihatsu'
    if 'MITSUBISHI' in b_upper:  return 'Mitsubishi'
    if 'SUZUKI' in b_upper:      return 'Suzuki'
    if 'WULING' in b_upper or 'W ULING' in b_upper: return 'Wuling'
    if 'NISSAN' in b_upper:      return 'Nissan'
    if 'HYUNDAI' in b_upper:     return 'Hyundai'
    if 'MAZDA' in b_upper:       return 'Mazda'
    if 'ISUZU' in b_upper:       return 'Isuzu'
    if 'HINO' in b_upper:        return 'Hino'
    if 'KIA' in b_upper:         return 'KIA'
    if 'MERCEDES' in b_upper or 'BENZ' in b_upper: return 'Mercedes-Benz'
    if 'BMW' in b_upper:         return 'BMW'
    if 'BYD' in b_upper:         return 'BYD'
    if 'CHERY' in b_upper:       return 'Chery'
    if 'GEELY' in b_upper:       return 'Geely'
    if 'GREAT WALL' in b_upper or 'GWM' in b_upper: return 'GWM'
    if 'HAVAL' in b_upper:       return 'Haval'
    if 'NETA' in b_upper:        return 'Neta'
    if 'OMODA' in b_upper:       return 'Omoda'
    if 'JETOUR' in b_upper:      return 'Jetour'
    
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
    """
    Klasifikasi tipe bahan bakar berdasarkan nama model.
    Urutan pengecekan: BEV → HEV/PHEV → Diesel → ICE (default)
    """
    m = str(model_name).upper()

    bev_keywords = [
        'BEV', 'AIR EV', 'BINGUO', 'ATTO', 'SEAL ', 'DOLPHIN', 'IONIQ 6', 'IONIQ6',
        'LEAF', 'ZS EV', 'MG4', 'NETA', 'ZEEKR', 'NEXO', 'EV6', 'GWM ORA',
        'TANK EV', 'HAVAL JOLION EV', ' EV ', 'ELECTRIC',
    ]
    hev_keywords = [
        'HEV', 'HYBRID', 'PHEV', 'CROSS HYBRID', 'ZENIX HEV',
        'YARIS CROSS HEV', 'INNOVA ZENIX', 'ALPHARD HEV', 'VELLFIRE',
        'IONIQ 5', 'IONIQ5', 'TUCSON HEV', 'SANTA FE HEV', 'CRETA HEV',
    ]
    diesel_keywords = [
        'DIESEL', 'DEXCEL', 'PAJERO SPORT', 'FORTUNER 2.4', 'FORTUNER 2.8',
        'HILUX', 'TRITON', 'PANTHER', 'TRAGA', 'TRUCK', 'BUS', 'PICK UP',
        'PICKUP', 'DUMP', 'RANGER', 'EVEREST', 'TERRA', 'NAVARA',
        'D-MAX', 'MU-X', 'ISUZU ELF', 'HINO', 'FUSO',
    ]

    if any(k in m for k in bev_keywords):
        return 'Elektrik (BEV)'
    if any(k in m for k in hev_keywords):
        return 'Hybrid (HEV)'
    if any(k in m for k in diesel_keywords):
        return 'Diesel'
    return 'Bensin (ICE)'


def _normalize_month_key(cell_str):
    s = str(cell_str).strip().lower()
    prefix = s[:3]
    if prefix in MONTH_ALIASES:
        prefix = MONTH_ALIASES[prefix]
    if prefix in MONTH_ORDER:
        return prefix
    return None


def _find_header_row(df_raw, search_limit=25):
    best_row = -1
    best_count = 1  
    for i in range(min(search_limit, len(df_raw))):
        row_vals = df_raw.iloc[i].astype(str)
        found = set()
        for v in row_vals:
            mk = _normalize_month_key(v)
            if mk:
                found.add(mk)
        if len(found) > best_count:
            best_count = len(found)
            best_row = i
    return best_row


def _extract_month_columns(header_row_vals):
    month_col_map = {}
    for col_idx, val in enumerate(header_row_vals):
        mk = _normalize_month_key(str(val))
        if mk:
            month_col_map.setdefault(mk, []).append(col_idx)
    return month_col_map


def _find_brand_col(df_raw, start_row, end_col, sample_rows=40):
    best_col = 0
    best_score = -1
    for col in range(end_col):
        sample = df_raw.iloc[start_row:start_row + sample_rows, col].astype(str).str.lower()
        score = sample.apply(lambda x: any(kb in x for kb in KNOWN_BRANDS)).sum()
        if score > best_score:
            best_score = score
            best_col = col
    return best_col


JUNK_PATTERNS = re.compile(
    r'\b(total|jumlah|cumul|subtotal|grand|segment\s+share|market\s+share'
    r'|pct|percent|%|keterangan|note|sumber|source|rank|no\.|nomor|forall|for\s*all|gvw|ton)\b',
    re.IGNORECASE
)

def _is_junk_row(brand_val, model_val):
    combined = f"{brand_val} {model_val}".replace(' ', '')
    return bool(JUNK_PATTERNS.search(combined))

def _is_junk_brand(brand_val):
    val_nospace = brand_val.lower().replace(' ', '').replace('\n', '')
    junk_keywords = ['forall', 'gvw', 'ton', 'total', 'jumlah', 'segment']
    
    if any(junk in val_nospace for junk in junk_keywords) or val_nospace.startswith('cc'):
        return True
    return False


def process_gaikindo_excel(uploaded_file):
    if hasattr(uploaded_file, 'name'):
        filename = uploaded_file.name
    else:
        filename = os.path.basename(str(uploaded_file))

    year = extract_year_from_filename(filename)
    if not year:
        return None, "Nama file harus mengandung tahun (misal: Gaikindo_2023.xlsx)."

    all_rows = []

    try:
        sheet_dict = pd.read_excel(
            uploaded_file, engine='openpyxl',
            header=None, sheet_name=None
        )
    except Exception as e:
        return None, f"Gagal membuka file Excel: {str(e)}"

    mem = {
        'brand_col':    None,
        'model_col':    None,
        'month_cols':   None,
    }

    for sheet_name, df_raw in sheet_dict.items():
        if df_raw is None or df_raw.empty:
            continue

        logging.info(f"  Memproses sheet: '{sheet_name}' ({len(df_raw)} baris)")

        header_idx = _find_header_row(df_raw)

        if header_idx != -1:
            raw_header = df_raw.iloc[header_idx].astype(str)
            month_col_map = _extract_month_columns(raw_header)

            if not month_col_map:
                logging.warning(f"    Sheet '{sheet_name}': header ditemukan tapi tidak ada kolom bulan. Skip.")
                continue

            all_month_cols_flat = sorted(set(c for cols in month_col_map.values() for c in cols))
            first_month_col = all_month_cols_flat[0]

            brand_col = _find_brand_col(df_raw, header_idx + 1, first_month_col)
            model_col = brand_col + 1 if (brand_col + 1) < first_month_col else brand_col

            mem['brand_col']  = brand_col
            mem['model_col']  = model_col
            mem['month_cols'] = month_col_map

            data_start = header_idx + 1

        else:
            if mem['month_cols'] is None:
                logging.warning(f"    Sheet '{sheet_name}': tidak ada header & belum punya template. Skip.")
                continue
            brand_col  = mem['brand_col']
            model_col  = mem['model_col']
            month_col_map = mem['month_cols']
            data_start = 0

        ordered_month_cols = []
        for mk in MONTH_ORDER:
            if mk in month_col_map:
                ordered_month_cols.append((mk, month_col_map[mk][0]))

        if not ordered_month_cols:
            logging.warning(f"    Sheet '{sheet_name}': tidak ada kolom bulan yang valid. Skip.")
            continue

        max_needed_col = max(brand_col, model_col,
                             max(c for _, c in ordered_month_cols))

        current_brand = "Unknown"

        for row_idx in range(data_start, len(df_raw)):
            row = df_raw.iloc[row_idx]

            if max_needed_col >= len(row):
                continue

            brand_val = str(row.iloc[brand_col]).strip()
            if brand_val.lower() not in ('nan', 'none', '', 'nat'):
                if not re.match(r'^[\d\s,.\-]+$', brand_val):
                    if not _is_junk_brand(brand_val):
                        current_brand = clean_brand(brand_val)

            if current_brand == "Unknown":
                continue

            model_val = str(row.iloc[model_col]).strip()
            if model_val.lower() in ('nan', 'none', '', 'nat'):
                continue

            if _is_junk_row(brand_val, model_val):
                continue

            if re.match(r'^[\d\s,.\-]+$', model_val):
                continue

            fuel_type = classify_fuel_type(model_val)

            any_positive = False
            for month_key, col_idx in ordered_month_cols:
                val = clean_number(row.iloc[col_idx])
                if val > 0:
                    date_str = f"{year}-{MONTH_MAP[month_key]}-01"
                    all_rows.append([date_str, current_brand, model_val, fuel_type, val])
                    any_positive = True

            if any_positive:
                pass 

    logging.info(f"[{year}] Selesai. Total entri: {len(all_rows)}")

    if not all_rows:
        return None, (
            "Gagal mengekstrak data dari file ini. Pastikan format file adalah "
            "laporan GAIKINDO standar dengan header bulan (Jan–Des) yang terdeteksi."
        )

    df_master = pd.DataFrame(all_rows, columns=['Tanggal', 'Brand', 'Model', 'Segment_Fuel', 'Aktual'])
    df_master['Tanggal'] = pd.to_datetime(df_master['Tanggal'])

    df_agg = (
        df_master
        .groupby(['Tanggal', 'Brand', 'Model', 'Segment_Fuel'], as_index=False)['Aktual']
        .sum()
    )

    logging.info(f"[{year}] Setelah agregasi: {len(df_agg)} baris unik Tanggal×Brand×Fuel.")
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
        if df_ext is not None:
            master_frames.append(df_ext)
        else:
            logging.error(f"  ✘ {fp}: {err}")

    if not master_frames:
        logging.error("Tidak ada data berhasil diekstrak.")
        return

    df_final = pd.concat(master_frames, ignore_index=True)
    df_final_agg = (
        df_final
        .groupby(['Tanggal', 'Brand', 'Model', 'Segment_Fuel'], as_index=False)['Aktual']
        .sum()
    )
    df_final_agg.to_csv(OUTPUT_FILE, index=False)
    
    detected_brands = sorted(df_final_agg['Brand'].unique())
    
    logging.info("=" * 55)
    logging.info(f"✅ BERHASIL! Diekspor ke: {OUTPUT_FILE}")
    logging.info(f"📊 Total baris: {len(df_final_agg)}")
    logging.info(f"🏷️ Merek Kendaraan Terdeteksi ({len(detected_brands)} Brand):")
    logging.info(detected_brands)
    logging.info("=" * 55)


if __name__ == "__main__":
    main()