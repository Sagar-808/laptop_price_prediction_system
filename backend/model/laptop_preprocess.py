import re
import pandas as pd
import numpy as np


def _extract_resolution(res_str: str):
    if not isinstance(res_str, str):
        return 0, 0, 0, 0
    s = res_str
    ips = 1 if 'IPS' in s.upper() else 0
    touch = 1 if 'TOUCH' in s.upper() else 0
    # Find WxH
    m = re.search(r"(\d+)x(\d+)", s)
    if m:
        w = int(m.group(1))
        h = int(m.group(2))
    else:
        w, h = 0, 0
    return w, h, ips, touch


def _extract_cpu_brand_and_freq(cpu_str: str):
    if not isinstance(cpu_str, str):
        return 'Other', 0.0
    s = cpu_str.strip()
    brand = 'Other'
    up = s.upper()
    if 'INTEL' in up:
        brand = 'Intel'
    if 'AMD' in up:
        brand = 'AMD' if brand == 'Other' else brand
    if 'APPLE' in up or 'M1' in up or 'M2' in up:
        brand = 'Apple'
    # Frequency
    freq = 0.0
    fm = re.search(r"([0-9]+\.?[0-9]*)\s*GHZ", up)
    if fm:
        try:
            freq = float(fm.group(1))
        except Exception:
            freq = 0.0
    return brand, freq


def _parse_ram(ram_str: str):
    if isinstance(ram_str, (int, float)):
        return float(ram_str)
    if not isinstance(ram_str, str):
        return 0.0
    m = re.search(r"(\d+)", ram_str)
    return float(m.group(1)) if m else 0.0


def _parse_memory(mem_str: str):
    ssd = hdd = flash = hybrid = 0.0
    if not isinstance(mem_str, str):
        return ssd, hdd, flash, hybrid
    # split on '+' if multiple drives
    parts = [p.strip() for p in mem_str.split('+')]
    for p in parts:
        up = p.upper()
        size_match = re.search(r"(\d+\.?\d*)\s*TB|GB", up)
        # Build size in GB
        size_gb = 0.0
        # Handle TB first
        m_tb = re.search(r"(\d+\.?\d*)\s*TB", up)
        if m_tb:
            try:
                size_gb = float(m_tb.group(1)) * 1024.0
            except Exception:
                size_gb = 0.0
        else:
            m_gb = re.search(r"(\d+\.?\d*)\s*GB", up)
            if m_gb:
                try:
                    size_gb = float(m_gb.group(1))
                except Exception:
                    size_gb = 0.0
        if 'SSD' in up:
            ssd += size_gb
        elif 'HDD' in up:
            hdd += size_gb
        elif 'FLASH' in up:
            flash += size_gb
        elif 'HYBRID' in up:
            hybrid += size_gb
    return ssd, hdd, flash, hybrid


def _gpu_brand(gpu_str: str):
    if not isinstance(gpu_str, str):
        return 'Other'
    up = gpu_str.upper()
    if 'NVIDIA' in up:
        return 'Nvidia'
    if 'AMD' in up:
        return 'AMD'
    if 'INTEL' in up:
        return 'Intel'
    return 'Other'


def _weight_kg(w_str: str):
    if isinstance(w_str, (int, float)):
        return float(w_str)
    if not isinstance(w_str, str):
        return 0.0
    m = re.search(r"([0-9]+\.?[0-9]*)\s*KG", w_str.upper())
    return float(m.group(1)) if m else 0.0


def parse_laptop_dataframe(df: pd.DataFrame, drop_target: bool = True):
    data = df.copy()
    # Base columns expected similar to dataset
    # Derive engineered features
    res_w, res_h, ips, touch = zip(*data['ScreenResolution'].apply(_extract_resolution))
    data['ResWidth'] = res_w
    data['ResHeight'] = res_h
    data['IPS'] = ips
    data['Touchscreen'] = touch

    data['CpuBrand'], data['CpuGHz'] = zip(*data['Cpu'].apply(_extract_cpu_brand_and_freq))
    data['RamGB'] = data['Ram'].apply(_parse_ram)
    ssd, hdd, flash, hybrid = zip(*data['Memory'].apply(_parse_memory))
    data['SSD_GB'] = ssd
    data['HDD_GB'] = hdd
    data['Flash_GB'] = flash
    data['Hybrid_GB'] = hybrid
    data['GpuBrand'] = data['Gpu'].apply(_gpu_brand)
    data['WeightKG'] = data['Weight'].apply(_weight_kg)

    # Pixels Per Inch (PPI)
    with np.errstate(divide='ignore', invalid='ignore'):
        diag_pixels = np.sqrt(data['ResWidth']**2 + data['ResHeight']**2)
        data['PPI'] = diag_pixels / data['Inches']
        data['PPI'] = data['PPI'].replace([np.inf, -np.inf], 0).fillna(0)

    # Choose features
    cat_cols = ['Company', 'TypeName', 'CpuBrand', 'GpuBrand', 'OpSys']
    num_cols = ['Inches', 'ResWidth', 'ResHeight', 'IPS', 'Touchscreen', 'CpuGHz', 'RamGB',
                'SSD_GB', 'HDD_GB', 'Flash_GB', 'Hybrid_GB', 'WeightKG', 'PPI']

    feature_cols = cat_cols + num_cols

    X = data[feature_cols]
    y = data['Price'] if 'Price' in data.columns else None

    return X, y, cat_cols, num_cols


def build_input_dataframe(form_or_dict: dict) -> pd.DataFrame:
    """Build a raw DataFrame from form input matching original dataset columns."""
    raw = {
        'Company': form_or_dict.get('Company', ''),
        'TypeName': form_or_dict.get('TypeName', ''),
        'Inches': float(form_or_dict.get('Inches', 0) or 0),
        'ScreenResolution': form_or_dict.get('ScreenResolution', ''),
        'Cpu': form_or_dict.get('Cpu', ''),
        'Ram': form_or_dict.get('Ram', ''),
        'Memory': form_or_dict.get('Memory', ''),
        'Gpu': form_or_dict.get('Gpu', ''),
        'OpSys': form_or_dict.get('OpSys', ''),
        'Weight': form_or_dict.get('Weight', ''),
    }
    return pd.DataFrame([raw])
