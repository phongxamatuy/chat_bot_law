import os
import re
import json
import shutil
import fitz
from pathlib import Path

RAW_DIR = Path("data/raw")
RENAME_DIR = Path("data/rename")
META_FILE = Path("data/metadata/metadata.json")

RENAME_DIR.mkdir(parents=True, exist_ok=True)
META_FILE.parent.mkdir(parents=True, exist_ok=True)

metadata = []


def clean_filename(name: str) -> str:
    """Loại bỏ ký tự không hợp lệ trong tên file"""
    name = re.sub(r'[\\/:*?"<>|]', "", name)
    name = re.sub(r"\s+", "_", name)
    return name.strip("_")


def parse_number_from_text(text: str) -> str:
    patterns = [
        r"Số[:\s]*([0-9]+/[0-9]{4}/[A-Za-zĐđ0-9\-]+)",
        r"Số[:\s]*([0-9]+)/([0-9]{4})/([A-Za-zĐđ0-9]+)[\s\-]+([A-Za-zĐđ0-9]+)",
        r"Số[:\s]*([0-9]+/[A-Za-zĐđ\-]+)"
    ]
    for i, p in enumerate(patterns):
        m = re.search(p, text, re.IGNORECASE)
        if m:
            if i == 1:
                return f"{m.group(1)}/{m.group(2)}/{m.group(3)}-{m.group(4)}"
            return m.group(1)
    return None

def parse_number_from_filename(filename: str) -> str:
    name = Path(filename).stem
    m = re.search(r"([0-9]+)[_.\-]([0-9]{4})[_.\-]([A-Za-zĐđ0-9_.\-]+)", name, re.IGNORECASE)
    if m:
        so, nam, ky_hieu_raw = m.group(1), m.group(2), m.group(3)
        ky_hieu_parts = re.findall(r"([A-Za-zĐđ0-9]+)", ky_hieu_raw)
        valid_ky_hieu = []
        for p in ky_hieu_parts:
            if p.isupper() or p.upper() == p or (len(p) <= 4 and not p.islower()) or any(c.isdigit() for c in p):
                valid_ky_hieu.append(p.upper())
            else:
                break
        if valid_ky_hieu:
            return f"{so}/{nam}/{'-'.join(valid_ky_hieu)}"
            
    m = re.search(r"([A-Za-zĐđ]+)[\s_.\-]*([0-9]+)[_.\-]([0-9]{4})[_.\-]([A-Za-zĐđ0-9_.\-]+)", name, re.IGNORECASE)
    if m:
        prefix, so, nam, ky_hieu_raw = m.group(1).upper(), m.group(2), m.group(3), m.group(4)
        ky_hieu_parts = re.findall(r"([A-Za-zĐđ0-9]+)", ky_hieu_raw)
        valid_ky_hieu = []
        for p in ky_hieu_parts:
            if p.upper() == p or (len(p) <= 4 and not p.islower()) or any(c.isdigit() for c in p):
                valid_ky_hieu.append(p.upper())
            else:
                break
        if valid_ky_hieu:
            return f"{so}/{nam}/{prefix}-{'-'.join(valid_ky_hieu)}"
        return f"{so}/{nam}/{prefix}"

    if "bo_luat" in name.lower() or "boluat" in name.lower():
        m = re.search(r"bo_?luat_([a-z_]+)_([0-9]{4})", name.lower())
        if m:
            return f"BL-{m.group(1).replace('_', '-')}-{m.group(2)}"
    elif "luat" in name.lower():
        m = re.search(r"luat_([a-z_]+)_([0-9]{4})", name.lower())
        if m:
            return f"Luat-{m.group(1).replace('_', '-')}-{m.group(2)}"
            
    return None

def extract_info(pdf_path: Path):
    """Đọc trang đầu để lấy thông tin hoặc lấy từ tên file"""
    text = ""
    try:
        doc = fitz.open(pdf_path)
        if len(doc) > 0:
            text = doc[0].get_text("text")
        doc.close()
    except Exception:
        pass

    text_single_line = text.replace("\n", " ")
    
    number = parse_number_from_filename(pdf_path.name)
    if not number:
        number = parse_number_from_text(text_single_line)

    doc_type = "VanBan"
    upper = text_single_line.upper()
    if not upper:
        upper = pdf_path.name.upper()

    if "BỘ LUẬT" in upper or "BO_LUAT" in upper:
        doc_type = "BoLuat"
    elif "LUẬT" in upper:
        doc_type = "Luat"
    elif "NGHỊ ĐỊNH" in upper or "ND_CP" in upper or "NĐ-CP" in upper:
        doc_type = "NghiDinh"
    elif "THÔNG TƯ" in upper or "TT_BTC" in upper or "TT-BTC" in upper:
        doc_type = "ThongTu"
    elif "QUYẾT ĐỊNH" in upper or "QĐ" in upper or "QD" in upper:
        doc_type = "QuyetDinh"
    elif "NGHỊ QUYẾT" in upper or "NQ" in upper:
        doc_type = "NghiQuyet"

    title = pdf_path.stem
    if text:
        lines = []
        for line in text.split(". "):
            line = line.strip().replace("\n", " ")
            if len(line) > 20 and "CỘNG HÒA" not in line and "Độc lập" not in line and "Số:" not in line:
                lines.append(line)
        if lines:
            title = lines[0][:150]

    return {
        "title": title,
        "number": number,
        "type": doc_type,
    }


for pdf in RAW_DIR.glob("*.pdf"):
    print(f"Đang xử lý: {pdf.name}")
    info = extract_info(pdf)

    if info is None:
        new_name = clean_filename(pdf.stem) + ".pdf"
    else:
        if info["number"]:
            # Format: Type_Number.pdf e.g. NghiDinh_102_2024_ND_CP.pdf
            safe_number = info["number"].replace("/", "_").replace("-", "_")
            filename = f"{info['type']}_{safe_number}"
            new_name = clean_filename(filename) + ".pdf"
        else:
            new_name = clean_filename(pdf.stem) + ".pdf"

    shutil.copy2(pdf, RENAME_DIR / new_name)

    metadata.append({
        "original_name": pdf.name,
        "file_name": new_name,
        "title": info["title"] if info else "",
        "number": info["number"] if info else "",
        "type": info["type"] if info else "",
    })

with open(META_FILE, "w", encoding="utf8") as f:
    json.dump(
        metadata,
        f,
        ensure_ascii=False,
        indent=4
    )

print("=" * 50)
print("Hoàn thành")
print(f"Tổng số file: {len(metadata)}")
print("=" * 50)