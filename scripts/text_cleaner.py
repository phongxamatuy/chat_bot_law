"""
Text Cleaner - Xử lý khoảng trắng thừa và xóa header trang trong file TXT pháp luật.

Thuật toán:
1. Xóa các dòng header trang (CÔNG BÁO/Số..., số trang đầu/cuối dòng).
2. Fix khoảng trắng thừa do lỗi PDF-to-text (ví dụ: "th ương" → "thương").
3. Chuẩn hóa khoảng trắng và dòng trống.
"""

import re
from pathlib import Path


class TextCleaner:

    def __init__(self, input_dir: str, output_dir: str):
        """
        :param input_dir: Thư mục chứa file TXT gốc (data/processed/cleaned).
        :param output_dir: Thư mục lưu file đã xử lý (data/processed/final).
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ==================================================
    # 1. XÓA HEADER TRANG CÔNG BÁO
    # ==================================================
    def _remove_page_headers(self, text: str) -> str:
        """
        Xóa các dòng header trang kiểu:
        - "CÔNG BÁO/Số 363 + 364/Ngày 01-3-2024 9"
        - "4 CÔNG BÁO/Số 363 + 364/Ngày 01-3-2024"
        - "42 CÔNG BÁO/Số 1263 + 1264/Ngày 31-12-2015"
        """
        # Pattern: (số tùy chọn) + CÔNG BÁO/Số ... + (số tùy chọn)
        pattern = r'^\s*\d*\s*CÔNG BÁO/Số.*$'
        text = re.sub(pattern, '', text, flags=re.MULTILINE)
        return text

    # ==================================================
    # 2. FIX KHOẢNG TRẮNG THỪA (CORE ALGORITHM)
    # ==================================================
    def _fix_broken_whitespace(self, text: str) -> str:
        """
        Sửa lỗi khoảng trắng thừa do PDF-to-text gây ra.

        Pattern lỗi: Khoảng trắng xuất hiện giữa phụ âm/nguyên âm và phần mang dấu.
        Ví dụ: "th ương" → "thương", "ng ười" → "người", "b ảo" → "bảo"

        Thuật toán:
        - Tìm các cặp "fragment ngắn (1-3 ký tự) + khoảng trắng + fragment ngắn có dấu tiếng Việt"
        - Nếu fragment trước KHÔNG phải là từ/ký hiệu có nghĩa đứng độc lập → gộp lại.
        """

        # Các ký tự nguyên âm có dấu tiếng Việt
        # Nếu phần sau khoảng trắng bắt đầu bằng ký tự có dấu → khả năng cao là bị tách nhầm
        diacritical_vowels = (
            'àáảãạăắằẳẵặâấầẩẫậ'
            'èéẻẽẹêếềểễệ'
            'ìíỉĩị'
            'òóỏõọôốồổỗộơớờởỡợ'
            'ùúủũụưứừửữự'
            'ỳýỷỹỵ'
            'ÀÁẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬ'
            'ÈÉẺẼẸÊẾỀỂỄỆ'
            'ÌÍỈĨỊ'
            'ÒÓỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢ'
            'ÙÚỦŨỤƯỨỪỬỮỰ'
            'ỲÝỶỸỴ'
        )

        # Các từ/ký hiệu 1 ký tự có nghĩa đứng độc lập (không được gộp)
        standalone_chars = {
            'a', 'b', 'c', 'd', 'e', 'g', 'h', 'i', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'v', 'x',
            'A', 'B', 'C', 'D', 'E', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'V', 'X',
        }

        # Pattern: 1-3 ký tự chữ cái + khoảng trắng + ký tự có dấu tiếng Việt + phần còn lại của âm tiết
        diacritical_pattern = f'[{re.escape(diacritical_vowels)}]'

        # Danh sách nguyên âm tiếng Việt (bao gồm không dấu) để phân biệt âm tiết
        all_vowels = set('aeiouyàáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵ'
                         'AEIOUYÀÁẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÈÉẺẼẸÊẾỀỂỄỆÌÍỈĨỊÒÓỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÙÚỦŨỤƯỨỪỬỮỰỲÝỶỸỴ')

        # Tất cả ký tự chữ cái tiếng Việt (dùng cho boundary)
        vn_letter = r'[a-zA-ZĐđàáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵ]'

        pattern = re.compile(
            r'(?<!' + vn_letter + r')'          # Trước prefix KHÔNG phải chữ cái VN (đầu từ thực sự)
            r'([a-zA-ZĐđ]{1,3})'               # Nhóm 1: Fragment đầu (1-3 phụ âm đầu)
            r' '                                # Khoảng trắng bị thừa
            r'(' + diacritical_pattern +         # Nhóm 2: Fragment sau bắt đầu bằng ký tự có dấu
            r'[a-zA-ZàáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđĐ]*)'
            r'(?!' + vn_letter + r')'           # Sau suffix KHÔNG phải chữ cái VN (cuối từ thực sự)
        )

        def _replace_match(match):
            prefix = match.group(1)
            suffix = match.group(2)
            merged = prefix + suffix

            # Kiểm tra ký tự ngay trước match
            start = match.start()
            if start > 0:
                char_before = match.string[start - 1]
                # Nếu đứng sau dấu đóng ngoặc/dấu câu → không gộp (ví dụ: "a) Ủy ban")
                if char_before in ').:;,':
                    return match.group(0)

            # Danh sách phụ âm đầu tiếng Việt hợp lệ (whitelist)
            # Bao gồm cả tổ hợp phụ âm + bán nguyên âm (y/i/u/o) vẫn đóng vai trò phần đầu âm tiết
            valid_prefixes = {
                # 1 ký tự (phụ âm đơn)
                'b', 'c', 'd', 'g', 'h', 'k', 'l', 'm', 'n', 'p', 'q', 'r', 's', 't', 'v', 'x',
                'B', 'C', 'D', 'G', 'H', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'X',
                'đ', 'Đ',
                # 2 ký tự (tổ hợp phụ âm đầu)
                'ch', 'gh', 'gi', 'kh', 'ng', 'nh', 'ph', 'qu', 'th', 'tr',
                'Ch', 'Gh', 'Gi', 'Kh', 'Ng', 'Nh', 'Ph', 'Qu', 'Th', 'Tr',
                'CH', 'GH', 'GI', 'KH', 'NG', 'NH', 'PH', 'QU', 'TH', 'TR',
                # 3 ký tự (phụ âm đầu dài)
                'ngh', 'Ngh', 'NGH',
                # 2-3 ký tự: phụ âm đầu + bán nguyên âm (u/o/y/i) trước vần chính
                # Ví dụ: "quy ền" → "quyền", "nhi ệm" → "nhiệm"
                'hu', 'ho', 'hi',
                'Hu', 'Ho', 'Hi',
                'HU', 'HO', 'HI',
                'lu', 'lo', 'li', 'ly',
                'Lu', 'Lo', 'Li', 'Ly',
                'LU', 'LO', 'LI', 'LY',
                'tu', 'to', 'ti', 'ty',
                'Tu', 'To', 'Ti', 'Ty',
                'TU', 'TO', 'TI', 'TY',
                'su', 'so', 'si', 'sy',
                'Su', 'So', 'Si', 'Sy',
                'nu', 'no', 'ni', 'ny',
                'Nu', 'No', 'Ni', 'Ny',
                'mu', 'mo', 'mi',
                'Mu', 'Mo', 'Mi',
                'bu', 'bo', 'bi',
                'Bu', 'Bo', 'Bi',
                'cu', 'co', 'ci',
                'Cu', 'Co', 'Ci',
                'du', 'do', 'di',
                'Du', 'Do', 'Di',
                'gu', 'go', 'gi',
                'Gu', 'Go', 'Gi',
                'vu', 'vo', 'vi',
                'Vu', 'Vo', 'Vi',
                'xu', 'xo', 'xi',
                'Xu', 'Xo', 'Xi',
                'ru', 'ro', 'ri',
                'Ru', 'Ro', 'Ri',
                'ky', 'ki', 'ku',
                'Ky', 'Ki', 'Ku',
                'py', 'pi', 'pu',
                'Py', 'Pi', 'Pu',
                # 3 ký tự: phụ âm kép + bán nguyên âm
                'thi', 'thu', 'tho', 'thy',
                'Thi', 'Thu', 'Tho', 'Thy',
                'chi', 'chu', 'cho',
                'Chi', 'Chu', 'Cho',
                'phi', 'phu', 'pho',
                'Phi', 'Phu', 'Pho',
                'khi', 'khu', 'kho',
                'Khi', 'Khu', 'Kho',
                'nhi', 'nhu', 'nho',
                'Nhi', 'Nhu', 'Nho',
                'ngi', 'ngu', 'ngo',
                'Ngi', 'Ngu', 'Ngo',
                'ghi', 'ghu',
                'Ghi', 'Ghu',
                'tri', 'tru', 'tro',
                'Tri', 'Tru', 'Tro',
                'qui', 'quy',
                'Qui', 'Quy',
            }

            # Nếu prefix nằm trong whitelist phụ âm đầu → đây là từ bị tách → gộp lại
            if prefix in valid_prefixes:
                return merged

            # Nếu prefix KHÔNG nằm trong whitelist → prefix là âm tiết đầy đủ → giữ nguyên
            return match.group(0)

        # Lặp nhiều lần vì sau khi gộp lần 1, có thể xuất hiện pattern mới
        # Ví dụ: "t ổ ch ức" → lần 1: "tổ ch ức" → lần 2: "tổ chức"
        max_iterations = 5
        for _ in range(max_iterations):
            new_text = pattern.sub(_replace_match, text)
            if new_text == text:
                break
            text = new_text

        return text

    # ==================================================
    # 3. FIX SỐ BỊ TÁCH (Ví dụ: "202 5" → "2025")
    # ==================================================
    def _fix_broken_numbers(self, text: str) -> str:
        """Fix số bị tách: "202 5" → "2025", "201 5" → "2015"."""
        # Pattern: số + khoảng trắng + 1-2 chữ số (ở vị trí cuối số)
        text = re.sub(r'(\d) (\d{1,2})(?=\D|$)', r'\1\2', text)
        return text

    # ==================================================
    # 4. CHUẨN HÓA KHOẢNG TRẮNG VÀ DÒNG TRỐNG
    # ==================================================
    def _normalize_whitespace(self, text: str) -> str:
        """Chuẩn hóa: xóa khoảng trắng thừa trên mỗi dòng, gộp dòng trống liên tiếp."""
        # Xóa khoảng trắng/tab thừa trên mỗi dòng (giữ nguyên xuống dòng)
        text = re.sub(r'[ \t]+', ' ', text)
        # Xóa khoảng trắng đầu/cuối mỗi dòng
        text = re.sub(r'^ +| +$', '', text, flags=re.MULTILINE)
        # Gộp 3+ dòng trống thành 1 dòng trống
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()

    # ==================================================
    # 5. HÀM XỬ LÝ CHÍNH
    # ==================================================
    def clean_file(self, file_path: Path) -> str:
        """Xử lý 1 file TXT và trả về text đã sạch."""
        text = file_path.read_text(encoding='utf-8')

        # Bỏ qua file rỗng hoặc quá ngắn
        if len(text.strip()) < 50:
            return text

        text = self._remove_page_headers(text)
        text = self._fix_broken_whitespace(text)
        text = self._fix_broken_numbers(text)
        text = self._normalize_whitespace(text)
        return text

    def clean_all(self):
        """Quét và xử lý toàn bộ file TXT trong thư mục input."""
        txt_files = sorted(self.input_dir.glob("*.txt"))
        print(f"[*] Tìm thấy {len(txt_files)} file TXT trong {self.input_dir}")

        stats = {"processed": 0, "skipped": 0}

        for txt_file in txt_files:
            file_size = txt_file.stat().st_size
            if file_size < 50:
                print(f"  [SKIP] {txt_file.name} (file rỗng/quá ngắn: {file_size} bytes)")
                stats["skipped"] += 1
                continue

            print(f"  [OK] Đang xử lý: {txt_file.name} ({file_size:,} bytes)")
            cleaned = self.clean_file(txt_file)

            output_file = self.output_dir / txt_file.name
            output_file.write_text(cleaned, encoding='utf-8')
            stats["processed"] += 1

        print(f"\n[*] Hoàn tất! Đã xử lý: {stats['processed']}, Bỏ qua: {stats['skipped']}")
        print(f"[*] File đã lưu tại: {self.output_dir}")


if __name__ == "__main__":
    # Chạy trực tiếp để test
    project_root = Path(__file__).parent.parent
    cleaner = TextCleaner(
        input_dir=str(project_root / "data" / "processed" / "cleaned"),
        output_dir=str(project_root / "data" / "processed" / "final")
    )
    cleaner.clean_all()
