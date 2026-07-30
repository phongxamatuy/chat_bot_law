import os
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from langchain_core.documents import Document
from preprocess import DocumentPreprocessor
import fitz

def extract_text_from_pdf(pdf_path: Path) -> list[Document]:
    docs = []
    
    try:
        pdf_document = fitz.open(pdf_path)
    except Exception as e:
        print(f"[!] Lỗi khi mở {pdf_path}: {e}")
        return []

    print(f"[*] Phân tích {pdf_path.name} ({len(pdf_document)} trang)...")
    
    for page_num in range(len(pdf_document)):
        page = pdf_document[page_num]
        
        # Thử lấy text thông thường trước (dành cho PDF text chuẩn)
        text = page.get_text().strip()
        
        # Nếu text quá ngắn (có thể là PDF scan), tạm thời bỏ qua phần OCR theo yêu cầu
        if len(text) < 50:
            print(f"    - Trang {page_num+1} có dạng ảnh (Scan), tạm thời bỏ qua OCR...")
            text = "" # Bỏ qua trang scan
                
        # Bọc kết quả vào Document của Langchain để truyền cho Preprocessor
        doc = Document(
            page_content=text,
            metadata={"source": str(pdf_path), "page": page_num}
        )
        docs.append(doc)
        
    pdf_document.close()
    return docs

def main():
    rename_dir = project_root / "data" / "rename"
    
    if not rename_dir.exists():
        print(f"Thư mục {rename_dir} không tồn tại. Hãy chạy rename_pdf_1.py trước!")
        return

    # Khởi tạo Preprocessor
    preprocessor = DocumentPreprocessor(
        header_lines=3,
        similarity_threshold=90,
        repeat_ratio=0.5,
        processed_dir=str(project_root / "data" / "processed")
    )

    pdf_files = sorted(rename_dir.glob("*.pdf"))
    print(f"[*] Tìm thấy {len(pdf_files)} file PDF cần preprocess.")

    for pdf_file in pdf_files:
        print(f"\n==========================================")
        print(f"Đang xử lý: {pdf_file.name}")
        try:
            docs = extract_text_from_pdf(pdf_file)
            if docs:
                # Hàm process() sẽ tự lưu text ra data/processed/cleaned/[tên_file].txt
                preprocessor.process(docs)
                print(f"[+] Đã xử lý xong: {pdf_file.name}")
            else:
                print(f"[!] Không trích xuất được text từ {pdf_file.name}")
        except Exception as e:
            print(f"[!] Lỗi tổng khi xử lý file {pdf_file.name}: {e}")

if __name__ == "__main__":
    main()
