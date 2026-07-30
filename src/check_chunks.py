import sys
from pathlib import Path
import json

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

from parsing.law_parser import LawParser
from chunking.hierarchical_chunker import HierarchicalChunker
from ingestion.loader import CleanTextLoader

def main():
    data_path = project_root / "data" / "processed" / "final"
    # Lấy 1 file để test, vd BoLuat_BL_hinh_su_2015.txt
    test_file = data_path / "BoLuat_BL_hinh_su_2015.txt"
    
    if not test_file.exists():
        print(f"Không tìm thấy file test: {test_file}")
        return

    print(f"Đang kiểm tra file: {test_file.name}")
    
    # 1. Đọc nội dung file
    with open(test_file, 'r', encoding='utf-8') as f:
        content = f.read()
        
    # 2. Parsing
    parser = LawParser()
    nodes = parser.parse(content, test_file.name)
    articles = parser.extract_all_articles(nodes)
    print(f"\n[*] Parsing xong. Tìm thấy {len(articles)} Điều (Articles).")
    
    # 3. Chunking
    chunker = HierarchicalChunker(chunk_size=1500, chunk_overlap=200)
    chunks, parent_docs = chunker.chunk(articles, test_file.name)
    print(f"[*] Chunking xong. Tạo ra {len(chunks)} chunks.")
    print(f"[*] Tạo ra {len(parent_docs)} parent docs.")
    
    # 4. In thử 5 chunks đầu tiên
    print("\n================ 5 CHUNKS ĐẦU TIÊN ================")
    for i, chunk in enumerate(chunks[:5]):
        print(f"\n--- Chunk {i+1} ---")
        print(f"Metadata:")
        print(json.dumps(chunk.metadata, ensure_ascii=False, indent=2))
        print(f"Content (trích 200 ký tự):")
        print(chunk.page_content[:200] + "...")
        
    # 5. In thử 2 parent docs đầu tiên
    print("\n================ 2 PARENT DOCS ĐẦU TIÊN ================")
    for i, (p_id, p_text) in enumerate(list(parent_docs.items())[:2]):
        print(f"\n--- Parent ID: {p_id} ---")
        print(p_text[:500] + "...")

if __name__ == "__main__":
    main()
