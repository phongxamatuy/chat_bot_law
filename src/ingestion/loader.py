from pathlib import Path
import json
from langchain_community.document_loaders import TextLoader


class CleanTextLoader:

    def __init__(self, txt_dir: str):
        self.txt_dir = Path(txt_dir)
        self.metadata_dict = {}
        
        # Load metadata.json if exists
        meta_file = self.txt_dir.parent.parent / "metadata" / "metadata.json"
        if meta_file.exists():
            with open(meta_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                for item in data:
                    self.metadata_dict[item["file_name"]] = item

    def load(self):
        """
        Đọc toàn bộ file TXT đã xử lý trong thư mục.
        Trả về list[Document]
        """

        documents = []

        txt_files = sorted(self.txt_dir.glob("*.txt"))

        print(f"[*] Found {len(txt_files)} TXT files")

        for txt_file in txt_files:

            print(f"Loading: {txt_file.name}")

            loader = TextLoader(str(txt_file), encoding="utf-8")

            docs = loader.load()

            # Làm sạch và cập nhật metadata của file
            for doc in docs:
                # Tìm lại tên PDF gốc trong metadata.json (thay đuôi .txt thành .pdf)
                pdf_file_name = txt_file.with_suffix(".pdf").name
                
                new_metadata = {
                    "source": str(txt_file),
                    "file_name": pdf_file_name
                }
                
                # Bổ sung metadata chuẩn từ metadata.json
                if pdf_file_name in self.metadata_dict:
                    item = self.metadata_dict[pdf_file_name]
                    if item.get("title"):
                        new_metadata["title"] = item["title"]
                    if item.get("number"):
                        new_metadata["number"] = item["number"]
                    if item.get("type"):
                        new_metadata["type"] = item["type"]
                
                doc.metadata = new_metadata

            documents.extend(docs)

        print(f"[*] Loaded {len(documents)} text files")

        return documents