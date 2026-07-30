import json
import os
from typing import Dict, List, Optional

class ParentChildStore:
    def __init__(self, store_path: str):
        self.store_path = store_path
        os.makedirs(os.path.dirname(store_path), exist_ok=True)

    def save(self, parent_docs: Dict[str, str]):
        existing = self.load()
        existing.update(parent_docs)
        with open(self.store_path, 'w', encoding='utf-8') as f:
            json.dump(existing, f, ensure_ascii=False, indent=2)

    def load(self) -> Dict[str, str]:
        if not os.path.exists(self.store_path):
            return {}
        with open(self.store_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def get_parent(self, parent_id: str) -> Optional[str]:
        docs = self.load()
        return docs.get(parent_id)

    def get_parents_for_chunks(self, chunks: List[Dict], max_parent_length: int = 3000) -> List[Dict]:
        """
        Lấy nội dung parent cho list các chunk (dict).
        Nếu parent dài hơn max_parent_length, sẽ bị cắt bớt để không làm tràn context LLM.
        """
        docs = self.load()
        enriched_chunks = []
        for chunk in chunks:
            new_chunk = chunk.copy()
            parent_id = chunk.get('metadata', {}).get('parent_id')
            if parent_id and parent_id in docs:
                parent_text = docs[parent_id]
                if len(parent_text) > max_parent_length:
                    parent_text = parent_text[:max_parent_length] + "\n... [ĐÃ CẮT BỚT DO QUÁ DÀI]"
                new_chunk['parent_content'] = parent_text
            else:
                new_chunk['parent_content'] = None
            enriched_chunks.append(new_chunk)
        return enriched_chunks
