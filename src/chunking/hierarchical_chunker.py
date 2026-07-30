import json
from typing import List, Tuple, Dict, Any
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

class HierarchicalChunker:
    def __init__(self, chunk_size: int = 1500, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def chunk(self, articles: List[Any], file_name: str) -> Tuple[List[Document], Dict[str, str]]:
        """
        Chia các Article (LawNode) thành các chunk nhỏ hơn dựa theo độ dài.
        Trả về (list các Document chunk, dict map parent_id -> parent_text)
        """
        chunks = []
        parent_docs = {}

        # Tiền xử lý file_name để lấy law id (bỏ đuôi .txt)
        law_id = file_name.replace('.txt', '')

        for article in articles:
            meta = {
                'law': law_id,
                'part': article.metadata.get('part'),
                'chapter': article.metadata.get('chapter'),
                'article': article.title,
                'clause': None,
                'point': None,
                'parent_id': None,
                'level': 'article',
                'chunk_id': f"{law_id}__{article.title.replace(' ', '_').replace('.', '')}"
            }
            
            article_text = self._get_full_text(article)
            
            if len(article_text) <= self.chunk_size or not article.children:
                doc = Document(page_content=article_text, metadata=meta.copy())
                chunks.append(doc)
            else:
                for clause in article.children:
                    if clause.node_type != 'clause':
                        # Nếu có text lạc (không phải khoản) nằm trong điều dài, ta xử lý sao?
                        # Ở đây code parse đảm bảo child chỉ có loại khoản/điểm
                        continue
                    
                    clause_text = self._get_full_text(clause)
                    c_meta = meta.copy()
                    c_meta['clause'] = json.dumps([clause.number])
                    c_meta['level'] = 'clause'
                    c_meta['parent_id'] = meta['chunk_id']
                    c_meta['chunk_id'] = f"{c_meta['parent_id']}__Khoan_{clause.number}"
                    
                    # Lưu parent doc (Article)
                    parent_docs[meta['chunk_id']] = article_text

                    if len(clause_text) <= self.chunk_size or not clause.children:
                        doc = Document(page_content=clause_text, metadata=c_meta)
                        chunks.append(doc)
                    else:
                        for point in clause.children:
                            if point.node_type != 'point':
                                continue
                            
                            point_text = self._get_full_text(point)
                            p_meta = c_meta.copy()
                            p_meta['point'] = json.dumps([point.number])
                            p_meta['level'] = 'point'
                            p_meta['parent_id'] = c_meta['chunk_id']
                            p_meta['chunk_id'] = f"{p_meta['parent_id']}__{point.number}"
                            
                            # Lưu parent doc (Clause)
                            parent_docs[c_meta['chunk_id']] = clause_text

                            if len(point_text) <= self.chunk_size:
                                doc = Document(page_content=point_text, metadata=p_meta)
                                chunks.append(doc)
                            else:
                                splits = self.text_splitter.create_documents([point_text])
                                # Lưu parent doc (Point)
                                parent_docs[p_meta['chunk_id']] = point_text
                                for i, split in enumerate(splits):
                                    s_meta = p_meta.copy()
                                    s_meta['level'] = 'sub_point'
                                    s_meta['parent_id'] = p_meta['chunk_id']
                                    s_meta['chunk_id'] = f"{p_meta['chunk_id']}_{i}"
                                    split.metadata = s_meta
                                    chunks.append(split)

        return chunks, parent_docs

    def _get_full_text(self, node: Any) -> str:
        text = node.content
        for child in node.children:
            text += "\n" + self._get_full_text(child)
        return text.strip()
