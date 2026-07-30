import re
from typing import List, Optional, Dict, Any

class LawNode:
    def __init__(self, node_type: str, title: str, content: str = "", number: str = "", parent=None):
        self.node_type = node_type  # 'part', 'chapter', 'article', 'clause', 'point', 'intro'
        self.title = title
        self.content = content
        self.number = number
        self.parent = parent
        self.children = []
        self.metadata = {}
        
    def to_dict(self):
        return {
            "node_type": self.node_type,
            "title": self.title,
            "number": self.number,
            "children_count": len(self.children),
            "content_length": len(self.content)
        }

class LawParser:
    def __init__(self):
        # Regex patterns
        self.part_pattern = re.compile(r'^\s*Phần\s+thứ\s+(\w+)', re.IGNORECASE)
        self.chapter_pattern = re.compile(r'^\s*Chương\s+([IVXLCDM]+)\b', re.IGNORECASE)
        self.article_pattern = re.compile(r'^\s*Điều\s+(\d+)\.')
        self.clause_pattern = re.compile(r'^\s*(\d+)\.\s')
        self.point_pattern = re.compile(r'^\s*([a-zđ])\)\s')

    def parse(self, text: str, file_name: str = "") -> List[LawNode]:
        lines = text.split('\n')
        
        root_nodes = []
        
        current_part = None
        current_chapter = None
        current_article = None
        current_clause = None
        current_point = None

        def add_node(node: LawNode, parent: Optional[LawNode], lst: List[LawNode]):
            if parent:
                parent.children.append(node)
                node.parent = parent
            else:
                lst.append(node)
        
        for line in lines:
            line_stripped = line.strip()
            if not line_stripped:
                continue

            # Check matches
            part_match = self.part_pattern.match(line)
            if part_match:
                number = part_match.group(1)
                node = LawNode('part', line_stripped, line, number)
                root_nodes.append(node)
                current_part = node
                current_chapter = None
                current_article = None
                current_clause = None
                current_point = None
                continue

            chapter_match = self.chapter_pattern.match(line)
            if chapter_match:
                number = chapter_match.group(1)
                node = LawNode('chapter', line_stripped, line, number)
                add_node(node, current_part, root_nodes)
                current_chapter = node
                current_article = None
                current_clause = None
                current_point = None
                continue

            article_match = self.article_pattern.match(line)
            if article_match:
                number = article_match.group(1)
                node = LawNode('article', line_stripped, line, number)
                
                # Metadata inherits from parents
                node.metadata['part'] = current_part.title if current_part else None
                node.metadata['chapter'] = current_chapter.title if current_chapter else None
                
                add_node(node, current_chapter or current_part, root_nodes)
                current_article = node
                current_clause = None
                current_point = None
                continue

            clause_match = self.clause_pattern.match(line)
            if clause_match and current_article: # Only match clause if inside article
                number = clause_match.group(1)
                node = LawNode('clause', line_stripped, line, number)
                add_node(node, current_article, root_nodes)
                current_clause = node
                current_point = None
                continue

            point_match = self.point_pattern.match(line)
            if point_match and current_clause: # Only match point if inside clause
                number = point_match.group(1)
                node = LawNode('point', line_stripped, line, number)
                add_node(node, current_clause, root_nodes)
                current_point = node
                continue

            # If no match, append to the most specific active node
            active_node = current_point or current_clause or current_article or current_chapter or current_part
            if active_node:
                active_node.content += "\n" + line
            else:
                if not root_nodes:
                    intro = LawNode('intro', 'Intro', line, '0')
                    root_nodes.append(intro)
                else:
                    root_nodes[-1].content += "\n" + line
                    
        return root_nodes

    def extract_all_articles(self, nodes: List[LawNode]) -> List[LawNode]:
        articles = []
        for node in nodes:
            if node.node_type == 'article':
                articles.append(node)
            articles.extend(self.extract_all_articles(node.children))
        return articles
