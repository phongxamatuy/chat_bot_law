from pathlib import Path
from collections import Counter
from rapidfuzz import fuzz
from langchain_core.documents import Document
import re


class DocumentPreprocessor:

    def __init__(
        self,
        header_lines=2,
        similarity_threshold=90,
        repeat_ratio=0.7,
        processed_dir="data/processed"
    ):
        self.header_lines = header_lines
        self.similarity_threshold = similarity_threshold
        self.repeat_ratio = repeat_ratio
        self.processed_dir = Path(processed_dir)

        (self.processed_dir / "original").mkdir(parents=True, exist_ok=True)
        (self.processed_dir / "cleaned").mkdir(parents=True, exist_ok=True)

    def normalize(self, text: str):

        text = text.lower()

        text = re.sub(r"\s+", " ", text)

        return text.strip()

    def detect_headers(self, docs):

        headers = []

        for doc in docs:

            lines = [
                x.strip()
                for x in doc.page_content.splitlines()
                if x.strip()
            ]

            header = "\n".join(lines[:self.header_lines])

            headers.append(header)

        groups = []

        for header in headers:

            matched = False

            for group in groups:

                score = fuzz.ratio(
                    self.normalize(header),
                    self.normalize(group["representative"])
                )

                if score >= self.similarity_threshold:

                    group["headers"].append(header)

                    matched = True

                    break

            if not matched:

                groups.append(
                    {
                        "representative": header,
                        "headers": [header]
                    }
                )

        threshold = len(docs) * self.repeat_ratio

        detected = set()

        for group in groups:

            if len(group["headers"]) >= threshold:

                detected.update(group["headers"])

        return detected

    def remove_headers(self, docs, detected_headers):

        cleaned_docs = []

        for doc in docs:

            lines = [
                x
                for x in doc.page_content.splitlines()
            ]

            candidate = "\n".join(
                line.strip()
                for line in lines[:self.header_lines]
                if line.strip()
            )

            remove = False

            for h in detected_headers:

                score = fuzz.ratio(
                    self.normalize(candidate),
                    self.normalize(h)
                )

                if score >= self.similarity_threshold:
                    remove = True
                    break

            if remove:

                lines = lines[self.header_lines:]

            cleaned_docs.append(
                Document(
                    page_content="\n".join(lines),
                    metadata=doc.metadata
                )
            )

        return cleaned_docs

    def normalize_text(self, text):

        text = re.sub(r"\n{3,}", "\n\n", text)

        text = re.sub(r"[ \t]+", " ", text)

        return text.strip()

    def save_text(self, path, text):

        path.parent.mkdir(parents=True, exist_ok=True)

        path.write_text(text, encoding="utf-8")

    def process(self, docs):

        source = Path(docs[0].metadata["source"]).stem

        original = "\n".join(
            doc.page_content
            for doc in docs
        )

        self.save_text(
            self.processed_dir /
            "original" /
            f"{source}.txt",
            original
        )

        headers = self.detect_headers(docs)

        print("Detected Headers")

        for h in headers:
            print("----------------------")
            print(h)

        docs = self.remove_headers(docs, headers)

        cleaned = "\n".join(
            doc.page_content
            for doc in docs
        )

        cleaned = self.normalize_text(cleaned)

        self.save_text(
            self.processed_dir /
            "cleaned" /
            f"{source}.txt",
            cleaned
        )

        return Document(
            page_content=cleaned,
            metadata={
                "source": docs[0].metadata["source"]
            }
        )