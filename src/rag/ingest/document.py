import json
import re
import uuid
from collections import defaultdict, Counter
from typing import List, Dict
from pathlib import Path

_num_re = re.compile(r"[-+]?[\d,.]+%?|\(\d[\d,.]*\)")


class Document:
    def __init__(self, table: Dict, paragraphs: List, questions: List, raw: Dict):
        self.id = str(uuid.uuid4())
        self.table = table
        self.paragraphs = paragraphs
        self.questions = questions
        self.rows = []
        self.raw = raw

    def serialize_table(self) -> None:

        k = self.infer_header_rows()
        header = self.table["table"][: k + 1]
        body = self.table["table"][k + 1 :]

        column_labels = self.uniquify(self.build_column_labels(header))
        value_cols = list(range(1, len(column_labels)))

        for row in body:
            if not row:
                continue

            row_name = row[0]
            row = [self.clean_cell(c) for c in row] + [""] * (
                len(column_labels) - len(row)
            )
            pairs = []
            for j in value_cols:
                v = row[j]
                pairs.append(f"{column_labels[j]} = {v}")

            if pairs:
                self.rows.append(f"{row_name} | " + "; ".join(pairs))

    def infer_header_rows(self, max_header_rows: int = 4, limit: float = 0.35) -> int:
        densities = [
            self.calc_numeric_density(r) for r in self.table["table"][1:max_header_rows]
        ]
        for i, d in enumerate(densities):
            if d >= limit:
                return max(1, i)
        return min(2, len(self.table["table"]))

    def build_column_labels(self, header: List[List[str]]) -> List[str]:
        header_ff = [self.forward_fill(r) for r in header]

        n_cols = max(len(r) for r in header_ff)
        labels = []
        for j in range(n_cols):
            parts = []
            for r in header_ff:
                if j < len(r):
                    v = self.clean_cell(r[j])
                    if v:
                        parts.append(v)
            labels.append(" / ".join(dict.fromkeys(parts)))
        return labels

    def uniquify(self, labels: List[str]) -> List[str]:
        counts = Counter()
        out = []
        for lab in labels:
            key = lab or "col"
            counts[key] += 1
            out.append(key if counts[key] == 1 else f"{key} #{counts[key]}")
        return out

    def forward_fill(self, seq: List[str]) -> List[str]:
        out = []
        last = ""
        for s in seq:
            s = self.clean_cell(s)
            if s:
                last = s
            out.append(last)
        return out

    def calc_numeric_density(self, row: List[str]) -> float:
        cells = [(c or "").strip() for c in row]
        if not cells:
            return 0.0
        return sum(self.is_numeric(c) for c in cells) / len(cells)

    def is_numeric(self, x: str) -> bool:
        x = (x or "").strip()
        return bool(x) and bool(_num_re.fullmatch(x.replace(" ", "")))

    def clean_cell(self, x: str) -> str:
        return " ".join((x or "").split()).strip()


class Data:
    def __init__(self, data_dir):
        self.path = Path(data_dir)
        self.datasets = defaultdict(list)

    def load_dataset(self, filename, split):
        with open(self.path / Path(filename)) as f:
            data = json.load(f)

        for element in data:
            doc = Document(
                table=element["table"],
                paragraphs=element["paragraphs"],
                questions=element["questions"],
                raw=element,
            )
            doc.serialize_table()
            self.datasets[split].append(doc)

    def get_dataset(self, split):
        return self.datasets[split]
