from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
from pathlib import Path

from bs4 import BeautifulSoup


@dataclass(slots=True)
class TextAnalysis:
    conteos: Counter[str]
    total_terminos: int
    lemma_set: frozenset[str]
    embedding: tuple[float, ...]
    embedding_norm: float

    @classmethod
    def empty(cls) -> "TextAnalysis":
        return cls(Counter(), 0, frozenset(), tuple(), 0.0)


@dataclass(slots=True)
class ChunkRecord:
    chunk_id: int
    titulo: str
    seccion: str
    texto: str
    analisis: TextAnalysis


@dataclass(slots=True)
class SearchResult:
    chunk: ChunkRecord
    score: float
    modo: str
    clasico_score: float = 0.0
    semantico_score: float = 0.0


@dataclass(slots=True)
class _DocFeatures:
    conteos: Counter[str]
    total_terminos: int
    lemma_set: frozenset[str]
    vector_sums: dict[str, list[float]]
    vector_counts: Counter[str]


class QuijoteIndex:
    def __init__(self, nlp, chunk_size_words: int = 180, chunk_overlap_words: int = 45) -> None:
        self.nlp = nlp
        self.chunk_size_words = chunk_size_words
        self.chunk_overlap_words = chunk_overlap_words
        self.chunks: list[ChunkRecord] = []
        self.chunk_by_id: dict[int, ChunkRecord] = {}
        self.df_global: Counter[str] = Counter()
        self._idf_cache: dict[str, float] = {}
        self.total_chunks = 0
        self.total_sections = 0

    def cargar_archivo(self, path: Path) -> dict[str, int]:
        html = path.read_text(encoding="utf-8")
        sections = self._extraer_secciones(html)
        raw_chunks = self._trocear_secciones(sections)

        if not raw_chunks:
            raise ValueError("No se pudieron extraer pasajes útiles del HTML.")

        self.chunks.clear()
        self.chunk_by_id.clear()
        self.df_global.clear()
        self._idf_cache.clear()

        features_by_chunk: list[_DocFeatures] = []
        docs = self.nlp.pipe((raw_chunk["texto"] for raw_chunk in raw_chunks), batch_size=32)
        for doc in docs:
            features = self._extraer_features_doc(doc)
            features_by_chunk.append(features)
            for lema in features.lemma_set:
                self.df_global[lema] += 1

        self.total_sections = len(sections)
        self.total_chunks = len(raw_chunks)
        self._idf_cache.clear()

        for raw_chunk, features in zip(raw_chunks, features_by_chunk):
            analisis = self._construir_analisis(features)
            record = ChunkRecord(
                chunk_id=raw_chunk["chunk_id"],
                titulo=raw_chunk["titulo"],
                seccion=raw_chunk["seccion"],
                texto=raw_chunk["texto"],
                analisis=analisis,
            )
            self.chunks.append(record)
            self.chunk_by_id[record.chunk_id] = record

        return {"sections": self.total_sections, "chunks": self.total_chunks}

    def analizar_texto(self, texto: str) -> TextAnalysis:
        if not texto.strip():
            return TextAnalysis.empty()
        return self._construir_analisis(self._extraer_features_doc(self.nlp(texto)))

    def calcular_score_tfidf(self, query_analysis: TextAnalysis, chunk: ChunkRecord) -> float:
        if chunk.analisis.total_terminos == 0:
            return 0.0

        score_total = 0.0
        for lema, query_count in query_analysis.conteos.items():
            frequency = chunk.analisis.conteos.get(lema, 0)
            if frequency == 0:
                continue

            tf = frequency / chunk.analisis.total_terminos
            df = self.df_global.get(lema, 0)
            idf = math.log((1 + self.total_chunks) / (1 + df)) + 1.0
            query_weight = 1.0 + math.log(query_count)
            score_total += tf * idf * query_weight

        return score_total

    def cosine_similarity(
        self,
        left_vector: tuple[float, ...],
        left_norm: float,
        right_vector: tuple[float, ...],
        right_norm: float,
    ) -> float:
        if left_norm == 0 or right_norm == 0:
            return 0.0
        dot_product = sum(left * right for left, right in zip(left_vector, right_vector))
        return dot_product / (left_norm * right_norm)

    def _extraer_secciones(self, html: str) -> list[tuple[str, list[str]]]:
        soup = BeautifulSoup(html, "html.parser")
        sections: list[tuple[str, list[str]]] = []
        current_title = "Prólogo / Inicio"
        current_paragraphs: list[str] = []

        for element in soup.find_all(["h1", "h2", "h3", "h4", "p"]):
            text = element.get_text(" ", strip=True)
            if not text:
                continue

            if self._es_titulo(element.name, text):
                if current_paragraphs:
                    sections.append((current_title, current_paragraphs))
                current_title = text
                current_paragraphs = []
                continue

            current_paragraphs.append(text)

        if current_paragraphs:
            sections.append((current_title, current_paragraphs))

        return sections

    def _es_titulo(self, tag_name: str, text: str) -> bool:
        text_upper = text.upper()
        return tag_name in {"h1", "h2", "h3"} or text_upper.startswith("CAPÍTULO") or text_upper.startswith("CAPITULO")

    def _trocear_secciones(self, sections: list[tuple[str, list[str]]]) -> list[dict[str, object]]:
        raw_chunks: list[dict[str, object]] = []
        chunk_id = 1

        for title, paragraphs in sections:
            normalized_paragraphs = self._segmentar_parrafos_largos(paragraphs)
            chunk_texts = self._crear_chunks_desde_parrafos(normalized_paragraphs)
            for fragment_index, chunk_text in enumerate(chunk_texts, start=1):
                raw_chunks.append(
                    {
                        "chunk_id": chunk_id,
                        "titulo": f"{title} · pasaje {fragment_index}",
                        "seccion": title,
                        "texto": chunk_text,
                    }
                )
                chunk_id += 1

        return raw_chunks

    def _segmentar_parrafos_largos(self, paragraphs: list[str]) -> list[str]:
        normalized: list[str] = []
        for paragraph in paragraphs:
            words = paragraph.split()
            if len(words) <= self.chunk_size_words:
                normalized.append(paragraph)
                continue

            start = 0
            while start < len(words):
                end = min(start + self.chunk_size_words, len(words))
                normalized.append(" ".join(words[start:end]))
                if end >= len(words):
                    break
                start = max(end - self.chunk_overlap_words, start + 1)

        return normalized

    def _crear_chunks_desde_parrafos(self, paragraphs: list[str]) -> list[str]:
        if not paragraphs:
            return []

        word_counts = [len(paragraph.split()) for paragraph in paragraphs]
        chunk_texts: list[str] = []
        start = 0

        while start < len(paragraphs):
            current_parts: list[str] = []
            total_words = 0
            end = start

            while end < len(paragraphs):
                paragraph_words = word_counts[end]
                if current_parts and total_words + paragraph_words > self.chunk_size_words:
                    break
                current_parts.append(paragraphs[end])
                total_words += paragraph_words
                end += 1
                if total_words >= self.chunk_size_words:
                    break

            if current_parts:
                chunk_texts.append("\n\n".join(current_parts))

            if end >= len(paragraphs):
                break

            overlap_words = 0
            next_start = end
            while next_start > start and overlap_words < self.chunk_overlap_words:
                next_start -= 1
                overlap_words += word_counts[next_start]

            start = end if next_start == start else next_start

        return chunk_texts

    def _idf_para_lema(self, lema: str) -> float:
        cached = self._idf_cache.get(lema)
        if cached is not None:
            return cached

        if self.total_chunks == 0:
            return 1.0

        df = self.df_global.get(lema, 0)
        idf = math.log((1 + self.total_chunks) / (1 + df)) + 1.0
        self._idf_cache[lema] = idf
        return idf

    def _extraer_features_doc(self, doc) -> _DocFeatures:
        conteos: Counter[str] = Counter()
        vector_sums: dict[str, list[float]] = {}
        vector_counts: Counter[str] = Counter()

        for token in doc:
            if not token.is_alpha or token.is_stop:
                continue

            lemma = token.lemma_.lower().strip()
            if not lemma:
                continue

            conteos[lemma] += 1

            if token.has_vector:
                token_vector = [float(value) for value in token.vector]
                lemma_vector_sum = vector_sums.get(lemma)
                if lemma_vector_sum is None:
                    vector_sums[lemma] = token_vector
                else:
                    for index, value in enumerate(token_vector):
                        lemma_vector_sum[index] += value
                vector_counts[lemma] += 1

        return _DocFeatures(
            conteos=conteos,
            total_terminos=sum(conteos.values()),
            lemma_set=frozenset(conteos.keys()),
            vector_sums=vector_sums,
            vector_counts=vector_counts,
        )

    def _construir_analisis(self, features: _DocFeatures) -> TextAnalysis:
        weighted_vector_sum: list[float] = []
        total_weight = 0.0

        for lemma, lemma_vector_sum in features.vector_sums.items():
            lemma_token_count = features.vector_counts[lemma]
            if lemma_token_count == 0:
                continue

            idf = self._idf_para_lema(lemma)
            if not weighted_vector_sum:
                weighted_vector_sum = [0.0] * len(lemma_vector_sum)

            for index, value in enumerate(lemma_vector_sum):
                weighted_vector_sum[index] += value * idf
            total_weight += lemma_token_count * idf

        embedding = tuple(value / total_weight for value in weighted_vector_sum) if total_weight else tuple()
        embedding_norm = math.sqrt(sum(value * value for value in embedding)) if embedding else 0.0

        return TextAnalysis(
            conteos=features.conteos,
            total_terminos=features.total_terminos,
            lemma_set=features.lemma_set,
            embedding=embedding,
            embedding_norm=embedding_norm,
        )
