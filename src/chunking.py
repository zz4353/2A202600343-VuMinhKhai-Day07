from __future__ import annotations

import math
import re


class FixedSizeChunker:
    """
    Split text into fixed-size chunks with optional overlap.

    Rules:
        - Each chunk is at most chunk_size characters long.
        - Consecutive chunks share overlap characters.
        - The last chunk contains whatever remains.
        - If text is shorter than chunk_size, return [text].
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        step = self.chunk_size - self.overlap
        chunks: list[str] = []
        for start in range(0, len(text), step):
            chunk = text[start : start + self.chunk_size]
            chunks.append(chunk)
            if start + self.chunk_size >= len(text):
                break
        return chunks


class SentenceChunker:
    """
    Split text into chunks of at most max_sentences_per_chunk sentences.

    Sentence detection: split on ". ", "! ", "? " or ".\n".
    Strip extra whitespace from each chunk.
    """

    def __init__(self, max_sentences_per_chunk: int = 3) -> None:
        self.max_sentences_per_chunk = max(1, max_sentences_per_chunk)

    def chunk(self, text: str) -> list[str]:
        z = [". ", "! ", "? ", ".\n"]
        chunked_text = self._chunk(text.strip(), z)
        result = []
        for i in range(0, len(chunked_text), 3):
            result.append(''.join(chunked_text[i:i+3]))
        return result


    def _chunk(self, text, z):
        last_idx = 0
        result = []
        for i in range(len(text)):
            if text[i:i+2] in z:
                r = text[last_idx:i+2]
                last_idx = i + 2
                result.append(r.strip())
        return result



class RecursiveChunker:
    """
    Recursively split text using separators in priority order.

    Default separator priority:
        ["\n\n", "\n", ". ", " ", ""]
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(self, separators: list[str] | None = None, chunk_size: int = 500) -> None:
        self.separators = self.DEFAULT_SEPARATORS if separators is None else list(separators)
        self.chunk_size = chunk_size

    def chunk(self, text: str) -> list[str]:
        return self._split(text, self.separators)

    def _split(self, current_text: str, remaining_separators: list[str]) -> list[str]:
        if len(remaining_separators) == 0:
            return [current_text]
        
        result = []
        r1 = current_text.split(remaining_separators[0])
        for t in r1:
            t = str(t)
            if len(t) <= self.chunk_size:
                result.append(t)
            else:
                result.extend(self._split(t, remaining_separators[1:]))

        return result


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def compute_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    cosine_similarity = dot(a, b) / (||a|| * ||b||)

    Returns 0.0 if either vector has zero magnitude.
    """
    norm_a = 0
    norm_b = 0

    for v in vec_a:
        norm_a += v * v
    norm_a = math.sqrt(norm_a)

    if norm_a == 0:
        return 0

    for v in vec_b:
        norm_b += v * v
    norm_b = math.sqrt(norm_b)

    if norm_b == 0:
        return 0
    
    return _dot(vec_a, vec_b) / (norm_a * norm_b)




class ChunkingStrategyComparator:
    """Run all built-in chunking strategies and compare their results."""

    def compare(self, text: str, chunk_size: int = 200) -> dict:
        z = {'fixed_size': FixedSizeChunker(chunk_size=chunk_size) , 'by_sentences' : SentenceChunker(), 'recursive': RecursiveChunker(chunk_size=chunk_size)}
        
        result = {}
        for key, value in z.items():
            r = value.chunk(text)
            lengths = [len(r1) for r1 in r]
            result[key] = {
                'count': len(r),
                'avg_length': sum(lengths) / len(lengths),
                'chunks': r
            }

        return result
