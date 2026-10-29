"""
Core DocuBot class responsible for:
- Loading documents from the docs/ folder
- Building a simple retrieval index (Phase 1)
- Retrieving relevant snippets (Phase 1)
- Supporting retrieval only answers
- Supporting RAG answers when paired with Gemini (Phase 2)
"""

import os
import re
import glob

class DocuBot:
    def __init__(self, docs_folder="docs", llm_client=None):
        """
        docs_folder: directory containing project documentation files
        llm_client: optional Gemini client for LLM based answers
        """
        self.docs_folder = docs_folder
        self.llm_client = llm_client

        # Load documents into memory
        self.documents = self.load_documents()  # List of (filename, text)

        # Build a retrieval index (implemented in Phase 1)
        self.index = self.build_index(self.documents)

    # -----------------------------------------------------------
    # Document Loading
    # -----------------------------------------------------------

    def load_documents(self):
        """
        Loads all .md and .txt files inside docs_folder.
        Returns a list of tuples: (filename, text)
        """
        docs = []
        pattern = os.path.join(self.docs_folder, "*.*")
        for path in glob.glob(pattern):
            if path.endswith(".md") or path.endswith(".txt"):
                with open(path, "r", encoding="utf8") as f:
                    text = f.read()
                filename = os.path.basename(path)
                docs.append((filename, text))
        return docs

    # -----------------------------------------------------------
    # Index Construction (Phase 1)
    # -----------------------------------------------------------

    def build_index(self, documents):
        """
        TODO (Phase 1):
        Build a tiny inverted index mapping lowercase words to the documents
        they appear in.

        Example structure:
        {
            "token": ["AUTH.md", "API_REFERENCE.md"],
            "database": ["DATABASE.md"]
        }

        Keep this simple: split on whitespace, lowercase tokens,
        ignore punctuation if needed.
        """
        index = {}
        for filename, text in documents:
            for token in self._tokenize(text):
                filenames = index.setdefault(token, [])
                if filename not in filenames:
                    filenames.append(filename)
        return index

    # -----------------------------------------------------------
    # Tokenization helper
    # -----------------------------------------------------------

    def _tokenize(self, text):
        """
        Lowercase and split text into word tokens, stripping punctuation.
        """
        return re.findall(r"[a-z0-9]+", text.lower())

    # -----------------------------------------------------------
    # Scoring and Retrieval (Phase 1)
    # -----------------------------------------------------------

    def score_document(self, query, text):
        """
        Count total occurrences of each query word in the document text.
        Words that appear multiple times contribute multiple points,
        which rewards documents that discuss the query terms more heavily.
        """
        query_words = [w for w in self._tokenize(query) if w not in self.STOPWORDS]
        if not query_words:
            return 0

        doc_words = self._tokenize(text)
        score = 0
        for word in query_words:
            score += doc_words.count(word)
        return score

    def retrieve(self, query, top_k=3):
        """
        Use the inverted index to find candidate documents (those containing
        at least one query word), score each with score_document, then
        return the top_k (filename, text) pairs sorted by score descending.
        Documents that score 0 are excluded.
        """
        query_words = [w for w in self._tokenize(query) if w not in self.STOPWORDS]

        candidate_filenames = set()
        for word in query_words:
            candidate_filenames.update(self.index.get(word, []))

        scored = []
        for filename, text in self.documents:
            if filename not in candidate_filenames:
                continue
            score = self.score_document(query, text)
            if score > 0:
                scored.append((score, filename, text))

        scored.sort(key=lambda item: item[0], reverse=True)

        results = [(filename, text) for _, filename, text in scored]
        return results[:top_k]

    # -----------------------------------------------------------
    # Answering Modes
    # -----------------------------------------------------------

    MIN_SCORE_THRESHOLD = 2
    SNIPPET_RADIUS = 300  # characters shown on each side of the best match

    STOPWORDS = {
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "of",
        "to", "in", "on", "for", "and", "or", "what", "how", "does", "do",
        "it", "this", "that", "with", "as", "at", "by", "from",
    }

    def _extract_snippet(self, query, text):
        """
        Return a short window of text centered on the first query word
        found in the document, instead of the full document body.
        """
        query_words = self._tokenize(query)
        lower_text = text.lower()

        best_pos = -1
        for word in query_words:
            pos = lower_text.find(word)
            if pos != -1 and (best_pos == -1 or pos < best_pos):
                best_pos = pos

        if best_pos == -1:
            best_pos = 0

        start = max(0, best_pos - self.SNIPPET_RADIUS)
        end = min(len(text), best_pos + self.SNIPPET_RADIUS)
        snippet = text[start:end].strip()

        if start > 0:
            snippet = "..." + snippet
        if end < len(text):
            snippet = snippet + "..."

        return snippet

    def answer_retrieval_only(self, query, top_k=3):
        """
        Phase 1 retrieval only mode.
        Returns short snippets (not full documents) with no LLM involved.

        Guardrail: if the best match's score is below MIN_SCORE_THRESHOLD,
        treat it as too weak a match and refuse rather than returning
        low-confidence results.
        """
        snippets = self.retrieve(query, top_k=top_k)

        if not snippets:
            return "I do not know based on these docs."

        best_score = self.score_document(query, snippets[0][1])
        if best_score < self.MIN_SCORE_THRESHOLD:
            return "I do not know based on these docs."

        formatted = []
        for filename, text in snippets:
            snippet = self._extract_snippet(query, text)
            formatted.append(f"[{filename}]\n{snippet}\n")

        return "\n---\n".join(formatted)

    def answer_rag(self, query, top_k=3):
        """
        Phase 2 RAG mode.
        Uses student retrieval to select snippets, then asks Gemini
        to generate an answer using only those snippets.
        """
        if self.llm_client is None:
            raise RuntimeError(
                "RAG mode requires an LLM client. Provide a GeminiClient instance."
            )

        snippets = self.retrieve(query, top_k=top_k)

        if not snippets:
            return "I do not know based on these docs."

        return self.llm_client.answer_from_snippets(query, snippets)

    # -----------------------------------------------------------
    # Bonus Helper: concatenated docs for naive generation mode
    # -----------------------------------------------------------

    def full_corpus_text(self):
        """
        Returns all documents concatenated into a single string.
        This is used in Phase 0 for naive 'generation only' baselines.
        """
        return "\n\n".join(text for _, text in self.documents)
