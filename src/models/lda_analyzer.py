"""LDA topic model — produces a crisis topic-shift score (0.0–1.0).

Workflow
--------
1. fit(texts, labels)        — train LDA, identify which topics correlate with crises
2. predict(texts)            — return per-tweet crisis score for the ensemble
3. coherence_search(texts)   — find optimal k (call before fit if you want to tune)
4. save / load               — persist to disk

The crisis score for a tweet is the sum of probabilities assigned to
topics that were identified as "crisis topics" during fitting.
"""

import os
import pickle
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

import nltk
nltk.download("stopwords", quiet=True)
nltk.download("wordnet",   quiet=True)
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from gensim import corpora, models
from gensim.models import CoherenceModel

RANDOM_SEED = 42
_STOP = set(stopwords.words("english")) | {
    "amp", "rt", "via", "get", "one", "like", "go", "us",
    "http", "https", "www", "co", "t", "s", "u",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class LDAAnalyzer:
    def __init__(self, n_topics: int = 10, passes: int = 10):
        self.n_topics    = n_topics
        self.passes      = passes
        self.lda_model:  Optional[models.LdaModel]    = None
        self.dictionary: Optional[corpora.Dictionary] = None
        self._crisis_topics: List[int] = []   # topic IDs that correlate with crises

    # ------------------------------------------------------------------
    # Coherence tuning — call to pick n_topics before fit()
    # ------------------------------------------------------------------

    def coherence_search(
        self,
        texts: List[str],
        k_range: range = range(5, 21),
    ) -> pd.DataFrame:
        """Train LDA for each k and return coherence scores as a DataFrame."""
        corpus_tokens = [_tokenize(t) for t in texts]
        dictionary    = corpora.Dictionary(corpus_tokens)
        dictionary.filter_extremes(no_below=5, no_above=0.95)
        bow_corpus    = [dictionary.doc2bow(t) for t in corpus_tokens]

        rows = []
        for k in k_range:
            model = models.LdaModel(
                bow_corpus, num_topics=k, id2word=dictionary,
                passes=5, random_state=RANDOM_SEED,
            )
            cm = CoherenceModel(
                model=model, texts=corpus_tokens,
                dictionary=dictionary, coherence="c_v",
                processes=1,
            )
            rows.append({"k": k, "coherence": cm.get_coherence()})
            print(f"  k={k:2d}  coherence={rows[-1]['coherence']:.4f}")

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, texts: List[str], labels: List[int]) -> "LDAAnalyzer":
        """Train LDA and label topics as crisis/normal via label correlation."""
        corpus_tokens = [_tokenize(t) for t in texts]

        self.dictionary = corpora.Dictionary(corpus_tokens)
        self.dictionary.filter_extremes(no_below=5, no_above=0.95)
        bow_corpus = [self.dictionary.doc2bow(t) for t in corpus_tokens]

        print(f"Training LDA: {self.n_topics} topics, {self.passes} passes, "
              f"vocab={len(self.dictionary):,}")

        self.lda_model = models.LdaModel(
            bow_corpus,
            num_topics=self.n_topics,
            id2word=self.dictionary,
            passes=self.passes,
            alpha="auto",
            eta="auto",
            random_state=RANDOM_SEED,
        )

        # Identify crisis topics: topics whose avg activation is higher
        # in crisis tweets than normal tweets
        topic_matrix = self._topic_matrix(bow_corpus)   # (n_docs, n_topics)
        labels_arr   = np.array(labels)

        crisis_mean = topic_matrix[labels_arr == 1].mean(axis=0)
        normal_mean = topic_matrix[labels_arr == 0].mean(axis=0)
        delta       = crisis_mean - normal_mean           # positive → crisis topic

        # Any topic where crisis mean > normal mean is a "crisis topic"
        self._crisis_topics = [i for i, d in enumerate(delta) if d > 0]
        print(f"Crisis topics identified: {self._crisis_topics}")
        self._print_topics()
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, texts: List[str]) -> List[float]:
        """Return crisis score 0.0–1.0 per tweet (sum of crisis-topic probs)."""
        assert self.lda_model is not None, "Call fit() first"
        bow_corpus   = [self.dictionary.doc2bow(_tokenize(t)) for t in texts]
        topic_matrix = self._topic_matrix(bow_corpus)
        if not self._crisis_topics:
            return [0.5] * len(texts)
        scores = topic_matrix[:, self._crisis_topics].sum(axis=1)
        # Normalise to [0, 1] using the number of crisis topics
        scores = scores / len(self._crisis_topics)
        return scores.tolist()

    def topic_words(self, n_words: int = 10) -> dict:
        """Return {topic_id: [(word, prob), ...]} for all topics."""
        assert self.lda_model is not None
        return {
            i: self.lda_model.show_topic(i, topn=n_words)
            for i in range(self.n_topics)
        }

    # ------------------------------------------------------------------
    # Persist
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        self.lda_model.save(os.path.join(path, "lda.model"))
        self.dictionary.save(os.path.join(path, "dictionary.dict"))
        meta = {"n_topics": self.n_topics, "passes": self.passes,
                "crisis_topics": self._crisis_topics}
        with open(os.path.join(path, "meta.pkl"), "wb") as f:
            pickle.dump(meta, f)
        print(f"LDA model saved to {path}")

    @classmethod
    def load(cls, path: str) -> "LDAAnalyzer":
        with open(os.path.join(path, "meta.pkl"), "rb") as f:
            meta = pickle.load(f)
        obj = cls(n_topics=meta["n_topics"], passes=meta["passes"])
        obj.lda_model      = models.LdaModel.load(os.path.join(path, "lda.model"))
        obj.dictionary     = corpora.Dictionary.load(os.path.join(path, "dictionary.dict"))
        obj._crisis_topics = meta["crisis_topics"]
        return obj

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _topic_matrix(self, bow_corpus) -> np.ndarray:
        """(n_docs, n_topics) dense matrix of topic probabilities."""
        n = len(bow_corpus)
        mat = np.zeros((n, self.n_topics), dtype=np.float32)
        for i, doc_bow in enumerate(bow_corpus):
            for topic_id, prob in self.lda_model.get_document_topics(doc_bow, minimum_probability=0):
                mat[i, topic_id] = prob
        return mat

    def _print_topics(self) -> None:
        print("\nTop words per topic  [C=crisis, N=normal]")
        for tid, words in self.topic_words(8).items():
            tag   = "C" if tid in self._crisis_topics else "N"
            label = ", ".join(w for w, _ in words)
            print(f"  [{tag}] Topic {tid:2d}: {label}")


# ---------------------------------------------------------------------------
# Text preprocessing
# ---------------------------------------------------------------------------

_lemmatizer = WordNetLemmatizer()

def _tokenize(text: str) -> List[str]:
    tokens = str(text).lower().split()
    return [
        _lemmatizer.lemmatize(w)
        for w in tokens
        if w.isalpha() and w not in _STOP and len(w) > 2
    ]
