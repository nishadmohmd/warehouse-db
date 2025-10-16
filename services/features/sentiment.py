from __future__ import annotations
import os
from typing import List, Dict, Optional

try:
    from transformers import pipeline  # type: ignore
except Exception:
    pipeline = None  # type: ignore

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer  # type: ignore
except Exception:
    SentimentIntensityAnalyzer = None  # type: ignore


class SentimentScorer:
    """Unified sentiment scorer with HF or VADER fallback.

    Returns sentiment in [-1, 1] and confidence in [0,1].
    """

    def __init__(self, domain: str = "generic") -> None:
        self.domain = domain
        self._hf = None
        self._vader = None
        # Try to initialize HF pipeline first (finance/social tuned)
        model_name = {
            "finance": os.environ.get("FIN_SENTIMENT_MODEL", "ProsusAI/finbert"),
            "social": os.environ.get(
                "SOCIAL_SENTIMENT_MODEL", "cardiffnlp/twitter-roberta-base-sentiment-latest"
            ),
            "generic": os.environ.get("GEN_SENTIMENT_MODEL", "nlptown/bert-base-multilingual-uncased-sentiment"),
        }.get(domain, "nlptown/bert-base-multilingual-uncased-sentiment")
        if pipeline is not None:
            try:
                self._hf = pipeline("sentiment-analysis", model=model_name)
            except Exception:
                self._hf = None
        if self._hf is None and SentimentIntensityAnalyzer is not None:
            try:
                self._vader = SentimentIntensityAnalyzer()
            except Exception:
                self._vader = None
        if self._hf is None and self._vader is None:
            raise RuntimeError("No sentiment backend available (Transformers or VADER)")

    def score_texts(self, texts: List[str]) -> List[Dict[str, float]]:
        if not texts:
            return []
        if self._hf is not None:
            results = self._hf(texts, truncation=True, top_k=None)
            normalized: List[Dict[str, float]] = []
            for item in results:
                # item could be list[dict] (when top_k) or dict
                if isinstance(item, list):
                    # Find positive vs negative probabilities if available
                    label_to_score = {e["label"].lower(): float(e["score"]) for e in item}
                    pos = label_to_score.get("positive", label_to_score.get("5 stars", 0.0))
                    neg = label_to_score.get("negative", label_to_score.get("1 star", 0.0))
                    score = pos - neg
                    conf = max(pos, neg)
                else:
                    label = str(item.get("label", "neutral")).lower()
                    score_raw = float(item.get("score", 0.5))
                    if "pos" in label or "5" in label:
                        score = score_raw
                    elif "neg" in label or "1" in label:
                        score = -score_raw
                    else:
                        score = 0.0
                    conf = score_raw if score >= 0 else -score
                normalized.append({"sentiment": max(-1.0, min(1.0, score)), "confidence": max(0.0, min(1.0, conf))})
            return normalized
        # VADER fallback
        assert self._vader is not None
        outputs: List[Dict[str, float]] = []
        for text in texts:
            vs = self._vader.polarity_scores(text)
            score = float(vs.get("compound", 0.0))  # already in [-1,1]
            conf = float(abs(score))
            outputs.append({"sentiment": score, "confidence": conf})
        return outputs
