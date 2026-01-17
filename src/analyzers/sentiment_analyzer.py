# src/analyzers/sentiment_analyzer.py
from transformers import pipeline

from src.analyzers.sentiment_result import SentimentResult, SentimentLabel


class SentimentAnalyzer:
    """Sentiment analyzer using FinTwitBERT model."""

    DEFAULT_MODEL = "StephanAkkerman/FinTwitBERT-sentiment"

    def __init__(self, model_name: str | None = None, batch_size: int = 32):
        """Initialize the sentiment analyzer.

        Args:
            model_name: HuggingFace model name. Defaults to FinTwitBERT.
            batch_size: Batch size for inference.
        """
        self.model_name = model_name or self.DEFAULT_MODEL
        self.batch_size = batch_size
        self._pipeline = pipeline(
            "text-classification",
            model=self.model_name,
            top_k=None,  # Return all labels with scores
        )

    def analyze(self, text: str) -> SentimentResult:
        """Analyze sentiment of a single text."""
        if not text or not text.strip():
            return SentimentResult(
                label=SentimentLabel.NEUTRAL,
                score=0.5,
                confidence=0.0,
            )

        results = self._pipeline(text)
        return self._parse_result(results[0] if isinstance(results[0], list) else results)

    def analyze_batch(self, texts: list[str]) -> list[SentimentResult]:
        """Analyze sentiment of multiple texts in batch."""
        if not texts:
            return []

        valid_texts = []
        valid_indices = []
        for i, text in enumerate(texts):
            if text and text.strip():
                valid_texts.append(text)
                valid_indices.append(i)

        if valid_texts:
            batch_results = self._pipeline(valid_texts, batch_size=self.batch_size)
        else:
            batch_results = []

        results = [
            SentimentResult(label=SentimentLabel.NEUTRAL, score=0.5, confidence=0.0)
            for _ in texts
        ]

        for idx, batch_result in zip(valid_indices, batch_results):
            parsed = batch_result if isinstance(batch_result, list) else [batch_result]
            results[idx] = self._parse_result(parsed)

        return results

    def _parse_result(self, predictions: list[dict]) -> SentimentResult:
        """Parse model output to SentimentResult."""
        label_scores = {}
        for pred in predictions:
            label = pred["label"].lower()
            label_scores[label] = pred["score"]

        bullish_score = label_scores.get("bullish", 0.0)
        bearish_score = label_scores.get("bearish", 0.0)
        neutral_score = label_scores.get("neutral", 0.0)

        max_score = max(bullish_score, bearish_score, neutral_score)
        if bullish_score == max_score:
            label = SentimentLabel.BULLISH
            score = bullish_score
        elif bearish_score == max_score:
            label = SentimentLabel.BEARISH
            score = 1.0 - bearish_score
        else:
            label = SentimentLabel.NEUTRAL
            score = 0.5

        return SentimentResult(
            label=label,
            score=score if label == SentimentLabel.BULLISH else (1.0 - bearish_score if label == SentimentLabel.BEARISH else 0.5),
            confidence=max_score,
        )
