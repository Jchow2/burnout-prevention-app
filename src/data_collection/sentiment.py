"""
Swappable sentiment analysis layer.

Modes
─────
  vader      VADER compound score in [-1, +1]; labels: positive/neutral/negative
             Requires: pip install vaderSentiment
  textblob   TextBlob polarity in [-1, +1]; same label thresholds
             Requires: pip install textblob
  none       No scoring — appends NA columns; use for model-ready export

To swap in a different model later, add a new branch in add_sentiment() and
keep the same three output columns: sentiment_score, sentiment_label,
sentiment_source.
"""
from __future__ import annotations

import logging
from typing import Literal

import pandas as pd

logger = logging.getLogger(__name__)

SentimentMode = Literal["vader", "textblob", "none"]

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as _Vader
    _VADER_OK = True
except ImportError:
    _VADER_OK = False

try:
    from textblob import TextBlob as _TextBlob
    _TEXTBLOB_OK = True
except ImportError:
    _TEXTBLOB_OK = False


def _vader_label(compound: float) -> str:
    if compound >= 0.05:
        return "positive"
    if compound <= -0.05:
        return "negative"
    return "neutral"


def _textblob_label(polarity: float) -> str:
    if polarity > 0.1:
        return "positive"
    if polarity < -0.1:
        return "negative"
    return "neutral"


def add_sentiment(df: pd.DataFrame, mode: SentimentMode = "vader") -> pd.DataFrame:
    """
    Add three columns to df and return a copy:
      sentiment_score   float   numeric score in [-1, +1] (or NA for mode='none')
      sentiment_label   str     'positive' | 'neutral' | 'negative' (or NA)
      sentiment_source  str     'vader' | 'textblob' | 'none'

    Operates on the 'text' column.  NaN text values are treated as empty strings.
    """
    df = df.copy()

    if mode == "none":
        df["sentiment_score"] = pd.NA
        df["sentiment_label"] = pd.NA
        df["sentiment_source"] = "none"
        return df

    if mode == "vader":
        if not _VADER_OK:
            logger.warning("vaderSentiment not installed — falling back to textblob")
            mode = "textblob"
        else:
            analyzer = _Vader()
            scores, labels = [], []
            for text in df["text"].fillna("").astype(str):
                compound = analyzer.polarity_scores(text)["compound"]
                scores.append(compound)
                labels.append(_vader_label(compound))
            df["sentiment_score"] = scores
            df["sentiment_label"] = labels
            df["sentiment_source"] = "vader"
            return df

    if mode == "textblob":
        if not _TEXTBLOB_OK:
            logger.warning("textblob not installed — sentiment skipped")
            df["sentiment_score"] = pd.NA
            df["sentiment_label"] = pd.NA
            df["sentiment_source"] = "none"
            return df
        scores, labels = [], []
        for text in df["text"].fillna("").astype(str):
            polarity = _TextBlob(text).sentiment.polarity
            scores.append(polarity)
            labels.append(_textblob_label(polarity))
        df["sentiment_score"] = scores
        df["sentiment_label"] = labels
        df["sentiment_source"] = "textblob"
        return df

    return df
