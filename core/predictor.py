"""
Predictive and Volatility Models for Phase 5.

Includes:
1. SyntheticFlowPredictor: Ridge Regression to predict 5-day forward return.
2. VolatilityProjector: Projects boundaries using ATR.
"""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
import pandas as pd

from sklearn.linear_model import RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from config.settings import RIDGE_CV_SPLITS, RIDGE_LOOKAHEAD, RIDGE_TRAIN_WINDOW
from core.indicators import atr, cmf, roc, vpt

logger = logging.getLogger(__name__)


class SyntheticFlowPredictor:
    """
    Trains a Ridge Regression model to predict the 5-day forward return.
    
    Features: Chaikin Money Flow (CMF), Volume Price Trend (VPT), 3-day ATR ROC.
    Label: 5-day forward return.
    Validation: TimeSeriesSplit (Walk-Forward CV).
    """

    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        features = pd.DataFrame(index=df.index)
        features["cmf"] = cmf(df, period=20)
        features["vpt"] = vpt(df)
        
        atr_series = atr(df, period=14)
        features["atr_roc"] = roc(atr_series, period=3)
        
        # Label: 5-day forward return
        # shift(-lookahead) gets the close 5 days in the future
        future_close = df["Close"].shift(-RIDGE_LOOKAHEAD)
        features["target"] = (future_close - df["Close"]) / df["Close"]
        
        return features

    def predict_next_return(self, df: pd.DataFrame) -> float | None:
        """
        Train using Walk-Forward CV and predict the forward return.
        
        Returns
        -------
        float | None
            The predicted return fraction (e.g. 0.05 for 5%), or None if
            insufficient data.
        """
        data = self.prepare_data(df)
        
        # Drop rows where features are NaN
        data_clean = data.dropna(subset=["cmf", "vpt", "atr_roc"])
        
        if data_clean.empty:
            return None
            
        # The target will be NaN for the last RIDGE_LOOKAHEAD days.
        train_df = data_clean.dropna(subset=["target"])
        
        if len(train_df) < max(RIDGE_CV_SPLITS * 2, 30):
            return None  # Not enough data for TimeSeriesSplit
            
        # Limit training window for relevance
        train_df = train_df.tail(RIDGE_TRAIN_WINDOW)
        
        X_train = train_df[["cmf", "vpt", "atr_roc"]].values
        y_train = train_df["target"].values
        
        # Today's features (the very last row)
        X_pred = data_clean[["cmf", "vpt", "atr_roc"]].iloc[-1:].values
        
        # Walk-Forward Cross Validation parameters
        cv = TimeSeriesSplit(n_splits=RIDGE_CV_SPLITS)
        model = RidgeCV(alphas=[0.1, 1.0, 10.0], cv=cv)
        
        # Scale features
        scaler = StandardScaler()
        try:
            X_train_scaled = scaler.fit_transform(X_train)
            X_pred_scaled = scaler.transform(X_pred)
            
            model.fit(X_train_scaled, y_train)
            pred = model.predict(X_pred_scaled)
            return float(pred[0])
        except Exception as e:
            logger.debug("Ridge Regression failed to fit: %s", e)
            return None


class VolatilityProjector:
    """Projects next-day price boundaries using ATR."""
    
    @staticmethod
    def project(df: pd.DataFrame) -> tuple[float, float] | None:
        """
        Calculate expected boundaries based on close +/- 14-day ATR.
        
        Returns
        -------
        tuple[float, float] | None
            (upper_boundary, lower_boundary), or None if insufficient data.
        """
        if len(df) < 15:
            return None
            
        last_close = float(df["Close"].iloc[-1])
        last_atr = float(atr(df, period=14).iloc[-1])
        
        if pd.isna(last_atr) or pd.isna(last_close):
            return None
            
        upper = last_close + last_atr
        lower = last_close - last_atr
        return upper, lower
