import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple

def cs_z(x: pd.Series):
    """Monthly cross-sectional z-score for firm characteristics."""
    mu = np.nanmean(x)
    sd = np.nanstd(x, ddof=0)
    if not np.isfinite(sd) or sd == 0:
        return np.zeros_like(x)
    return (x - mu) / sd

def fill_all_nan_month(s: pd.Series):
    """If a whole month is NaN, fill it with 0."""
    if s.isna().all():
        return s.fillna(0.0)
    return s

@dataclass
class SplitData:
    train_df: pd.DataFrame
    valid_df: pd.DataFrame
    test_df: pd.DataFrame
    feature_names: List[str]
    firm_cols: List[str]
    macro_cols: List[str]
    df_all: pd.DataFrame

def preprocess_openap(
    data_path: str,
    start='2000-01-31',
    end='2020-12-31',
    cut_valid='2015-01-31',
    cut_test='2018-01-31',
    min_firms_per_month=100,
    clip_x=20,
    clip_y=5
) -> SplitData:
    # --- load, period ---
    df = pd.read_parquet(data_path)
    df = df[(df['DateYM'] >= pd.Timestamp(start)) &
            (df['DateYM'] <= pd.Timestamp(end))].copy()

    # --- sort, de-dup ---
    df = (df.sort_values(['permno','DateYM'])
            .drop_duplicates(['permno','DateYM']))

    # --- target: y_excess(t+1) ---
    df['retadj_lead1'] = df.groupby('permno')['retadj'].shift(-1)
    df['Rfree_lead1']  = df.groupby('permno')['Rfree'].shift(-1)
    df['y_excess']     = df['retadj_lead1'] - df['Rfree_lead1']

    exclude_cols = ['permno','DateYM','retadj','Rfree',
                    'retadj_lead1','Rfree_lead1','y_excess']
    feature_raw = [c for c in df.columns if c not in exclude_cols]

    # --- lag 1: feature(t-1) ---
    for c in feature_raw:
        df[c] = df.groupby('permno')[c].shift(1)

    # --- NaN month handling ---
    for c in feature_raw:
        df[c] = df.groupby('DateYM')[c].transform(fill_all_nan_month)

    # --- firm / macro split ---
    firm_cols_raw  = [c for c in feature_raw if not c.startswith('macro_')]
    macro_cols_raw = [c for c in feature_raw if c.startswith('macro_')]

    # --- firm: cross-sectional z-score ---
    for c in firm_cols_raw:
        df[c] = df.groupby('DateYM')[c].transform(cs_z)

    # --- effective sample filtering ---
    valid_mask = df['y_excess'].notna()
    n_by_month = df.groupby('DateYM')['permno'].nunique()
    good_months = n_by_month[n_by_month >= min_firms_per_month].index
    df_sdf = df.loc[valid_mask & df['DateYM'].isin(good_months)].copy()

    feature_names = firm_cols_raw + macro_cols_raw

    # --- NaN/Inf & clipping ---
    df_sdf[feature_names] = df_sdf.groupby('DateYM')[feature_names].transform(lambda x: x.fillna(0.0))
    df_sdf[feature_names] = (df_sdf[feature_names]
                             .replace([np.inf, -np.inf], np.nan)
                             .fillna(0.0))
    df_sdf[feature_names] = df_sdf[feature_names].clip(-clip_x, clip_x)
    df_sdf['y_excess']    = df_sdf['y_excess'].clip(-clip_y, clip_y)

    # --- split ---
    cut_valid = pd.Timestamp(cut_valid)
    cut_test  = pd.Timestamp(cut_test)

    train_df = df_sdf[df_sdf['DateYM'] < cut_valid].copy()
    valid_df = df_sdf[(df_sdf['DateYM'] >= cut_valid) & (df_sdf['DateYM'] < cut_test)].copy()
    test_df  = df_sdf[df_sdf['DateYM'] >= cut_test].copy()

    # --- macro re-standardize using train months ---
    macro_cols = macro_cols_raw
    if len(macro_cols) > 0:
        macro_monthly_train = (train_df.groupby('DateYM')[macro_cols].first().sort_index())
        macro_mu = macro_monthly_train.mean()
        macro_sd = macro_monthly_train.std(ddof=0).replace(0, np.nan).fillna(1.0)
        for df_ in (train_df, valid_df, test_df):
            for c in macro_cols:
                df_[c] = (df_[c] - macro_mu[c]) / macro_sd[c]
            df_[macro_cols] = (df_[macro_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0))

    return SplitData(
        train_df=train_df, valid_df=valid_df, test_df=test_df,
        feature_names=feature_names, firm_cols=firm_cols_raw, macro_cols=macro_cols_raw,
        df_all=df_sdf
    )

def check_splits(train_df, valid_df, test_df, feature_names, macro_cols, min_n=100):
    # shape
    assert train_df[feature_names].shape[1] == valid_df[feature_names].shape[1] == test_df[feature_names].shape[1] == len(feature_names)
    # date non-overlap
    tr_max = train_df['DateYM'].max()
    va_min = valid_df['DateYM'].min()
    va_max = valid_df['DateYM'].max()
    te_min = test_df['DateYM'].min()
    assert tr_max < va_min <= va_max < te_min
    # y_excess no nan
    for df_ in (train_df, valid_df, test_df):
        assert df_['y_excess'].isna().sum() == 0
    # monthly min firms
    for df_ in (train_df, valid_df, test_df):
        n_by_m = df_.groupby('DateYM')['permno'].nunique()
        assert (n_by_m < min_n).sum() == 0
    # macro standard (train)
    if len(macro_cols) > 0:
        monthly = train_df.groupby('DateYM')[macro_cols].first()
        _ = monthly.mean(), monthly.std(ddof=0)
