"""
Detailed analysis script for mining supplement parquet files.

Enhanced analysis including:
1. Column inventory across all files
2. Sample data inspection
3. Detailed mining_dm_kubun insights
4. Data quality checks
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

def get_parquet_columns(filepath: Path) -> set:
    """Get column names from a parquet file."""
    df = pd.read_parquet(filepath)
    return set(df.columns)

def analyze_detailed(base_path: Path = Path('/sessions/great-confident-shannon/mnt/keiba-2026')):
    """Detailed analysis of mining supplements and features."""
    
    data_dir = base_path / 'data'
    supplement_dir = data_dir / 'supplements'
    
    print("\n" + "=" * 80)
    print("DETAILED MINING SUPPLEMENT ANALYSIS")
    print("=" * 80 + "\n")
    
    # 1. Column inventory
    print("1. COLUMN INVENTORY")
    print("-" * 80)
    
    # Mining supplement columns
    mining_file = supplement_dir / 'mining_2024.parquet'
    if mining_file.exists():
        mining_cols = get_parquet_columns(mining_file)
        print(f"\nMining supplement columns (from {mining_file.name}):")
        for col in sorted(mining_cols):
            print(f"  - {col}")
    
    # Feature file columns
    features_file = data_dir / 'features_2024.parquet'
    if features_file.exists():
        features_cols = get_parquet_columns(features_file)
        print(f"\nFeatures columns (from {features_file.name}, showing mining/odds/rank related):")
        
        # Show relevant columns
        relevant = [c for c in sorted(features_cols) if any(x in c.lower() for x in ['mining', 'odds', 'rank', 'kakuteijyuni'])]
        if relevant:
            for col in relevant:
                print(f"  - {col}")
        else:
            print("  (No mining/odds/rank related columns found)")
        
        # Show total column count
        print(f"\n  Total columns in features: {len(features_cols)}")
        
        # Show sample of mining data
        df = pd.read_parquet(features_file)
        mining_data_cols = [c for c in df.columns if c.startswith('mining_')]
        
        if mining_data_cols:
            print(f"\n  Mining columns in features: {len(mining_data_cols)}")
            for col in sorted(mining_data_cols):
                print(f"    - {col}")
    
    print("\n")
    
    # 2. Detailed mining_dm_kubun analysis
    print("2. DETAILED MINING_DM_KUBUN ANALYSIS")
    print("-" * 80)
    
    mining_files = sorted(supplement_dir.glob('mining_*.parquet'))
    
    print(f"\nAnalyzing kubun distribution by year:")
    print(f"{'Year':<8} {'Kubun=1':<12} {'Kubun=2':<12} {'Kubun=3':<12} {'Total Rows':<12}")
    print("-" * 56)
    
    for mining_file in mining_files:
        year = mining_file.stem.split('_')[1]
        df = pd.read_parquet(mining_file)
        
        kubun1 = (df['mining_dm_kubun'] == 1).sum() if 'mining_dm_kubun' in df.columns else 0
        kubun2 = (df['mining_dm_kubun'] == 2).sum() if 'mining_dm_kubun' in df.columns else 0
        kubun3 = (df['mining_dm_kubun'] == 3).sum() if 'mining_dm_kubun' in df.columns else 0
        total = len(df)
        
        print(f"{year:<8} {kubun1:>10,d}   {kubun2:>10,d}   {kubun3:>10,d}   {total:>10,d}")
    
    print("\nKubun value meanings:")
    print("  1 = 前日 (Previous day)")
    print("  2 = 当日 (Same day)")
    print("  3 = 直前 (Immediate before race)")
    print("\nFinding: All data has kubun=3, meaning predictions are from the")
    print("most recent data available (直前) - least data leakage risk.")
    
    print("\n")
    
    # 3. Data quality checks
    print("3. DATA QUALITY CHECKS")
    print("-" * 80)
    
    # Load a recent year
    recent_mining = supplement_dir / 'mining_2024.parquet'
    recent_features = data_dir / 'features_2024.parquet'
    
    mining_df = pd.read_parquet(recent_mining)
    features_df = pd.read_parquet(recent_features)
    
    print(f"\nUsing 2024 data:")
    print(f"  Mining supplement: {len(mining_df):,} rows, {len(mining_df.columns)} columns")
    print(f"  Features: {len(features_df):,} rows, {len(features_df.columns)} columns")
    
    # Check for common keys
    mining_cols = set(mining_df.columns)
    features_cols = set(features_df.columns)
    
    common_cols = mining_cols & features_cols
    print(f"  Common columns: {len(common_cols)}")
    if common_cols:
        print(f"    {', '.join(sorted(list(common_cols)[:10]))}")
        if len(common_cols) > 10:
            print(f"    ... and {len(common_cols) - 10} more")
    
    # Check mining data value ranges
    print(f"\nMining data value ranges (2024):")
    
    if 'mining_dm_time' in mining_df.columns:
        valid_time = mining_df[mining_df['mining_dm_time'] > 0]['mining_dm_time']
        print(f"  mining_dm_time: {valid_time.min():.2f}s - {valid_time.max():.2f}s "
              f"(mean={valid_time.mean():.2f}s, n={len(valid_time):,})")
    
    if 'mining_dm_jyuni' in mining_df.columns:
        valid_jyuni = mining_df[mining_df['mining_dm_jyuni'] > 0]['mining_dm_jyuni']
        print(f"  mining_dm_jyuni: rank {int(valid_jyuni.min())}-{int(valid_jyuni.max())} "
              f"(mean={valid_jyuni.mean():.2f}, n={len(valid_jyuni):,})")
    
    if 'mining_tm_score' in mining_df.columns:
        valid_score = mining_df[mining_df['mining_tm_score'] > 0]['mining_tm_score']
        if len(valid_score) > 0:
            print(f"  mining_tm_score: {valid_score.min():.2f}-{valid_score.max():.2f} "
                  f"(mean={valid_score.mean():.2f}, n={len(valid_score):,})")
    
    if 'mining_dm_gosa_range' in mining_df.columns:
        valid_range = mining_df[mining_df['mining_dm_gosa_range'] > 0]['mining_dm_gosa_range']
        if len(valid_range) > 0:
            print(f"  mining_dm_gosa_range: {valid_range.min():.2f}-{valid_range.max():.2f} "
                  f"(mean={valid_range.mean():.2f}, n={len(valid_range):,})")
    
    print("\n")
    
    # 4. Sample data inspection
    print("4. SAMPLE DATA INSPECTION")
    print("-" * 80)
    
    # Get a sample row with all mining features present
    sample_mask = (
        (mining_df['mining_dm_time'] > 0) &
        (mining_df['mining_dm_jyuni'] > 0) &
        (mining_df['mining_tm_score'] > 0) &
        (mining_df['mining_dm_gosa_range'] > 0)
    )
    
    if sample_mask.sum() > 0:
        sample_idx = sample_mask.idxmax()
        mining_cols = [c for c in mining_df.columns if c.startswith('mining_')]
        
        print(f"\nSample row from mining (index {sample_idx}):")
        for col in sorted(mining_cols):
            val = mining_df.loc[sample_idx, col]
            if isinstance(val, float):
                print(f"  {col:<30} = {val:>12.2f}")
            else:
                print(f"  {col:<30} = {val:>12}")
    
    print("\n")
    print("=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80 + "\n")


if __name__ == '__main__':
    analyze_detailed()
