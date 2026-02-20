"""
Analysis script for mining supplement parquet files.

Analyzes:
1. Distribution of mining_dm_kubun values (1=前日, 2=当日, 3=直前)
2. Missing rates of each mining feature
3. Correlation between mining_dm_jyuni and odds
4. Statistical summary of mining_dm_time
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
from typing import Dict, List, Tuple

warnings.filterwarnings('ignore')

def analyze_mining_supplements(base_path: Path = Path('/sessions/great-confident-shannon/mnt/keiba-2026')):
    """
    Main analysis function for mining supplements.
    
    Args:
        base_path: Base directory containing data/ and data/supplements/
    """
    
    data_dir = base_path / 'data'
    supplement_dir = data_dir / 'supplements'
    
    print("=" * 80)
    print("MINING SUPPLEMENT ANALYSIS")
    print("=" * 80)
    print()
    
    # 1. Load and analyze mining supplements
    print("1. MINING_DM_KUBUN DISTRIBUTION ANALYSIS")
    print("-" * 80)
    
    mining_files = sorted(supplement_dir.glob('mining_*.parquet'))
    
    if not mining_files:
        print("ERROR: No mining_*.parquet files found in supplements/")
        return
    
    print(f"Found {len(mining_files)} mining supplement files:")
    for f in mining_files:
        print(f"  - {f.name}")
    print()
    
    # Aggregate mining_dm_kubun across all years
    kubun_counts = {}
    feature_missing_rates = {}
    dm_time_stats = []
    dm_jyuni_values = []
    total_rows = 0
    
    for mining_file in mining_files:
        year = mining_file.stem.split('_')[1]
        print(f"Loading {mining_file.name}...")
        
        try:
            df = pd.read_parquet(mining_file)
            total_rows += len(df)
            
            # Collect mining_dm_kubun distribution
            if 'mining_dm_kubun' in df.columns:
                kubun_vals = df['mining_dm_kubun'].fillna(-1).value_counts().to_dict()
                for k, v in kubun_vals.items():
                    kubun_counts[k] = kubun_counts.get(k, 0) + v
            
            # Collect missing rates for all mining features
            mining_cols = [c for c in df.columns if c.startswith('mining_')]
            for col in mining_cols:
                missing_count = (df[col] == -1.0).sum() + df[col].isna().sum()
                if col not in feature_missing_rates:
                    feature_missing_rates[col] = {'missing': 0, 'total': 0}
                feature_missing_rates[col]['missing'] += missing_count
                feature_missing_rates[col]['total'] += len(df)
            
            # Collect dm_time stats
            if 'mining_dm_time' in df.columns:
                dm_time_valid = df[df['mining_dm_time'] > 0]['mining_dm_time'].dropna()
                dm_time_stats.append({
                    'year': year,
                    'count': len(dm_time_valid),
                    'mean': dm_time_valid.mean() if len(dm_time_valid) > 0 else None,
                    'std': dm_time_valid.std() if len(dm_time_valid) > 0 else None,
                    'min': dm_time_valid.min() if len(dm_time_valid) > 0 else None,
                    'max': dm_time_valid.max() if len(dm_time_valid) > 0 else None,
                })
            
            # Collect dm_jyuni values
            if 'mining_dm_jyuni' in df.columns:
                dm_jyuni_valid = df[df['mining_dm_jyuni'] > 0]['mining_dm_jyuni'].dropna()
                dm_jyuni_values.extend(dm_jyuni_valid.tolist())
            
        except Exception as e:
            print(f"  ERROR reading {mining_file.name}: {e}")
            continue
    
    print()
    print(f"Total rows analyzed: {total_rows:,}")
    print()
    
    # Print mining_dm_kubun distribution
    print("mining_dm_kubun VALUE DISTRIBUTION:")
    print("-" * 40)
    kubun_map = {1: '前日', 2: '当日', 3: '直前', -1: 'Missing/NaN'}
    
    if kubun_counts:
        for kubun_val in sorted(kubun_counts.keys()):
            count = kubun_counts[kubun_val]
            pct = (count / total_rows) * 100 if total_rows > 0 else 0
            label = kubun_map.get(kubun_val, f'Unknown({kubun_val})')
            print(f"  {kubun_val:3d} ({label:8s}): {count:8,d} rows ({pct:6.2f}%)")
    else:
        print("  No mining_dm_kubun data found")
    
    print()
    print()
    
    # 2. Missing rates by feature
    print("2. MISSING RATE BY MINING FEATURE")
    print("-" * 80)
    
    if feature_missing_rates:
        print(f"{'Feature Name':<30} {'Missing':<12} {'Total':<10} {'Missing %':<12}")
        print("-" * 64)
        
        for col in sorted(feature_missing_rates.keys()):
            missing = feature_missing_rates[col]['missing']
            total = feature_missing_rates[col]['total']
            pct = (missing / total) * 100 if total > 0 else 0
            print(f"{col:<30} {missing:>10,d} {total:>10,d} {pct:>10.2f}%")
    else:
        print("  No missing rate data collected")
    
    print()
    print()
    
    # 3. dm_time statistics
    print("3. MINING_DM_TIME STATISTICS (SECONDS)")
    print("-" * 80)
    
    if dm_time_stats:
        print(f"{'Year':<8} {'Count':<10} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10}")
        print("-" * 58)
        
        all_means = []
        for stat in dm_time_stats:
            if stat['mean'] is not None:
                print(f"{stat['year']:<8} {stat['count']:>9,d} {stat['mean']:>10.2f} "
                      f"{stat['std']:>10.2f} {stat['min']:>10.2f} {stat['max']:>10.2f}")
                all_means.append(stat['mean'])
            else:
                print(f"{stat['year']:<8} {stat['count']:>9,d} {'N/A':>10} {'N/A':>10} "
                      f"{'N/A':>10} {'N/A':>10}")
        
        if all_means:
            print("-" * 58)
            print(f"{'Overall':<8} {'':<10} {np.mean(all_means):>10.2f} "
                  f"{np.std(all_means):>10.2f}")
    else:
        print("  No dm_time stats collected")
    
    print()
    print()
    
    # 4. dm_jyuni statistics
    print("4. MINING_DM_JYUNI STATISTICS (PREDICTED RANK)")
    print("-" * 80)
    
    if dm_jyuni_values:
        dm_jyuni_array = np.array(dm_jyuni_values)
        
        print(f"Total valid predictions: {len(dm_jyuni_values):,}")
        print(f"Mean rank: {np.mean(dm_jyuni_array):.2f}")
        print(f"Median rank: {np.median(dm_jyuni_array):.2f}")
        print(f"Std dev: {np.std(dm_jyuni_array):.2f}")
        print(f"Min rank: {int(np.min(dm_jyuni_array))}")
        print(f"Max rank: {int(np.max(dm_jyuni_array))}")
        print()
        
        # Distribution by rank buckets
        print("Distribution by predicted rank:")
        for rank_bucket in [1, 2, 3, 4, 5, (6, 10), (11, 20)]:
            if isinstance(rank_bucket, tuple):
                mask = (dm_jyuni_array >= rank_bucket[0]) & (dm_jyuni_array <= rank_bucket[1])
                label = f"{rank_bucket[0]}-{rank_bucket[1]}"
            else:
                mask = dm_jyuni_array == rank_bucket
                label = str(rank_bucket)
            
            count = np.sum(mask)
            pct = (count / len(dm_jyuni_values)) * 100
            print(f"  Rank {label:<6}: {count:>8,d} ({pct:>6.2f}%)")
    else:
        print("  No dm_jyuni data collected")
    
    print()
    print()
    
    # 5. Correlation with odds (if available)
    print("5. CORRELATION ANALYSIS: MINING_DM_JYUNI vs ODDS")
    print("-" * 80)
    
    correlation_found = False
    
    for features_file in sorted(data_dir.glob('features_*.parquet')):
        year = features_file.stem.split('_')[1]
        
        try:
            features_df = pd.read_parquet(features_file)
            
            # Check if odds columns exist
            odds_cols = [c for c in features_df.columns if 'odds' in c.lower()]
            
            if odds_cols and 'mining_dm_jyuni' in features_df.columns:
                correlation_found = True
                
                # Filter rows where both are present
                valid_mask = (features_df['mining_dm_jyuni'] > 0)
                
                for odds_col in odds_cols[:3]:  # Limit to top 3 odds columns
                    if odds_col in features_df.columns:
                        valid_odds_mask = valid_mask & (features_df[odds_col] > 0)
                        
                        if valid_odds_mask.sum() > 0:
                            corr = features_df.loc[valid_odds_mask, ['mining_dm_jyuni', odds_col]].corr().iloc[0, 1]
                            n_pairs = valid_odds_mask.sum()
                            print(f"Year {year}: {odds_col}")
                            print(f"  Correlation with mining_dm_jyuni: {corr:.4f} (n={n_pairs:,})")
        
        except Exception as e:
            pass
    
    if not correlation_found:
        print("  No odds columns found in features files, or mining_dm_jyuni not present")
    
    print()
    print()
    print("=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    analyze_mining_supplements()
