#!/usr/bin/env python3
"""
calculate_impact_full.py - ENHANCED VERSION

Batch prediction + resource & cost impact simulation with:
- Realistic bed occupancy forecasting
- Monte Carlo simulation with confidence intervals
- Model performance metrics
- Stratified analysis by diagnosis/risk
- Automated visualizations
- Alert generation
- What-if scenario analysis

Usage examples:
  python calculate_impact_full.py --features data/mimic_iv_processed/advanced_features.csv
  python calculate_impact_full.py --features data/advanced_features.csv --use-sample --sample-size 100
  python calculate_impact_full.py --features data/advanced_features.csv --chunk-size 5000 --output results/preds.csv
"""

import os
import sys
import re
import argparse
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from joblib import load

# Try to import visualization libraries (optional)
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("Warning: matplotlib/seaborn not available. Plots will be skipped.")

# Try to import sklearn for metrics (optional)
try:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: sklearn not available. Performance metrics will be limited.")

# --- Configuration ---
config_dir = os.path.join(os.path.dirname(__file__), '..', 'config')
sys.path.append(config_dir)
try:
    from hospital_assumptions import (
        TOTAL_ICU_BEDS,
        SIMULATION_HORIZON_DAYS,
        AVERAGE_DAILY_PATIENT_COST,
        SAPS_II_RISK_THRESHOLD,
    )
    try:
        from hospital_assumptions import AVERAGE_DAILY_PATIENT_COST_CURRENCY
    except ImportError:
        AVERAGE_DAILY_PATIENT_COST_CURRENCY = None
except Exception as e:
    print("ERROR: Could not import required hospital_assumptions from config directory.")
    print("Make sure config/hospital_assumptions.py exists and defines required constants.")
    print("Exception:", e)
    sys.exit(1)

# --- Model paths ---
ADVANCED_MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models', 'advanced')
MODEL_PATHS = {
    "generalist_p50": os.path.join(ADVANCED_MODEL_DIR, "los_p50_model.joblib"),
    "specialist_p50": os.path.join(ADVANCED_MODEL_DIR, "specialist_model.joblib"),
    "generalist_p90": os.path.join(ADVANCED_MODEL_DIR, "los_p90_model.joblib"),
    "specialist_p90": os.path.join(ADVANCED_MODEL_DIR, "los_p90_model.joblib"),
}

# --- Diagnosis mapping ---
DIAGNOSIS_MAP = {
    'infection': ['sepsis', 'pneumonia', 'cellulitis', 'urinary tract infection', 'infection'],
    'cardiovascular': ['heart failure', 'atrial fibrillation', 'myocardial infarction', 'stroke', 
                      'cardiac', 'hypertension', 'vascular', 'arrhythmia'],
    'respiratory': ['respiratory failure', 'copd', 'asthma', 'pulmonary embolism', 'respiratory'],
    'gastrointestinal': ['gastrointestinal bleed', 'pancreatitis', 'liver failure', 
                        'bowel obstruction', 'abdominal pain'],
    'neurological': ['seizure', 'altered mental status', 'neurological', 'brain'],
    'renal': ['renal failure', 'kidney'],
    'trauma/injury': ['trauma', 'fall', 'fracture', 'injury'],
    'cancer': ['cancer', 'leukemia', 'lymphoma', 'tumor'],
    'metabolic/endocrine': ['diabetes', 'electrolyte imbalance', 'endocrine'],
    'other': []
}

DIAGNOSIS_PATTERNS = {
    category: re.compile('|'.join(re.escape(k) for k in keywords), re.IGNORECASE)
    for category, keywords in DIAGNOSIS_MAP.items() if keywords
}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def map_diagnosis_to_category(diagnosis_text):
    """Map free-text diagnosis -> category."""
    if not isinstance(diagnosis_text, str):
        return 'other'
    for category, pattern in DIAGNOSIS_PATTERNS.items():
        if pattern.search(diagnosis_text):
            return category
    return 'other'

def load_model_safe(path):
    """Load joblib model, return None on error."""
    try:
        m = load(path)
        print(f"✓ Loaded: {os.path.basename(path)}")
        return m
    except FileNotFoundError:
        print(f"✗ Not found: {os.path.basename(path)}")
        return None
    except Exception as e:
        print(f"✗ Error loading {os.path.basename(path)}: {e}")
        return None

def format_currency(amount, currency_code):
    """Return formatted string for currency."""
    try:
        amount_int = int(round(amount))
        if currency_code == "INR":
            s = str(amount_int)
            n = len(s)
            if n <= 3: return f"₹{s}"
            prefix = s[:-3]
            last_three = s[-3:]
            formatted_prefix = ""
            while len(prefix) > 2:
                formatted_prefix = "," + prefix[-2:] + formatted_prefix
                prefix = prefix[:-2]
            formatted_prefix = prefix + formatted_prefix
            return f"₹{formatted_prefix},{last_three}"
        elif currency_code == "USD":
            return f"${amount_int:,}"
        elif currency_code == "EUR":
            return f"€{amount_int:,}".replace(",", ".")
        elif currency_code == "GBP":
            return f"£{amount_int:,}"
        else:
            return f"{currency_code} {amount_int:,}"
    except Exception:
        return str(amount)

# ============================================================================
# PREDICTION FUNCTIONS
# ============================================================================

def batch_predict_routing(df_chunk, generalist_model, specialist_model, threshold):
    """Predict using routing logic based on SAPS-II threshold."""
    n = len(df_chunk)
    preds = np.zeros(n, dtype=float)
    
    if 'saps_ii_score' not in df_chunk.columns:
        raise ValueError("'saps_ii_score' missing from DataFrame.")

    high_mask = df_chunk['saps_ii_score'] > threshold
    low_mask = ~high_mask

    # Predict low-risk with generalist
    if low_mask.any():
        low_df = df_chunk.loc[low_mask]
        try:
            preds_low = generalist_model.predict(low_df)
            preds[np.where(low_mask)[0]] = np.maximum(0, preds_low)
        except Exception as e:
            print(f"Warning: Generalist prediction error: {e}")
            for idx in low_df.index:
                try:
                    pred_single = generalist_model.predict(low_df.loc[[idx]])[0]
                    preds[df_chunk.index.get_loc(idx)] = max(0, pred_single)
                except Exception:
                    preds[df_chunk.index.get_loc(idx)] = np.nan

    # Predict high-risk with specialist
    if high_mask.any():
        high_df = df_chunk.loc[high_mask]
        try:
            preds_high = specialist_model.predict(high_df)
            preds[np.where(high_mask)[0]] = np.maximum(0, preds_high)
        except Exception as e:
            print(f"Warning: Specialist prediction error: {e}")
            for idx in high_df.index:
                try:
                    pred_single = specialist_model.predict(high_df.loc[[idx]])[0]
                    preds[df_chunk.index.get_loc(idx)] = max(0, pred_single)
                except Exception:
                    preds[df_chunk.index.get_loc(idx)] = np.nan

    return preds

# ============================================================================
# PERFORMANCE EVALUATION
# ============================================================================

def evaluate_model_performance(df, actual_col='actual_los'):
    """Calculate model performance metrics if actual LOS is available."""
    if actual_col not in df.columns:
        print(f"\nNote: '{actual_col}' column not found. Skipping performance evaluation.")
        return None
    
    if not SKLEARN_AVAILABLE:
        print("\nNote: sklearn not available. Skipping detailed performance metrics.")
        return None
    
    # Remove rows with missing values
    eval_df = df[[actual_col, 'predicted_median_los', 'predicted_p90_los']].dropna()
    
    if len(eval_df) == 0:
        print("\nWarning: No valid rows for performance evaluation.")
        return None
    
    actual = eval_df[actual_col].values
    pred_median = eval_df['predicted_median_los'].values
    pred_p90 = eval_df['predicted_p90_los'].values
    
    # Calculate metrics
    mae_median = mean_absolute_error(actual, pred_median)
    rmse_median = np.sqrt(mean_squared_error(actual, pred_median))
    r2_median = r2_score(actual, pred_median)
    
    median_ae = np.median(np.abs(actual - pred_median))
    
    # P90 calibration: what % of actuals fall below p90 prediction
    p90_coverage = (actual <= pred_p90).mean() * 100
    
    metrics = {
        'mae': mae_median,
        'rmse': rmse_median,
        'r2': r2_median,
        'median_ae': median_ae,
        'p90_coverage': p90_coverage,
        'n_samples': len(eval_df)
    }
    
    print("\n" + "="*70)
    print("MODEL PERFORMANCE METRICS")
    print("="*70)
    print(f"Sample Size: {metrics['n_samples']:,}")
    print(f"\nMedian Predictions (P50):")
    print(f"  Mean Absolute Error (MAE):    {metrics['mae']:.2f} days")
    print(f"  Root Mean Squared Error:      {metrics['rmse']:.2f} days")
    print(f"  Median Absolute Error:        {metrics['median_ae']:.2f} days")
    print(f"  R² Score:                     {metrics['r2']:.3f}")
    print(f"\nP90 Predictions:")
    print(f"  Coverage (% actuals ≤ P90):   {metrics['p90_coverage']:.1f}%")
    print(f"  Target:                       ~90%")
    
    if metrics['p90_coverage'] < 85:
        print("  ⚠️  P90 predictions may be too optimistic")
    elif metrics['p90_coverage'] > 95:
        print("  ⚡ P90 predictions may be too conservative")
    
    return metrics

def stratified_analysis(df, actual_col='actual_los'):
    """Analyze performance by diagnosis category and risk level."""
    print("\n" + "="*70)
    print("STRATIFIED ANALYSIS")
    print("="*70)
    
    # By diagnosis category
    if 'diagnosis_category' in df.columns:
        print("\nBy Diagnosis Category:")
        print("-"*70)
        diag_stats = df.groupby('diagnosis_category').agg({
            'predicted_median_los': ['mean', 'std', 'count'],
            'saps_ii_score': 'mean'
        }).round(2)
        diag_stats.columns = ['_'.join(col).strip() for col in diag_stats.columns.values]
        print(diag_stats.to_string())
    
    # By risk level
    if 'saps_ii_score' in df.columns:
        print("\n\nBy Risk Level (SAPS-II):")
        print("-"*70)
        df['risk_category'] = pd.cut(
            df['saps_ii_score'],
            bins=[-1, 15, 30, 100],
            labels=['Low (0-15)', 'Medium (16-30)', 'High (>30)']
        )
        risk_stats = df.groupby('risk_category', observed=True).agg({
            'predicted_median_los': ['mean', 'std', 'count'],
            'saps_ii_score': 'mean'
        }).round(2)
        risk_stats.columns = ['_'.join(col).strip() for col in risk_stats.columns.values]
        print(risk_stats.to_string())
    
    # Actual vs predicted if available
    if actual_col in df.columns and SKLEARN_AVAILABLE:
        print("\n\nPerformance by Diagnosis Category:")
        print("-"*70)
        for category in df['diagnosis_category'].unique():
            subset = df[df['diagnosis_category'] == category]
            if len(subset) > 10:  # Only if enough samples
                valid = subset[[actual_col, 'predicted_median_los']].dropna()
                if len(valid) > 0:
                    mae = mean_absolute_error(valid[actual_col], valid['predicted_median_los'])
                    print(f"  {category:20s} - MAE: {mae:5.2f} days (n={len(valid):,})")

# ============================================================================
# IMPROVED SIMULATION CLASS
# ============================================================================

class ImprovedICUSimulator:
    """Monte Carlo simulation for realistic ICU forecasting."""
    
    def __init__(self, total_beds, daily_cost, currency="INR", current_occupancy_rate=0.75):
        self.total_beds = total_beds
        self.daily_cost = daily_cost
        self.currency = currency
        self.current_occupancy_rate = current_occupancy_rate
        
    def analyze_historical_patterns(self, df_historical, use_typical_rates=True):
        """Extract patterns from historical predictions."""
        patterns = {
            'mean_los': df_historical['predicted_median_los'].mean(),
            'median_los': df_historical['predicted_median_los'].median(),
            'std_los': df_historical['predicted_median_los'].std(),
            'p90_los': df_historical['predicted_p90_los'].quantile(0.9),
            'total_patients': len(df_historical)
        }
        
        if use_typical_rates:
            # Use typical ICU admission rates based on bed capacity
            # Rule of thumb: ICU beds turn over ~2-3x per month
            # For 50 beds: expect ~3-6 admissions per day typically
            patterns['daily_admissions'] = self.total_beds * 0.1  # 10% of beds per day
            patterns['admission_rate_method'] = 'typical_rate_estimate'
        else:
            # Estimate from dataset (only use if you have actual admission dates)
            if 'admission_datetime' in df_historical.columns:
                df_sorted = df_historical.sort_values('admission_datetime')
                date_range = (df_sorted['admission_datetime'].max() - 
                            df_sorted['admission_datetime'].min()).days
                patterns['daily_admissions'] = len(df_historical) / max(date_range, 1)
                patterns['admission_rate_method'] = 'calculated_from_data'
            else:
                # Fallback: assume dataset spans 2-3 years
                estimated_days = len(df_historical) / (self.total_beds * 0.8 * 365 / patterns['mean_los'])
                patterns['daily_admissions'] = len(df_historical) / max(estimated_days, 365)
                patterns['admission_rate_method'] = 'heuristic_estimate'
        
        return patterns
    def generate_current_census(self):
        """Generate realistic current patient census."""
        num_patients = int(self.total_beds * self.current_occupancy_rate)
        rng = np.random.default_rng(seed=42)
        
        # Patients at various stages of stay
        days_in_icu = np.minimum(rng.exponential(scale=3, size=num_patients), 30)
        remaining_los = rng.lognormal(mean=1.0, sigma=0.8, size=num_patients)
        
        now = datetime.now()
        
        return pd.DataFrame({
            'patient_id': [f'current_{i}' for i in range(num_patients)],
            'admission_datetime': [now - timedelta(days=float(d)) for d in days_in_icu],
            'predicted_discharge_datetime': [now + timedelta(days=float(r)) for r in remaining_los],
            'days_in_icu': days_in_icu,
            'predicted_remaining_los': remaining_los
        })
    
    def forecast_occupancy_monte_carlo(self, current_patients, daily_admission_rate, 
                                       avg_los, horizon_days=14, num_simulations=1000):
        """Run Monte Carlo simulation for occupancy forecasting."""
        now = datetime.now()
        forecast_dates = [now + timedelta(days=i) for i in range(horizon_days)]
        
        occupancy_matrix = np.zeros((num_simulations, horizon_days))
        rng = np.random.default_rng(seed=42)
        
        print(f"\nRunning Monte Carlo simulation ({num_simulations} iterations)...")
        
        for sim in range(num_simulations):
            if (sim + 1) % 200 == 0:
                print(f"  Progress: {sim + 1}/{num_simulations}", end='\r')
            
            patients = current_patients.copy()
            
            for day_idx, forecast_date in enumerate(forecast_dates):
                # Remove discharged patients
                patients = patients[patients['predicted_discharge_datetime'] >= forecast_date]
                
                # Simulate new admissions (Poisson)
                num_new_admits = rng.poisson(daily_admission_rate)
                
                if num_new_admits > 0:
                    # Generate LOS for new patients
                    new_los = rng.lognormal(mean=np.log(max(avg_los, 1)), sigma=0.6, 
                                           size=num_new_admits)
                    new_discharges = [forecast_date + timedelta(days=float(los)) 
                                     for los in new_los]
                    
                    new_patients = pd.DataFrame({
                        'patient_id': [f'sim{sim}_new{i}' for i in range(num_new_admits)],
                        'admission_datetime': [forecast_date] * num_new_admits,
                        'predicted_discharge_datetime': new_discharges,
                        'predicted_remaining_los': new_los
                    })
                    
                    patients = pd.concat([patients, new_patients], ignore_index=True)
                
                occupancy_matrix[sim, day_idx] = len(patients)
        
        print(f"  Progress: {num_simulations}/{num_simulations} - Complete!     ")
        
        # Calculate statistics
        forecast_df = pd.DataFrame({
            'date': [d.strftime('%Y-%m-%d') for d in forecast_dates],
            'day_from_now': list(range(horizon_days)),
            'mean_occupancy': occupancy_matrix.mean(axis=0),
            'median_occupancy': np.median(occupancy_matrix, axis=0),
            'p10_occupancy': np.percentile(occupancy_matrix, 10, axis=0),
            'p90_occupancy': np.percentile(occupancy_matrix, 90, axis=0),
            'std_occupancy': occupancy_matrix.std(axis=0),
            'prob_near_capacity': (occupancy_matrix >= self.total_beds * 0.9).mean(axis=0) * 100,
            'prob_at_capacity': (occupancy_matrix >= self.total_beds).mean(axis=0) * 100,
            'prob_over_capacity': (occupancy_matrix > self.total_beds).mean(axis=0) * 100
        })
        
        return forecast_df
    
    def calculate_financial_impact(self, forecast_df):
        """Calculate cost projections."""
        forecast_df['expected_daily_cost'] = forecast_df['mean_occupancy'] * self.daily_cost
        forecast_df['p10_daily_cost'] = forecast_df['p10_occupancy'] * self.daily_cost
        forecast_df['p90_daily_cost'] = forecast_df['p90_occupancy'] * self.daily_cost
        forecast_df['cumulative_cost'] = forecast_df['expected_daily_cost'].cumsum()
        
        return forecast_df
    
    def generate_alerts(self, forecast_df):
        """Generate actionable alerts."""
        alerts = []
        
        for _, row in forecast_df.iterrows():
            if row['prob_over_capacity'] > 10:
                alerts.append({
                    'date': row['date'],
                    'severity': 'CRITICAL',
                    'probability': row['prob_over_capacity'],
                    'message': f"{row['prob_over_capacity']:.0f}% chance of exceeding capacity"
                })
            elif row['prob_at_capacity'] > 15:
                alerts.append({
                    'date': row['date'],
                    'severity': 'HIGH',
                    'probability': row['prob_at_capacity'],
                    'message': f"{row['prob_at_capacity']:.0f}% chance of reaching capacity"
                })
            elif row['prob_near_capacity'] > 30:
                alerts.append({
                    'date': row['date'],
                    'severity': 'MEDIUM',
                    'probability': row['prob_near_capacity'],
                    'message': f"{row['prob_near_capacity']:.0f}% chance of >90% occupancy"
                })
        
        return pd.DataFrame(alerts) if alerts else None
    
    def print_forecast_report(self, forecast_df, historical_patterns, alerts_df):
        """Print comprehensive forecast report."""
        print("\n" + "="*80)
        print("ICU CAPACITY & FINANCIAL FORECAST REPORT")
        print("="*80)
        
        print(f"\nICU Configuration:")
        print(f"  Total Beds:               {self.total_beds}")
        print(f"  Current Occupancy Rate:   {self.current_occupancy_rate*100:.0f}%")
        print(f"  Current Census:           {int(self.total_beds * self.current_occupancy_rate)}")
        print(f"  Daily Cost per Patient:   {format_currency(self.daily_cost, self.currency)}")
        
        print(f"\nHistorical Patterns:")
        print(f"  Estimated Daily Admissions: {historical_patterns['daily_admissions']:.1f}")
        print(f"  Average LOS:                {historical_patterns['mean_los']:.1f} days")
        print(f"  Median LOS:                 {historical_patterns['median_los']:.1f} days")
        print(f"  90th Percentile LOS:        {historical_patterns['p90_los']:.1f} days")
        
        print(f"\n{'-'*80}")
        print(f"{'Date':<12} {'Mean':<7} {'P10-P90':<12} {'Near Cap%':<11} {'At Cap%':<9} {'Daily Cost':<15} Status")
        print(f"{'-'*80}")
        
        for _, row in forecast_df.iterrows():
            status = "✓ OK"
            if row['prob_over_capacity'] > 10:
                status = "🔴 CRITICAL"
            elif row['prob_at_capacity'] > 15:
                status = "🟠 HIGH RISK"
            elif row['prob_near_capacity'] > 30:
                status = "🟡 WATCH"
            
            occupancy_range = f"{row['p10_occupancy']:.0f}-{row['p90_occupancy']:.0f}"
            
            print(f"{row['date']:<12} {row['mean_occupancy']:>6.1f} {occupancy_range:>12} "
                  f"{row['prob_near_capacity']:>9.1f}% {row['prob_at_capacity']:>8.1f}% "
                  f"{format_currency(row['expected_daily_cost'], self.currency):>15} {status}")
        
        # Summary
        total_cost = forecast_df['cumulative_cost'].iloc[-1]
        total_cost_p90 = forecast_df['p90_daily_cost'].sum()
        
        print(f"\n{'='*80}")
        print(f"FINANCIAL SUMMARY ({len(forecast_df)} days):")
        print(f"  Expected Total Cost:      {format_currency(total_cost, self.currency)}")
        print(f"  Pessimistic Total (P90):  {format_currency(total_cost_p90, self.currency)}")
        print(f"  Daily Average:            {format_currency(total_cost/len(forecast_df), self.currency)}")
        
        # Alerts summary
        if alerts_df is not None and len(alerts_df) > 0:
            print(f"\n{'='*80}")
            print("⚠️  CAPACITY ALERTS:")
            critical = len(alerts_df[alerts_df['severity'] == 'CRITICAL'])
            high = len(alerts_df[alerts_df['severity'] == 'HIGH'])
            medium = len(alerts_df[alerts_df['severity'] == 'MEDIUM'])
            
            if critical > 0:
                print(f"  🔴 CRITICAL: {critical} day(s) with >10% risk of exceeding capacity")
            if high > 0:
                print(f"  🟠 HIGH:     {high} day(s) with >15% risk of reaching capacity")
            if medium > 0:
                print(f"  🟡 MEDIUM:   {medium} day(s) with >30% risk of >90% occupancy")
            
            print("\n  Detailed Alerts:")
            for _, alert in alerts_df.iterrows():
                icon = {'CRITICAL': '🔴', 'HIGH': '🟠', 'MEDIUM': '🟡'}.get(alert['severity'], '⚪')
                print(f"    {icon} {alert['date']}: {alert['message']}")
        else:
            print(f"\n✓ No capacity alerts for forecast period")

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_visualizations(df, forecast_df, output_dir='results/plots'):
    """Generate all visualization plots."""
    if not PLOTTING_AVAILABLE:
        print("\nSkipping visualizations (matplotlib not available)")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nGenerating visualizations in {output_dir}/...")
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 6)
    
    # 1. Occupancy Forecast with Confidence Intervals
    try:
        fig, ax = plt.subplots(figsize=(14, 7))
        days = forecast_df['day_from_now']
        
        ax.fill_between(days, forecast_df['p10_occupancy'], forecast_df['p90_occupancy'],
                        alpha=0.3, color='steelblue', label='10th-90th Percentile Range')
        ax.plot(days, forecast_df['mean_occupancy'], 'b-', linewidth=2.5, label='Expected Occupancy')
        ax.plot(days, forecast_df['median_occupancy'], 'g--', linewidth=1.5, label='Median')
        ax.axhline(y=TOTAL_ICU_BEDS, color='red', linestyle='--', linewidth=2, label=f'Total Capacity ({TOTAL_ICU_BEDS})')
        ax.axhline(y=TOTAL_ICU_BEDS * 0.9, color='orange', linestyle=':', linewidth=1.5, label='90% Capacity')
        
        ax.set_xlabel('Days from Now', fontsize=12, fontweight='bold')
        ax.set_ylabel('Occupied Beds', fontsize=12, fontweight='bold')
        ax.set_title('ICU Occupancy Forecast with Uncertainty Bounds', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/occupancy_forecast.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ occupancy_forecast.png")
    except Exception as e:
        print(f"  ✗ Error creating occupancy forecast: {e}")
    
    # 2. LOS Distribution by Diagnosis Category
    try:
        if 'diagnosis_category' in df.columns:
            fig, ax = plt.subplots(figsize=(12, 8))
            diag_data = df.groupby('diagnosis_category')['predicted_median_los'].apply(list)
            diag_sorted = diag_data.sort_index()
            
            positions = range(len(diag_sorted))
            bp = ax.boxplot(diag_sorted.values, positions=positions, patch_artist=True,
                           labels=diag_sorted.index, widths=0.6)
            
            for patch in bp['boxes']:
                patch.set_facecolor('lightblue')
                patch.set_alpha(0.7)
            
            ax.set_xlabel('Diagnosis Category', fontsize=12, fontweight='bold')
            ax.set_ylabel('Predicted LOS (days)', fontsize=12, fontweight='bold')
            ax.set_title('Length of Stay Distribution by Diagnosis Category', fontsize=14, fontweight='bold')
            plt.xticks(rotation=45, ha='right')
            ax.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            plt.savefig(f'{output_dir}/los_by_diagnosis.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("  ✓ los_by_diagnosis.png")
    except Exception as e:
        print(f"  ✗ Error creating LOS by diagnosis: {e}")
    
    # 3. Risk Score vs Predicted LOS
    try:
        if 'saps_ii_score' in df.columns:
            fig, ax = plt.subplots(figsize=(12, 7))
            
            # Sample if dataset too large
            plot_df = df.sample(n=min(5000, len(df)), random_state=42)
            
            scatter = ax.scatter(plot_df['saps_ii_score'], plot_df['predicted_median_los'],
                               c=plot_df['predicted_median_los'], cmap='viridis',
                               alpha=0.5, s=20, edgecolors='none')
            
            # Add threshold line
            ax.axvline(x=SAPS_II_RISK_THRESHOLD, color='red', linestyle='--', 
                      linewidth=2, label=f'Risk Threshold ({SAPS_II_RISK_THRESHOLD})')
            
            ax.set_xlabel('SAPS-II Score', fontsize=12, fontweight='bold')
            ax.set_ylabel('Predicted Median LOS (days)', fontsize=12, fontweight='bold')
            ax.set_title('Risk Score vs Predicted Length of Stay', fontsize=14, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Predicted LOS (days)', fontsize=10)
            
            plt.tight_layout()
            plt.savefig(f'{output_dir}/risk_vs_los.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("  ✓ risk_vs_los.png")
    except Exception as e:
        print(f"  ✗ Error creating risk vs LOS plot: {e}")
    
    # 4. Capacity Risk Heatmap
    try:
        fig, ax = plt.subplots(figsize=(14, 6))
        
        risk_data = forecast_df[['day_from_now', 'prob_near_capacity', 
                                 'prob_at_capacity', 'prob_over_capacity']].set_index('day_from_now').T
        
        sns.heatmap(risk_data, annot=True, fmt='.0f', cmap='RdYlGn_r', 
                   cbar_kws={'label': 'Probability (%)'}, ax=ax, vmin=0, vmax=100)
        
        ax.set_xlabel('Days from Now', fontsize=12, fontweight='bold')
        ax.set_ylabel('Risk Level', fontsize=12, fontweight='bold')
        ax.set_yticklabels(['Near Capacity\n(>90%)', 'At Capacity\n(100%)', 'Over Capacity'], rotation=0)
        ax.set_title('ICU Capacity Risk Probability Heatmap', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/capacity_risk_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ capacity_risk_heatmap.png")
    except Exception as e:
        print(f"  ✗ Error creating capacity risk heatmap: {e}")
    
    # 5. Cost Projection
    try:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # Daily costs
        days = forecast_df['day_from_now']
        ax1.fill_between(days, forecast_df['p10_daily_cost'], forecast_df['p90_daily_cost'],
                        alpha=0.3, color='green', label='10th-90th Percentile')
        ax1.plot(days, forecast_df['expected_daily_cost'], 'g-', linewidth=2.5, label='Expected Daily Cost')
        
        ax1.set_xlabel('Days from Now', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Daily Cost', fontsize=11, fontweight='bold')
        ax1.set_title('Daily Cost Projections', fontsize=13, fontweight='bold')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        
        # Cumulative costs
        ax2.plot(days, forecast_df['cumulative_cost'], 'b-', linewidth=2.5, label='Cumulative Cost')
        ax2.fill_between(days, 0, forecast_df['cumulative_cost'], alpha=0.3, color='blue')
        
        ax2.set_xlabel('Days from Now', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Cumulative Cost', fontsize=11, fontweight='bold')
        ax2.set_title('Cumulative Cost Projection', fontsize=13, fontweight='bold')
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/cost_projections.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ cost_projections.png")
    except Exception as e:
        print(f"  ✗ Error creating cost projections: {e}")
    
    # 6. Patient Distribution by Risk Category
    try:
        if 'risk_category' in df.columns:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            
            # Count by risk category
            risk_counts = df['risk_category'].value_counts()
            colors = ['#2ecc71', '#f39c12', '#e74c3c']
            ax1.pie(risk_counts.values, labels=risk_counts.index, autopct='%1.1f%%',
                   colors=colors, startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'})
            ax1.set_title('Patient Distribution by Risk Level', fontsize=13, fontweight='bold')
            
            # Average LOS by risk category
            risk_los = df.groupby('risk_category', observed=True)['predicted_median_los'].mean().sort_index()
            bars = ax2.bar(range(len(risk_los)), risk_los.values, color=colors, alpha=0.7, edgecolor='black')
            ax2.set_xticks(range(len(risk_los)))
            ax2.set_xticklabels(risk_los.index, rotation=0)
            ax2.set_ylabel('Average Predicted LOS (days)', fontsize=11, fontweight='bold')
            ax2.set_title('Average LOS by Risk Level', fontsize=13, fontweight='bold')
            ax2.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}', ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(f'{output_dir}/risk_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("  ✓ risk_distribution.png")
    except Exception as e:
        print(f"  ✗ Error creating risk distribution: {e}")
    
    print("Visualization generation complete!")

# ============================================================================
# SCENARIO ANALYSIS
# ============================================================================

def scenario_analysis(simulator, current_patients, base_admission_rate, avg_los, horizon_days=7):
    """Run what-if scenario analysis."""
    print("\n" + "="*70)
    print("WHAT-IF SCENARIO ANALYSIS")
    print("="*70)
    
    scenarios = {
        'Base Case': base_admission_rate,
        'Flu Season (+30%)': base_admission_rate * 1.3,
        'Pandemic (+50%)': base_admission_rate * 1.5,
        'Reduced Admits (-20%)': base_admission_rate * 0.8
    }
    
    print(f"\nBase admission rate: {base_admission_rate:.1f} patients/day\n")
    print(f"{'Scenario':<25} {'Avg Occupancy':<16} {'Max Occupancy':<16} {'Capacity Risk'}")
    print("-"*70)
    
    results = {}
    for name, rate in scenarios.items():
        # Quick simulation with fewer iterations for speed
        forecast = simulator.forecast_occupancy_monte_carlo(
            current_patients, rate, avg_los, 
            horizon_days=horizon_days, 
            num_simulations=200
        )
        
        avg_occ = forecast['mean_occupancy'].mean()
        max_occ = forecast['p90_occupancy'].max()
        capacity_risk = (forecast['prob_at_capacity'] > 15).sum()
        
        risk_status = "✓ Low" if capacity_risk == 0 else f"⚠️  {capacity_risk} days"
        
        print(f"{name:<25} {avg_occ:>6.1f} beds      {max_occ:>6.1f} beds      {risk_status}")
        
        results[name] = {
            'avg_occupancy': avg_occ,
            'max_occupancy': max_occ,
            'high_risk_days': capacity_risk
        }
    
    return results

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main(args):
    start_time = datetime.now()
    print("\n" + "="*70)
    print("ICU LENGTH OF STAY PREDICTION & IMPACT ANALYSIS")
    print("="*70)
    print(f"Execution started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # --- Load models ---
    print("Loading models from:", ADVANCED_MODEL_DIR)
    models = {k: load_model_safe(p) for k, p in MODEL_PATHS.items()}

    if not models.get("generalist_p50") or not models.get("specialist_p50"):
        print("\n✗ ERROR: Required models missing. Ensure models exist in models/advanced.")
        return

    # P90 fallback logic
    generalist_p90 = models.get("generalist_p90") or models.get("generalist_p50")
    specialist_p90 = models.get("specialist_p90") or generalist_p90

    if not specialist_p90:
        specialist_p90 = models.get("specialist_p50") or generalist_p90

    # --- Load features ---
    if not os.path.exists(args.features):
        print(f"\n✗ ERROR: Feature file not found: {args.features}")
        return

    print(f"\nReading features CSV: {args.features}")
    try:
        df_full = pd.read_csv(args.features)
    except Exception as e:
        print(f"✗ ERROR: Could not read CSV file. Error: {e}")
        return
    
    print(f"✓ Read {len(df_full):,} rows")

    # Check required columns
    required_cols = ['primary_diagnosis', 'saps_ii_score']
    missing_cols = [c for c in required_cols if c not in df_full.columns]
    if missing_cols:
        print(f"✗ ERROR: Missing required columns: {missing_cols}")
        return

    # Sample if requested
    if args.use_sample:
        sample_n = min(args.sample_size, len(df_full))
        print(f"Sampling {sample_n} rows (seed=42)")
        df = df_full.sample(n=sample_n, random_state=42).reset_index(drop=True)
    else:
        df = df_full.reset_index(drop=True)

    # Map diagnosis to category
    if 'diagnosis_category' not in df.columns:
        print("Mapping diagnosis text to categories...")
        df['diagnosis_category'] = df['primary_diagnosis'].astype(str).apply(map_diagnosis_to_category)

    # Ensure numeric saps_ii_score
    df['saps_ii_score'] = pd.to_numeric(df['saps_ii_score'], errors='coerce').fillna(0)

    # --- Run predictions ---
    preds_median = np.zeros(len(df), dtype=float)
    preds_p90 = np.zeros(len(df), dtype=float)

    chunk_size = args.chunk_size if args.chunk_size and args.chunk_size > 0 else len(df)
    print(f"\nPredicting in chunks of {chunk_size:,} rows...")

    for start in range(0, len(df), chunk_size):
        end = min(start + chunk_size, len(df))
        chunk = df.iloc[start:end].copy()
        
        if (start // chunk_size + 1) % 5 == 0 or end == len(df):
            print(f"  Progress: {end:,}/{len(df):,} rows ({end/len(df)*100:.1f}%)", end='\r')

        chunk_preds_median = batch_predict_routing(chunk, models['generalist_p50'], 
                                                   models['specialist_p50'], SAPS_II_RISK_THRESHOLD)
        chunk_preds_p90 = batch_predict_routing(chunk, generalist_p90, specialist_p90, 
                                                SAPS_II_RISK_THRESHOLD)

        preds_median[start:end] = chunk_preds_median
        preds_p90[start:end] = chunk_preds_p90

    print(f"  Progress: {len(df):,}/{len(df):,} rows (100.0%) - Complete!     ")

    # Handle NaNs
    failed_median = np.isnan(preds_median).sum()
    failed_p90 = np.isnan(preds_p90).sum()
    if failed_median > 0:
        print(f"⚠️  Warning: {failed_median} median predictions failed, replacing with 0")
        preds_median = np.nan_to_num(preds_median, nan=0.0)
    if failed_p90 > 0:
        print(f"⚠️  Warning: {failed_p90} p90 predictions failed, replacing with 0")
        preds_p90 = np.nan_to_num(preds_p90, nan=0.0)

    # Attach predictions
    df['predicted_median_los'] = preds_median
    df['predicted_p90_los'] = preds_p90

    print("\n✓ Predictions completed")

    # --- Evaluate performance (if actual data available) ---
    metrics = evaluate_model_performance(df, actual_col='actual_los')
    
    # --- Stratified analysis ---
    stratified_analysis(df, actual_col='actual_los')

    # --- Currency setup ---
    currency = AVERAGE_DAILY_PATIENT_COST_CURRENCY or "INR"
    daily_cost = float(AVERAGE_DAILY_PATIENT_COST)

    # --- Initialize simulator ---
    simulator = ImprovedICUSimulator(
        total_beds=TOTAL_ICU_BEDS,
        daily_cost=daily_cost,
        currency=currency,
        current_occupancy_rate=0.75  # Assume 75% current occupancy
    )

    # --- Analyze historical patterns ---
    historical_patterns = simulator.analyze_historical_patterns(df)

    # --- Generate current census ---
    current_patients = simulator.generate_current_census()

    # --- Run Monte Carlo forecast ---
    forecast_df = simulator.forecast_occupancy_monte_carlo(
        current_patients=current_patients,
        daily_admission_rate=historical_patterns['daily_admissions'],
        avg_los=historical_patterns['mean_los'],
        horizon_days=SIMULATION_HORIZON_DAYS,
        num_simulations=1000
    )

    # --- Calculate financial impact ---
    forecast_df = simulator.calculate_financial_impact(forecast_df)

    # --- Generate alerts ---
    alerts_df = simulator.generate_alerts(forecast_df)

    # --- Print forecast report ---
    simulator.print_forecast_report(forecast_df, historical_patterns, alerts_df)

    # --- Scenario analysis ---
    if not args.skip_scenarios:
        scenario_results = scenario_analysis(
            simulator, current_patients,
            historical_patterns['daily_admissions'],
            historical_patterns['mean_los'],
            horizon_days=7
        )

    # --- Generate visualizations ---
    if not args.skip_plots:
        create_visualizations(df, forecast_df, output_dir=args.plot_dir)

    # --- Save outputs ---
    print(f"\n{'='*70}")
    print("SAVING OUTPUTS")
    print(f"{'='*70}")
    
    # Save predictions
    out_path = args.output or "results/predictions_with_impact.csv"
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    save_cols = [
        'subject_id', 'hadm_id', 'anchor_age', 'gender', 'admission_type',
        'insurance', 'primary_diagnosis', 'diagnosis_category', 'saps_ii_score',
        'predicted_median_los', 'predicted_p90_los'
    ]
    
    # Add optional columns if they exist
    optional_cols = ['actual_los', 'procedure_count', 'max_creatinine', 
                    'min_hemoglobin', 'age_hr_interaction', 'risk_category']
    save_cols.extend([col for col in optional_cols if col in df.columns])
    
    save_cols_exist = [col for col in save_cols if col in df.columns]

    try:
        df[save_cols_exist].to_csv(out_path, index=False)
        print(f"✓ Predictions saved to: {out_path}")
    except Exception as e:
        print(f"✗ ERROR saving predictions: {e}")

    # Save forecast
    forecast_path = out_path.replace('.csv', '_forecast.csv')
    try:
        forecast_df.to_csv(forecast_path, index=False)
        print(f"✓ Forecast saved to: {forecast_path}")
    except Exception as e:
        print(f"✗ ERROR saving forecast: {e}")

    # Save alerts
    if alerts_df is not None:
        alerts_path = out_path.replace('.csv', '_alerts.csv')
        try:
            alerts_df.to_csv(alerts_path, index=False)
            print(f"✓ Alerts saved to: {alerts_path}")
        except Exception as e:
            print(f"✗ ERROR saving alerts: {e}")

    # Save performance metrics
    if metrics:
        metrics_path = out_path.replace('.csv', '_metrics.txt')
        try:
            with open(metrics_path, 'w') as f:
                f.write("MODEL PERFORMANCE METRICS\n")
                f.write("="*70 + "\n\n")
                f.write(f"Sample Size: {metrics['n_samples']:,}\n\n")
                f.write(f"Median Predictions (P50):\n")
                f.write(f"  MAE:  {metrics['mae']:.2f} days\n")
                f.write(f"  RMSE: {metrics['rmse']:.2f} days\n")
                f.write(f"  MedAE: {metrics['median_ae']:.2f} days\n")
                f.write(f"  R²:   {metrics['r2']:.3f}\n\n")
                f.write(f"P90 Predictions:\n")
                f.write(f"  Coverage: {metrics['p90_coverage']:.1f}%\n")
            print(f"✓ Metrics saved to: {metrics_path}")
        except Exception as e:
            print(f"✗ ERROR saving metrics: {e}")

    # --- Summary statistics ---
    print(f"\n{'='*70}")
    print("SUMMARY STATISTICS")
    print(f"{'='*70}")
    print(f"\nPredictions:")
    print(f"  Total patients analyzed:      {len(df):,}")
    print(f"  Total predicted patient-days: {df['predicted_median_los'].sum():,.0f} (median)")
    print(f"  Total predicted patient-days: {df['predicted_p90_los'].sum():,.0f} (P90)")
    print(f"  Average predicted LOS:        {df['predicted_median_los'].mean():.2f} days")
    print(f"  Median predicted LOS:         {df['predicted_median_los'].median():.2f} days")
    
    print(f"\nCost Impact:")
    total_cost_median = df['predicted_median_los'].sum() * daily_cost
    total_cost_p90 = df['predicted_p90_los'].sum() * daily_cost
    print(f"  Expected total cost (median): {format_currency(total_cost_median, currency)}")
    print(f"  Conservative total (P90):     {format_currency(total_cost_p90, currency)}")

    # --- Sample preview ---
    print(f"\n{'='*70}")
    print("SAMPLE PREDICTIONS (first 10 rows)")
    print(f"{'='*70}")
    preview_cols = ['primary_diagnosis', 'saps_ii_score', 'predicted_median_los', 'predicted_p90_los']
    preview_cols = [col for col in preview_cols if col in df.columns]
    if preview_cols:
        print(df[preview_cols].head(10).to_string(index=False, float_format="%.2f"))

    # --- Execution summary ---
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    print(f"\n{'='*70}")
    print(f"✓ Analysis complete!")
    print(f"  Execution time: {duration:.1f} seconds ({duration/60:.1f} minutes)")
    print(f"  Finished at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Enhanced ICU impact analysis with forecasting, metrics, and visualizations.",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--features", "-f", required=True, 
                       help="Path to advanced features CSV")
    parser.add_argument("--use-sample", action="store_true", 
                       help="Use random sample instead of full dataset")
    parser.add_argument("--sample-size", type=int, default=100, 
                       help="Sample size if using --use-sample")
    parser.add_argument("--chunk-size", type=int, default=10000, 
                       help="Chunk size for batch prediction")
    parser.add_argument("--output", "-o", default="results/predictions_with_impact.csv", 
                       help="Output CSV path")
    parser.add_argument("--skip-plots", action="store_true", 
                       help="Skip generating visualizations")
    parser.add_argument("--plot-dir", default="results/plots", 
                       help="Directory for plots")
    parser.add_argument("--skip-scenarios", action="store_true", 
                       help="Skip what-if scenario analysis")
    
    args = parser.parse_args()
    
    try:
        main(args)
    except KeyboardInterrupt:
        print("\n\n✗ Analysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n✗ FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)