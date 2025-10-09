#!/usr/bin/env python3
"""
Column Summary Analyzer for Pickle Files
Generates comprehensive summaries for each column across all patient data files
"""
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from collections import defaultdict, Counter
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class ColumnSummaryAnalyzer:
    def __init__(self, pickle_directory, output_dir="column_analysis"):
        self.pickle_dir = pickle_directory
        self.output_dir = output_dir
        
        # Column-level aggregated statistics
        self.column_stats = defaultdict(lambda: {
            'files_present': 0,
            'total_observations': 0,
            'non_null_observations': 0,
            'null_observations': 0,
            'data_type': set(),
            'is_numerical': False,
            'is_categorical': False,
            'values': [],  # For numerical data
            'categories': Counter(),  # For categorical data
            'file_level_stats': [],  # Track stats per file
            'has_any_non_null': False  # Track if column ever has non-null values
        })
        
        # Dataset overview
        self.dataset_summary = {
            'total_files_processed': 0,
            'total_files_failed': 0,
            'total_patients': 0,
            'total_observations': 0,
            'all_columns': set(),
            'column_categories': {}
        }
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/plots", exist_ok=True)
        os.makedirs(f"{output_dir}/plots/individual_features", exist_ok=True)
    
    def categorize_columns(self, columns):
        """Categorize columns by medical domain"""
        categories = {
            'demographics': ['age', 'gender', 'race', 'ethnicity', 'gender_code'],
            'lab_values': ['anion_gap', 'base_excess', 'bicarb', 'bun', 'calcium', 'chloride', 
                          'creatinine', 'glucose', 'potassium', 'sodium', 'hematocrit', 
                          'hemoglobin', 'platelets', 'white_blood_cell', 'wbc', 'albumin',
                          'bilirubin', 'lactate', 'troponin', 'bnp'],
            'vital_signs': ['temperature', 'sbp', 'dbp', 'map', 'pulse', 'resp_rate', 'spo2', 
                           'heart_rate', 'blood_pressure', 'cvp', 'oxygen'],
            'clinical_scores': ['cci9', 'cci10', 'sirs', 'sofa', 'meld', 'aki', 'gcs'],
            'medications': ['norepinephrine', 'epinephrine', 'dobutamine', 'dopamine', 
                           'phenylephrine', 'vasopressin', 'dose'],
            'binary_flags': ['sepsis', 'infection', 'dialysis', 'pressors', 'covid', 'vent'],
            'temporal': ['elapsed', 'time', 'duration'],
            'procedures': ['procedure', 'cpt', 'icd'],
            'text_data': ['notes', 'desc', 'description'],
            'fluids': ['sodium_chloride', 'lactated', 'ringers', 'plasma', 'albumin_human']
        }
        
        categorized = defaultdict(list)
        for col in columns:
            col_lower = col.lower()
            assigned = False
            
            for category, keywords in categories.items():
                if any(keyword in col_lower for keyword in keywords):
                    categorized[category].append(col)
                    assigned = True
                    break
            
            if not assigned:
                categorized['other'].append(col)
        
        return dict(categorized)
    
    def process_single_file(self, filepath):
        """Process a single pickle file and update column statistics"""
        try:
            with open(filepath, 'rb') as f:
                df = pickle.load(f)
            
            filename = os.path.basename(filepath)
            
            # Update dataset summary
            self.dataset_summary['total_files_processed'] += 1
            self.dataset_summary['total_patients'] += 1  # Each file is one patient
            self.dataset_summary['total_observations'] += len(df)
            self.dataset_summary['all_columns'].update(df.columns)
            
            # Process each column (handle duplicate column names)
            for i, col in enumerate(df.columns):
                series = df.iloc[:, i]  # Use iloc to get column by position, not name
                col_stat = self.column_stats[col]
                
                # Update presence tracking
                col_stat['files_present'] += 1
                col_stat['total_observations'] += len(series)
                non_null_count = series.count()
                col_stat['non_null_observations'] += non_null_count
                col_stat['null_observations'] += series.isnull().sum()
                col_stat['data_type'].add(str(series.dtype))
                
                # Track if this column ever has non-null values
                if non_null_count > 0:
                    col_stat['has_any_non_null'] = True
                
                # File-level statistics for this column
                file_stats = {
                    'filename': filename,
                    'count': len(series),
                    'non_null': series.count(),
                    'null_pct': (series.isnull().sum() / len(series)) * 100 if len(series) > 0 else 0
                }
                
                # Determine if numerical or categorical
                # Special case: icu_type should be treated as categorical even if numeric
                if series.dtype in ['int64', 'float64'] and col != 'icu_type':
                    col_stat['is_numerical'] = True
                    
                    # Collect numerical values
                    non_null_values = series.dropna()
                    if len(non_null_values) > 0:

                        sampled_values = non_null_values.values
                        
                        col_stat['values'].extend(sampled_values)
                        
                        # File-level numerical stats
                        file_stats.update({
                            'mean': float(non_null_values.mean()),
                            'std': float(non_null_values.std()),
                            'min': float(non_null_values.min()),
                            'max': float(non_null_values.max()),
                            'median': float(non_null_values.median())
                        })
                
                elif series.dtype == 'object':
                    col_stat['is_categorical'] = True
                    
                    # Update category counts
                    value_counts = series.value_counts()
                    for value, count in value_counts.items():
                        col_stat['categories'][str(value)] += count
                    
                    # File-level categorical stats
                    file_stats.update({
                        'unique_count': series.nunique(),
                        'most_common': str(value_counts.index[0]) if len(value_counts) > 0 else None,
                        'most_common_count': int(value_counts.iloc[0]) if len(value_counts) > 0 else 0
                    })
                
                col_stat['file_level_stats'].append(file_stats)
            
            return True
            
        except Exception as e:
            print(f"❌ Error processing {os.path.basename(filepath)}: {e}")
            self.dataset_summary['total_files_failed'] += 1
            return False
    
    def compute_column_summaries(self):
        """Compute final summary statistics for each column"""
        print("📊 Computing column summaries...")
        
        column_summaries = {}
        
        for col, stats in self.column_stats.items():
            summary = {
                'column_name': col,
                'data_type': list(stats['data_type']),
                'files_present': stats['files_present'],
                'files_missing': self.dataset_summary['total_files_processed'] - stats['files_present'],
                'presence_percentage': (stats['files_present'] / self.dataset_summary['total_files_processed']) * 100,
                'total_observations': stats['total_observations'],
                'non_null_observations': stats['non_null_observations'],
                'null_observations': stats['null_observations'],
                'overall_null_percentage': (stats['null_observations'] / stats['total_observations']) * 100 if stats['total_observations'] > 0 else 0,
                'is_numerical': stats['is_numerical'],
                'is_categorical': stats['is_categorical']
            }
            
            # Numerical column summaries
            if stats['is_numerical'] and len(stats['values']) > 0:
                values = np.array(stats['values'])
                summary.update({
                    'numerical_stats': {
                        'count': len(values),
                        'mean': float(np.mean(values)),
                        'std': float(np.std(values)),
                        'min': float(np.min(values)),
                        'max': float(np.max(values)),
                        'median': float(np.median(values)),
                        'q25': float(np.percentile(values, 25)),
                        'q75': float(np.percentile(values, 75)),
                        'skewness': float(pd.Series(values).skew()),
                        'kurtosis': float(pd.Series(values).kurtosis()),
                        'unique_count': len(np.unique(values)),
                        'zero_count': int(np.sum(values == 0)),
                        'negative_count': int(np.sum(values < 0))
                    }
                })
                
                # Outlier detection
                if summary['numerical_stats']['std'] > 0:
                    z_scores = np.abs((values - summary['numerical_stats']['mean']) / summary['numerical_stats']['std'])
                    summary['numerical_stats']['outliers_2std'] = int(np.sum(z_scores > 2))
                    summary['numerical_stats']['outliers_3std'] = int(np.sum(z_scores > 3))
            
            # Categorical column summaries
            if stats['is_categorical'] and len(stats['categories']) > 0:
                total_categorical_obs = sum(stats['categories'].values())
                most_common = stats['categories'].most_common(10)
                least_common = stats['categories'].most_common()[-5:] if len(stats['categories']) > 5 else []
                
                summary.update({
                    'categorical_stats': {
                        'unique_categories': len(stats['categories']),
                        'total_categorical_observations': total_categorical_obs,
                        'most_common_categories': [
                            {'value': str(val), 'count': count, 'percentage': (count/total_categorical_obs)*100}
                            for val, count in most_common
                        ],
                        'least_common_categories': [
                            {'value': str(val), 'count': count, 'percentage': (count/total_categorical_obs)*100}
                            for val, count in least_common
                        ],
                        'is_high_cardinality': len(stats['categories']) > total_categorical_obs * 0.8,
                        'is_constant': len(stats['categories']) == 1
                    }
                })
            
            column_summaries[col] = summary
        
        return column_summaries
    
    def run_analysis(self, max_files=None):
        """Run the complete analysis"""
        print(f"🚀 Starting column summary analysis...")
        print(f"📁 Analyzing pickle files in: {self.pickle_dir}")
        
        # Get all pickle files
        pickle_files = []
        for filename in os.listdir(self.pickle_dir):
            if filename.endswith('.pkl'):
                pickle_files.append(os.path.join(self.pickle_dir, filename))
                if max_files and len(pickle_files) >= max_files:
                    break
        
        total_files = len(pickle_files)
        print(f"📊 Found {total_files:,} pickle files" + (f" (processing first {max_files:,})" if max_files else ""))
        
        # Process files
        for i, filepath in enumerate(pickle_files):
            if i % 100 == 0 or i == total_files - 1:
                print(f"📈 Processing file {i+1:,}/{total_files:,} ({((i+1)/total_files)*100:.1f}%)")
            
            self.process_single_file(filepath)
        
        # Compute final summaries
        column_summaries = self.compute_column_summaries()
        
        # Update dataset summary
        self.dataset_summary['all_columns'] = sorted(list(self.dataset_summary['all_columns']))
        self.dataset_summary['total_columns'] = len(self.dataset_summary['all_columns'])
        self.dataset_summary['column_categories'] = self.categorize_columns(self.dataset_summary['all_columns'])
        
        print(f"✅ Analysis complete!")
        print(f"   📊 Processed: {self.dataset_summary['total_files_processed']:,} files")
        print(f"   ❌ Failed: {self.dataset_summary['total_files_failed']:,} files")
        print(f"   👥 Total patients: {self.dataset_summary['total_patients']:,}")
        print(f"   📝 Total observations: {self.dataset_summary['total_observations']:,}")
        print(f"   📋 Total columns: {self.dataset_summary['total_columns']:,}")
        
        return column_summaries
    
    def generate_report(self, column_summaries):
        """Generate comprehensive column summary report"""
        print("📄 Generating column summary report...")
        
        report_lines = []
        report_lines.append("="*100)
        report_lines.append("COMPREHENSIVE COLUMN SUMMARY ANALYSIS REPORT")
        report_lines.append("="*100)
        
        # Dataset Overview
        report_lines.append(f"\n📊 DATASET OVERVIEW")
        report_lines.append(f"Analysis timestamp: {pd.Timestamp.now().isoformat()}")
        report_lines.append(f"Total patients (files): {self.dataset_summary['total_patients']:,}")
        report_lines.append(f"Total observations: {self.dataset_summary['total_observations']:,}")
        report_lines.append(f"Total unique columns: {self.dataset_summary['total_columns']:,}")
        report_lines.append(f"Successfully processed: {self.dataset_summary['total_files_processed']:,}")
        report_lines.append(f"Failed to process: {self.dataset_summary['total_files_failed']:,}")
        
        # Column Categories
        report_lines.append(f"\n📋 COLUMN CATEGORIES")
        for category, columns in self.dataset_summary['column_categories'].items():
            if columns:
                report_lines.append(f"{category.title()}: {len(columns)} columns")
                report_lines.append(f"  {', '.join(columns[:5])}" + (f" ... and {len(columns)-5} more" if len(columns) > 5 else ""))
        
        # All columns by presence
        sorted_by_presence = sorted(column_summaries.items(), 
                                  key=lambda x: x[1]['presence_percentage'], reverse=True)
        
        report_lines.append(f"\n📋 ALL COLUMNS BY PRESENCE ACROSS PATIENTS")
        report_lines.append(f"{'Column':<35} {'Present':<8} {'%':<6} {'Data Type':<15} {'Null %':<8}")
        report_lines.append("-" * 80)
        
        for col, summary in sorted_by_presence:
            dtype_str = str(summary['data_type'][0]) if summary['data_type'] else 'unknown'
            report_lines.append(f"{col[:34]:<35} {summary['files_present']:<8} "
                              f"{summary['presence_percentage']:<6.1f} "
                              f"{dtype_str[:14]:<15} "
                              f"{summary['overall_null_percentage']:<8.1f}")
        
        # Numerical columns summary
        numerical_cols = [(col, summary) for col, summary in column_summaries.items() 
                         if summary['is_numerical'] and 'numerical_stats' in summary]
        
        if numerical_cols:
            report_lines.append(f"\n📊 ALL NUMERICAL COLUMNS SUMMARY")
            report_lines.append(f"{'Column':<35} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10} {'Null %':<8}")
            report_lines.append("-" * 95)
            
            numerical_sorted = sorted(numerical_cols, key=lambda x: x[1]['presence_percentage'], reverse=True)
            for col, summary in numerical_sorted:
                stats = summary['numerical_stats']
                report_lines.append(f"{col[:34]:<35} {stats['mean']:<10.2f} "
                                  f"{stats['std']:<10.2f} {stats['min']:<10.2f} "
                                  f"{stats['max']:<10.2f} {summary['overall_null_percentage']:<8.1f}")
        
        # Categorical columns summary
        categorical_cols = [(col, summary) for col, summary in column_summaries.items() 
                           if summary['is_categorical'] and 'categorical_stats' in summary]
        
        if categorical_cols:
            report_lines.append(f"\n📝 ALL CATEGORICAL COLUMNS SUMMARY")
            report_lines.append(f"{'Column':<35} {'Unique':<8} {'Null %':<8} {'Top 5 Values (count)'}")
            report_lines.append("-" * 120)
            
            categorical_sorted = sorted(categorical_cols, key=lambda x: x[1]['presence_percentage'], reverse=True)
            for col, summary in categorical_sorted:
                stats = summary['categorical_stats']
                top_5_values = []
                for i, cat_info in enumerate(stats['most_common_categories'][:5]):
                    top_5_values.append(f"{cat_info['value']}({cat_info['count']})")
                top_5_str = ", ".join(top_5_values) if top_5_values else "N/A"
                
                report_lines.append(f"{col[:34]:<35} {stats['unique_categories']:<8} "
                                  f"{summary['overall_null_percentage']:<8.1f} {top_5_str}")
        
        # Data quality issues
        report_lines.append(f"\n⚠️  DATA QUALITY ISSUES")
        
        # Check for completely null columns vs columns with some data
        completely_null = []
        high_missing_with_data = []
        
        for col, summary in column_summaries.items():
            if summary['overall_null_percentage'] >= 99.9:  # Essentially 100% null
                has_data = self.column_stats[col]['has_any_non_null']
                if has_data:
                    completely_null.append((col, summary, "HAS_DATA"))
                else:
                    completely_null.append((col, summary, "NO_DATA"))
            elif summary['overall_null_percentage'] > 50:
                high_missing_with_data.append((col, summary))
        
        # Report completely null columns
        if completely_null:
            report_lines.append(f"Columns with ≥99.9% missing data: {len(completely_null)}")
            
            # Separate those with some data vs no data
            with_data = [x for x in completely_null if x[2] == "HAS_DATA"]
            no_data = [x for x in completely_null if x[2] == "NO_DATA"]
            
            if with_data:
                report_lines.append(f"  Columns with some non-null values ({len(with_data)}):")
                for col, summary, _ in sorted(with_data, key=lambda x: x[1]['overall_null_percentage'], reverse=True)[:10]:
                    non_null_count = self.column_stats[col]['non_null_observations']
                    report_lines.append(f"    {col}: {summary['overall_null_percentage']:.1f}% missing ({non_null_count:,} non-null values)")
            
            if no_data:
                report_lines.append(f"  Columns with no non-null values ({len(no_data)}):")
                for col, summary, _ in sorted(no_data, key=lambda x: x[0])[:10]:
                    report_lines.append(f"    {col}: 100.0% missing (completely empty)")
        
        # High missing data (50-99%)
        if high_missing_with_data:
            report_lines.append(f"Columns with 50-99% missing data: {len(high_missing_with_data)}")
            for col, summary in sorted(high_missing_with_data, key=lambda x: x[1]['overall_null_percentage'], reverse=True)[:10]:
                non_null_count = self.column_stats[col]['non_null_observations']
                report_lines.append(f"  {col}: {summary['overall_null_percentage']:.1f}% missing ({non_null_count:,} non-null values)")
        
        # Inconsistent data types
        inconsistent_types = [(col, summary) for col, summary in column_summaries.items() 
                             if len(summary['data_type']) > 1]
        if inconsistent_types:
            report_lines.append(f"Columns with inconsistent data types: {len(inconsistent_types)}")
            for col, summary in inconsistent_types[:10]:
                report_lines.append(f"  {col}: {summary['data_type']}")
        
        # Save report
        report_text = "\n".join(report_lines)
        report_path = f"{self.output_dir}/column_summary_report.txt"
        with open(report_path, 'w') as f:
            f.write(report_text)
        
        print(report_text)
        print(f"\n📄 Report saved to: {report_path}")
    
    def create_visualizations(self, column_summaries):
        """Create column-focused visualizations"""
        print("📈 Creating visualizations...")
        
        # 1. Column presence heatmap
        presence_data = [(col, summary['presence_percentage']) 
                        for col, summary in column_summaries.items()]
        presence_data.sort(key=lambda x: x[1], reverse=True)
        
        # Top 30 columns by presence
        top_cols = presence_data[:30]
        col_names = [col[:20] + '...' if len(col) > 20 else col for col, _ in top_cols]
        presence_pcts = [pct for _, pct in top_cols]
        
        plt.figure(figsize=(12, 10))
        colors = ['red' if pct < 50 else 'orange' if pct < 80 else 'green' for pct in presence_pcts]
        bars = plt.barh(range(len(col_names)), presence_pcts, color=colors, alpha=0.7)
        plt.yticks(range(len(col_names)), col_names)
        plt.xlabel('Presence Percentage Across Patients')
        plt.title('Top 30 Columns by Presence Across All Patients')
        plt.grid(axis='x', alpha=0.3)
        
        # Add percentage labels
        for i, (bar, pct) in enumerate(zip(bars, presence_pcts)):
            plt.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, 
                    f'{pct:.1f}%', va='center', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/plots/column_presence.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ Saved column presence plot")
        
        # 2. Missing data distribution
        missing_data = [(col, summary['overall_null_percentage']) 
                       for col, summary in column_summaries.items()]
        missing_pcts = [pct for _, pct in missing_data]
        
        plt.figure(figsize=(12, 6))
        plt.hist(missing_pcts, bins=20, alpha=0.7, color='coral', edgecolor='black')
        plt.xlabel('Missing Data Percentage')
        plt.ylabel('Number of Columns')
        plt.title('Distribution of Missing Data Across All Columns')
        plt.axvline(np.mean(missing_pcts), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(missing_pcts):.1f}%')
        plt.axvline(np.median(missing_pcts), color='blue', linestyle='--', 
                   label=f'Median: {np.median(missing_pcts):.1f}%')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/plots/missing_data_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ Saved missing data distribution plot")
        
        # 3. Individual plots for each numerical feature
        numerical_cols = [(col, summary) for col, summary in column_summaries.items() 
                         if summary['is_numerical'] and 'numerical_stats' in summary]
        
        if numerical_cols:
            print(f"  📊 Creating individual plots for {len(numerical_cols)} numerical features...")
            
            # Sort by presence percentage
            numerical_sorted = sorted(numerical_cols, key=lambda x: x[1]['presence_percentage'], reverse=True)
            
            for col, summary in numerical_sorted:
                try:
                    stats = summary['numerical_stats']
                    
                    # Create individual plot for this feature
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                    
                    # Left plot: Histogram of actual data
                    if stats['count'] > 1:
                        # Get the actual data values for this column
                        column_values = self.column_stats[col]['values']
                        
                        if len(column_values) > 0:
                            # Create histogram
                            n_bins = min(100, max(10, int(np.sqrt(len(column_values)))))  # Adaptive bin count
                            ax1.hist(column_values, bins=n_bins, alpha=0.7, color='skyblue', 
                                   edgecolor='black', linewidth=0.5)
                            
                            # Add statistical markers
                            ax1.axvline(stats['mean'], color='red', linestyle='--', alpha=0.8, 
                                      linewidth=2, label=f'Mean: {stats["mean"]:.2f}')
                            ax1.axvline(stats['median'], color='green', linestyle='--', alpha=0.8, 
                                      linewidth=2, label=f'Median: {stats["median"]:.2f}')
                            ax1.axvline(stats['q25'], color='orange', linestyle=':', alpha=0.6, 
                                      linewidth=1.5, label=f'Q25: {stats["q25"]:.2f}')
                            ax1.axvline(stats['q75'], color='orange', linestyle=':', alpha=0.6, 
                                      linewidth=1.5, label=f'Q75: {stats["q75"]:.2f}')
                        else:
                            # No data available for histogram
                            ax1.text(0.5, 0.5, 'No data available\nfor histogram', 
                                   transform=ax1.transAxes, ha='center', va='center',
                                   fontsize=12, bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
                    
                    ax1.set_title(f'Histogram: {col}')
                    ax1.set_xlabel('Value')
                    ax1.set_ylabel('Frequency')
                    ax1.legend(fontsize=9)
                    ax1.grid(alpha=0.3)
                    
                    # Right plot: Statistics summary
                    ax2.axis('off')
                    
                    # Create statistics text
                    stats_text = f"""
STATISTICS SUMMARY
{'='*30}

Data Availability:
  • Total observations: {stats['count']:,}
  • Missing data: {summary['overall_null_percentage']:.1f}%
  • Present in {summary['files_present']} files ({summary['presence_percentage']:.1f}%)

Descriptive Statistics:
  • Mean: {stats['mean']:.3f}
  • Median: {stats['median']:.3f}
  • Std Dev: {stats['std']:.3f}
  • Min: {stats['min']:.3f}
  • Max: {stats['max']:.3f}
  • Q25: {stats['q25']:.3f}
  • Q75: {stats['q75']:.3f}

Distribution Properties:
  • Unique values: {stats['unique_count']:,}
  • Zero values: {stats['zero_count']:,}
  • Negative values: {stats['negative_count']:,}
  • Skewness: {stats['skewness']:.3f}
  • Kurtosis: {stats['kurtosis']:.3f}"""
                    
                    if 'outliers_2std' in stats:
                        stats_text += f"""

Outlier Detection:
  • Beyond 2σ: {stats['outliers_2std']:,}
  • Beyond 3σ: {stats['outliers_3std']:,}"""
                    
                    ax2.text(0.05, 0.95, stats_text, transform=ax2.transAxes, fontsize=10,
                            verticalalignment='top', fontfamily='monospace',
                            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
                    
                    plt.suptitle(f'Feature Analysis: {col}', fontsize=14, fontweight='bold')
                    plt.tight_layout()
                    
                    # Save plot with safe filename
                    safe_filename = "".join(c for c in col if c.isalnum() or c in (' ', '-', '_')).rstrip()
                    safe_filename = safe_filename.replace(' ', '_')[:50]  # Limit length
                    plt.savefig(f'{self.output_dir}/plots/individual_features/{safe_filename}.png', 
                               dpi=300, bbox_inches='tight')
                    plt.close()
                    
                except Exception as e:
                    print(f"    ⚠ Error creating plot for {col}: {e}")
                    plt.close()
            
            print(f"  ✓ Saved {len(numerical_cols)} individual feature plots")
        
        print(f"📊 All visualizations saved to: {self.output_dir}/plots/")
    
    def save_column_summaries(self, column_summaries):
        """Save column summaries to JSON"""
        # Prepare data for JSON serialization
        json_data = {
            'dataset_summary': self.dataset_summary,
            'column_summaries': column_summaries,
            'analysis_metadata': {
                'timestamp': pd.Timestamp.now().isoformat(),
                'total_columns_analyzed': len(column_summaries),
                'numerical_columns': sum(1 for s in column_summaries.values() if s['is_numerical']),
                'categorical_columns': sum(1 for s in column_summaries.values() if s['is_categorical'])
            }
        }
        
        # Convert sets to lists for JSON
        json_data['dataset_summary']['all_columns'] = list(json_data['dataset_summary']['all_columns'])
        
        # Save to JSON
        json_path = f"{self.output_dir}/column_summaries.json"
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2, default=str)
        
        print(f"💾 Column summaries saved to: {json_path}")

def main():
    """Main execution function"""
    # Configuration
    pickle_directory = "/hpc/group/kamaleswaranlab/GradyDataset/sepy_processed/grady_supertables/2015/Supertables"
    output_directory = "column_analysis"
    
    # Initialize analyzer
    analyzer = ColumnSummaryAnalyzer(pickle_directory, output_directory)
    
    # Run analysis (set max_files=None to process all files)
    print("🚀 Starting comprehensive column analysis...")
    column_summaries = analyzer.run_analysis(max_files=None)  # Process first 1000 files
    
    # Generate outputs
    analyzer.generate_report(column_summaries)
    analyzer.create_visualizations(column_summaries)
    analyzer.save_column_summaries(column_summaries)
    
    print(f"\n🎉 Column analysis complete! All outputs saved to '{output_directory}/' directory")

if __name__ == "__main__":
    main()
