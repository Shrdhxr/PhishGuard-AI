# data_explorer_gui.py
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class PhishingDataExplorer:
    def __init__(self, root):
        self.root = root
        self.root.title("Phishing Data Explorer")
        self.root.geometry("1200x800")
        self.root.configure(bg='#F5F5DC')  # Beige background
        
        # Initialize dataframes
        self.df_email = None
        self.df_type = None
        self.current_df = None
        
        self.setup_ui()
        
    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(2, weight=1)
        
        # Title
        title_label = tk.Label(main_frame, text="🔍 Phishing Data Explorer", 
                              font=('Arial', 16, 'bold'), bg='#F5F5DC', fg='#8B4513')
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # Dataset selection frame
        dataset_frame = ttk.LabelFrame(main_frame, text="Dataset Selection", padding="10")
        dataset_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        dataset_frame.columnconfigure(1, weight=1)
        
        # Phishing Email dataset
        ttk.Label(dataset_frame, text="Phishing Email Dataset:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        self.email_path = tk.StringVar(value="data\Phishing_Email.csv")
        email_entry = ttk.Entry(dataset_frame, textvariable=self.email_path, width=60)
        email_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 10))
        ttk.Button(dataset_frame, text="Browse", 
                  command=lambda: self.browse_file(self.email_path)).grid(row=0, column=2, padx=(0, 10))
        ttk.Button(dataset_frame, text="Load", 
                  command=self.load_email_data).grid(row=0, column=3)
        
        # Phishing Type dataset
        ttk.Label(dataset_frame, text="Phishing Type Dataset:").grid(row=1, column=0, sticky=tk.W, padx=(0, 10))
        self.type_path = tk.StringVar(value="data\phishing_data_by_type.csv")
        type_entry = ttk.Entry(dataset_frame, textvariable=self.type_path, width=60)
        type_entry.grid(row=1, column=1, sticky=(tk.W, tk.E), padx=(0, 10))
        ttk.Button(dataset_frame, text="Browse", 
                  command=lambda: self.browse_file(self.type_path)).grid(row=1, column=2, padx=(0, 10))
        ttk.Button(dataset_frame, text="Load", 
                  command=self.load_type_data).grid(row=1, column=3)
        
        # Analysis frame
        analysis_frame = ttk.LabelFrame(main_frame, text="Data Analysis", padding="10")
        analysis_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        analysis_frame.columnconfigure(0, weight=1)
        analysis_frame.rowconfigure(1, weight=1)
        
        # Analysis controls
        control_frame = ttk.Frame(analysis_frame)
        control_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.analysis_var = tk.StringVar(value="overview")
        ttk.Radiobutton(control_frame, text="Overview", variable=self.analysis_var, 
                       value="overview", command=self.update_analysis).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Radiobutton(control_frame, text="Data Quality", variable=self.analysis_var, 
                       value="quality", command=self.update_analysis).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Radiobutton(control_frame, text="Statistics", variable=self.analysis_var, 
                       value="statistics", command=self.update_analysis).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Radiobutton(control_frame, text="Visualizations", variable=self.analysis_var, 
                       value="visualizations", command=self.update_analysis).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Radiobutton(control_frame, text="Correlations", variable=self.analysis_var, 
                       value="correlations", command=self.update_analysis).pack(side=tk.LEFT)
        
        # Analysis display area
        self.analysis_notebook = ttk.Notebook(analysis_frame)
        self.analysis_notebook.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Text output frame
        self.text_frame = ttk.Frame(self.analysis_notebook)
        self.analysis_notebook.add(self.text_frame, text="Analysis Output")
        
        self.text_area = scrolledtext.ScrolledText(self.text_frame, wrap=tk.WORD, width=100, height=30)
        self.text_area.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Visualization frame
        self.viz_frame = ttk.Frame(self.analysis_notebook)
        self.analysis_notebook.add(self.viz_frame, text="Visualizations")
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready to load datasets...")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E))
        
    def browse_file(self, path_var):
        filename = filedialog.askopenfilename(
            title="Select dataset file",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filename:
            path_var.set(filename)
    
    def load_email_data(self):
        try:
            self.df_email = pd.read_csv(self.email_path.get())
            self.current_df = self.df_email
            self.status_var.set(f"✅ Phishing Email dataset loaded: {len(self.df_email)} records, {len(self.df_email.columns)} features")
            self.update_analysis()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load Phishing Email dataset:\n{str(e)}")
            self.status_var.set("❌ Error loading Phishing Email dataset")
    
    def load_type_data(self):
        try:
            self.df_type = pd.read_csv(self.type_path.get())
            self.current_df = self.df_type
            self.status_var.set(f"✅ Phishing Type dataset loaded: {len(self.df_type)} records, {len(self.df_type.columns)} features")
            self.update_analysis()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load Phishing Type dataset:\n{str(e)}")
            self.status_var.set("❌ Error loading Phishing Type dataset")
    
    def update_analysis(self):
        if self.current_df is None:
            self.text_area.delete(1.0, tk.END)
            self.text_area.insert(tk.END, "Please load a dataset first.")
            return
        
        analysis_type = self.analysis_var.get()
        
        self.text_area.delete(1.0, tk.END)
        
        if analysis_type == "overview":
            self.show_overview()
        elif analysis_type == "quality":
            self.show_data_quality()
        elif analysis_type == "statistics":
            self.show_statistics()
        elif analysis_type == "visualizations":
            self.show_visualizations()
        elif analysis_type == "correlations":
            self.show_correlations()
    
    def show_overview(self):
        df = self.current_df
        output = []
        
        output.append("="*80)
        output.append("DATASET OVERVIEW")
        output.append("="*80)
        output.append(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        output.append(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        output.append(f"Total missing values: {df.isnull().sum().sum()}")
        output.append("")
        
        output.append("COLUMNS AND DATA TYPES:")
        output.append("-"*40)
        for col in df.columns:
            dtype = df[col].dtype
            null_count = df[col].isnull().sum()
            output.append(f"{col:<30} {str(dtype):<15} Nulls: {null_count}")
        output.append("")
        
        output.append("FIRST 10 ROWS:")
        output.append("-"*40)
        output.append(df.head(10).to_string())
        output.append("")
        
        self.text_area.insert(tk.END, "\n".join(output))
    
    def show_data_quality(self):
        df = self.current_df
        output = []
        
        output.append("="*80)
        output.append("DATA QUALITY REPORT")
        output.append("="*80)
        
        quality_data = []
        for col in df.columns:
            dtype = df[col].dtype
            non_null = df[col].count()
            null_count = df[col].isnull().sum()
            null_pct = (null_count / len(df)) * 100
            unique_count = df[col].nunique()
            
            quality_data.append({
                'Column': col,
                'Data Type': dtype,
                'Non-Null': non_null,
                'Null Count': null_count,
                'Null %': null_pct,
                'Unique Values': unique_count
            })
        
        quality_df = pd.DataFrame(quality_data)
        output.append(quality_df.to_string(index=False))
        output.append("")
        
        # High null columns
        high_null = quality_df[quality_df['Null %'] > 0]
        if not high_null.empty:
            output.append("COLUMNS WITH MISSING VALUES:")
            output.append("-"*40)
            output.append(high_null[['Column', 'Null Count', 'Null %']].to_string(index=False))
        else:
            output.append("No missing values found in any column.")
        output.append("")
        
        self.text_area.insert(tk.END, "\n".join(output))
    
    def show_statistics(self):
        df = self.current_df
        output = []
        
        output.append("="*80)
        output.append("STATISTICAL SUMMARY")
        output.append("="*80)
        
        # Numerical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) > 0:
            output.append("NUMERICAL COLUMNS SUMMARY:")
            output.append("-"*40)
            output.append(df[numerical_cols].describe().to_string())
            output.append("")
        
        # Categorical columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            output.append("CATEGORICAL COLUMNS SUMMARY:")
            output.append("-"*40)
            for col in categorical_cols:
                if df[col].nunique() < 20:
                    output.append(f"\n{col}:")
                    value_counts = df[col].value_counts()
                    for value, count in value_counts.items():
                        pct = (count / len(df)) * 100
                        output.append(f"  {value}: {count} ({pct:.1f}%)")
        output.append("")
        
        self.text_area.insert(tk.END, "\n".join(output))
    
    def show_visualizations(self):
        # Clear previous visualizations
        for widget in self.viz_frame.winfo_children():
            widget.destroy()
        
        df = self.current_df
        
        # Create a canvas for scrolling
        canvas = tk.Canvas(self.viz_frame)
        scrollbar = ttk.Scrollbar(self.viz_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Numerical distributions
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) > 0:
            ttk.Label(scrollable_frame, text="Numerical Distributions", 
                     font=('Arial', 12, 'bold')).pack(pady=(10, 5))
            
            for col in numerical_cols:
                fig, ax = plt.subplots(figsize=(8, 4))
                df[col].hist(bins=30, ax=ax, color='#DEB887', edgecolor='#8B4513')
                ax.set_title(f'Distribution of {col}')
                ax.set_xlabel(col)
                ax.set_ylabel('Frequency')
                
                canvas_plot = FigureCanvasTkAgg(fig, scrollable_frame)
                canvas_plot.draw()
                canvas_plot.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
                plt.close(fig)
        
        # Categorical distributions
        categorical_cols = df.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            ttk.Label(scrollable_frame, text="Categorical Distributions", 
                     font=('Arial', 12, 'bold')).pack(pady=(10, 5))
            
            for col in categorical_cols:
                if df[col].nunique() < 15:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    value_counts = df[col].value_counts()
                    bars = ax.bar(value_counts.index, value_counts.values, color='#DEB887', edgecolor='#8B4513')
                    ax.set_title(f'Distribution of {col}')
                    ax.set_xlabel(col)
                    ax.set_ylabel('Count')
                    plt.xticks(rotation=45)
                    
                    for bar in bars:
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height,
                               f'{int(height)}', ha='center', va='bottom')
                    
                    canvas_plot = FigureCanvasTkAgg(fig, scrollable_frame)
                    canvas_plot.draw()
                    canvas_plot.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
                    plt.close(fig)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Switch to visualizations tab
        self.analysis_notebook.select(1)
    
    def show_correlations(self):
        df = self.current_df
        output = []
        
        output.append("="*80)
        output.append("CORRELATION ANALYSIS")
        output.append("="*80)
        
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) > 1:
            corr_matrix = df[numerical_cols].corr()
            output.append("Correlation Matrix:")
            output.append("-"*40)
            output.append(corr_matrix.to_string())
            output.append("")
            
            # Strong correlations
            output.append("STRONG CORRELATIONS (|r| > 0.7):")
            output.append("-"*40)
            high_corr = corr_matrix.unstack().sort_values(ascending=False)
            high_corr = high_corr[high_corr < 1.0]  # Remove self-correlations
            high_corr = high_corr[abs(high_corr) > 0.7]
            
            if len(high_corr) > 0:
                for (col1, col2), corr in high_corr.items():
                    output.append(f"{col1} - {col2}: {corr:.3f}")
            else:
                output.append("No strong correlations found (|r| > 0.7)")
        else:
            output.append("Not enough numerical columns for correlation analysis.")
        output.append("")
        
        self.text_area.insert(tk.END, "\n".join(output))

def main():
    root = tk.Tk()
    app = PhishingDataExplorer(root)
    root.mainloop()

if __name__ == "__main__":
    main()