import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime
from fuzzywuzzy import fuzz, process
from openpyxl import Workbook
from openpyxl.styles import PatternFill, Font, NamedStyle, Alignment
from openpyxl.utils import get_column_letter
from io import BytesIO
import plotly.graph_objects as go

class FinancialInsightsAnalyzer:
    def __init__(self):
        self.key_metrics = {
            'Balance Sheet': {
                'Liquidity': ['Cash And Cash Equivalents', 'Current Assets', 'Current Liabilities'],
                'Asset Structure': ['Total Assets', 'Property Plant And Equipment', 'Intangible Assets'],
                'Capital Structure': ['Total Liabilities', 'Total Stockholder Equity', 'Long Term Debt']
            },
            'Income Statement': {
                'Profitability': ['Gross Profit', 'Operating Income', 'Net Income'],
                'Margins': ['Gross Profit', 'Operating Income', 'Net Income'],
                'Efficiency': ['Research And Development', 'Selling General And Administrative']
            },
            'Cash Flow': {
                'Operations': ['Operating Cash Flow', 'Change In Working Capital'],
                'Investment': ['Capital Expenditure', 'Investments In Property Plant And Equipment'],
                'Financing': ['Total Cash From Financing Activities', 'Dividends Paid']
            }
        }

    def get_company_name(self, ticker):
        """Get full company name from ticker"""
        try:
            company = yf.Ticker(ticker)
            return company.info.get('longName', ticker)
        except:
            return ticker

    def fetch_financial_data(self, tickers, statements):
        """Fetch financial data for multiple companies"""
        results = {}
        for ticker in tickers:
            ticker = ticker.strip().upper()
            try:
                company = yf.Ticker(ticker)
                company_name = self.get_company_name(ticker)
                results[ticker] = {
                    'name': company_name,
                    'data': {}
                }
                
                for statement in statements:
                    if statement == 'Balance Sheet':
                        df = company.balance_sheet
                    elif statement == 'Income Statement':
                        df = company.income_stmt
                    else:
                        df = company.cash_flow
                    
                    if df is not None:
                        results[ticker]['data'][statement] = df
                        
            except Exception as e:
                st.error(f"Error fetching data for {ticker}: {str(e)}")
                
        return results

    def perform_common_size_analysis(self, df, statement_type):
        """Perform common size analysis"""
        try:
            if df is None or df.empty:
                return None
                
            df = df.copy()
            
            if statement_type == 'Balance Sheet':
                base = df.loc['Total Assets']
            elif statement_type == 'Income Statement':
                base = df.loc['Total Revenue']
            else:
                base = df.loc['Operating Cash Flow']
                
            common_size = (df.div(base.abs()) * 100).round(2)
            return common_size
            
        except Exception as e:
            st.error(f"Error in common size analysis: {str(e)}")
            return None

    def generate_insights(self, companies_data, analysis_type="Common Size"):
        """Generate insights from financial data"""
        insights = {
            'Summary': [],
            'Metrics': {},
            'Comparisons': [],
            'Anomalies': []
        }
        
        for statement_type in ['Balance Sheet', 'Income Statement', 'Cash Flow']:
            metrics = {}
            for company, data in companies_data.items():
                if statement_type in data['data']:
                    df = data['data'][statement_type]
                    if analysis_type == "Common Size":
                        df = self.perform_common_size_analysis(df, statement_type)
                    
                    for category, items in self.key_metrics[statement_type].items():
                        for item in items:
                            if item in df.index:
                                metric_key = f"{category} - {item}"
                                if metric_key not in metrics:
                                    metrics[metric_key] = {}
                                metrics[metric_key][company] = df.loc[item].iloc[-1]
            
            for metric, values in metrics.items():
                if len(values) > 1:
                    avg_value = np.mean(list(values.values()))
                    std_dev = np.std(list(values.values()))
                    
                    for company, value in values.items():
                        z_score = (value - avg_value) / std_dev if std_dev != 0 else 0
                        
                        if abs(z_score) > 2:
                            insights['Anomalies'].append({
                                'Metric': metric,
                                'Company': company,
                                'Value': value,
                                'Average': avg_value,
                                'Difference': f"{((value - avg_value) / avg_value * 100):.1f}%"
                            })
                
                insights['Metrics'][metric] = values
        
        return insights

    def create_summary_sheet(self, workbook, insights):
        """Create summary dashboard"""
        worksheet = workbook.create_sheet("Summary Dashboard", 0)
        
        # Define styles
        header_style = NamedStyle(name='header')
        header_style.font = Font(bold=True)
        header_style.fill = PatternFill(start_color='CCE5FF', end_color='CCE5FF', fill_type='solid')
        header_style.alignment = Alignment(horizontal='center')
        
        highlight_style = NamedStyle(name='highlight')
        highlight_style.font = Font(color='006100')
        highlight_style.fill = PatternFill(start_color='C6EFCE', end_color='C6EFCE', fill_type='solid')
        
        warning_style = NamedStyle(name='warning')
        warning_style.font = Font(color='9C0006')
        warning_style.fill = PatternFill(start_color='FFC7CE', end_color='FFC7CE', fill_type='solid')
        
        # Title
        worksheet['A1'] = 'Financial Analysis Summary'
        worksheet['A1'].style = header_style
        worksheet.merge_cells('A1:E1')
        
        # Notable Findings section
        current_row = 3
        worksheet[f'A{current_row}'] = 'Notable Findings'
        worksheet[f'A{current_row}'].style = header_style
        current_row += 1
        
        # Add findings
        for anomaly in insights['Anomalies']:
            cell = worksheet[f'A{current_row}']
            cell.value = f"{anomaly['Company']}: {anomaly['Metric']} is {anomaly['Difference']} different from average"
            cell.style = highlight_style if float(anomaly['Difference'].strip('%')) > 0 else warning_style
            current_row += 1
        
        # Key Metrics section
        current_row += 2
        worksheet[f'A{current_row}'] = 'Key Metrics Comparison'
        worksheet[f'A{current_row}'].style = header_style
        current_row += 1
        
        # Get companies for comparison
        if insights['Metrics']:
            companies = list(next(iter(insights['Metrics'].values())).keys())
            
            # Write company headers
            for i, company in enumerate(companies):
                cell = worksheet[f'{chr(66+i)}{current_row}']
                cell.value = company
                cell.style = header_style
            
            # Write metrics
            for metric, values in insights['Metrics'].items():
                current_row += 1
                worksheet[f'A{current_row}'] = metric
                
                for i, company in enumerate(companies):
                    value = values.get(company, 'N/A')
                    cell = worksheet[f'{chr(66+i)}{current_row}']
                    cell.value = value if value == 'N/A' else f"{value:.1f}%"
                    
                    if value != 'N/A':
                        avg_value = np.mean([v for v in values.values() if v != 'N/A'])
                        if abs(value - avg_value) > avg_value * 0.2:
                            cell.style = highlight_style if value > avg_value else warning_style
        
        # Adjust column widths safely
        for col in ['A', 'B', 'C', 'D', 'E']:  # Adjust for used columns
            max_length = 0
            for cell in worksheet[col]:
                if cell.value:
                    try:
                        max_length = max(max_length, len(str(cell.value)))
                    except:
                        pass
            worksheet.column_dimensions[col].width = max_length + 2

    def export_to_excel(self, data, insights, analysis_type="Original"):
        """Export to Excel with summary dashboard"""
        output = BytesIO()
        
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            workbook = writer.book
            self.create_summary_sheet(workbook, insights)
            
            for ticker, company_data in data.items():
                company_name = company_data['name']
                
                for statement_type, df in company_data['data'].items():
                    if df is not None and not df.empty:
                        sheet_name = f"{company_name[:30]} {statement_type[:15]}"
                        sheet_name = sheet_name.replace('/', '-')
                        
                        if analysis_type == "Common Size":
                            df = self.perform_common_size_analysis(df, statement_type)
                        
                        if df is not None:
                            df.to_excel(writer, sheet_name=sheet_name)
                            worksheet = writer.sheets[sheet_name]
                            for idx, col in enumerate(df.columns):
                                max_length = max(df[col].astype(str).apply(len).max(),
                                              len(str(col)))
                                worksheet.column_dimensions[chr(65 + idx)].width = max_length + 2
        
        return output.getvalue()

class FuzzyMatcher:
    def __init__(self):
        self.default_threshold = 80

    def preprocess_text(self, text):
        if pd.isna(text):
            return ""
        return str(text).lower().strip()

    def perform_fuzzy_match(self, df1, df2, col1, col2, threshold=None):
        if threshold is None:
            threshold = self.default_threshold

        matches = []
        unmatched = []

        df1[col1] = df1[col1].apply(self.preprocess_text)
        df2[col2] = df2[col2].apply(self.preprocess_text)

        lookup_dict = dict(zip(df2[col2], range(len(df2))))

        for idx1, row1 in df1.iterrows():
            best_match = process.extractOne(row1[col1], lookup_dict.keys())
            
            if best_match and best_match[1] >= threshold:
                match_idx = lookup_dict[best_match[0]]
                matches.append({
                    'Source': row1[col1],
                    'Match': best_match[0],
                    'Similarity': best_match[1],
                    'Source_Row': idx1,
                    'Match_Row': match_idx
                })
            else:
                unmatched.append({
                    'Source': row1[col1],
                    'Source_Row': idx1
                })

        matches_df = pd.DataFrame(matches)
        unmatched_df = pd.DataFrame(unmatched)

        return matches_df, unmatched_df

    def export_matches_to_excel(self, matches_df, unmatched_df, df1, df2, col1, col2):
        output = BytesIO()
        
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            if not matches_df.empty:
                detailed_matches = []
                for _, match in matches_df.iterrows():
                    source_data = df1.iloc[match['Source_Row']].to_dict()
                    match_data = df2.iloc[match['Match_Row']].to_dict()
                    
                    row_data = {
                        'Similarity': match['Similarity'],
                        **{f'Source_{k}': v for k, v in source_data.items()},
                        **{f'Match_{k}': v for k, v in match_data.items()}
                    }
                    detailed_matches.append(row_data)
                
                detailed_matches_df = pd.DataFrame(detailed_matches)
                detailed_matches_df.to_excel(writer, sheet_name='Detailed Matches', index=False)

            matches_df.to_excel(writer, sheet_name='Matches Summary', index=False)
            
            if not unmatched_df.empty:
                unmatched_df.to_excel(writer, sheet_name='Unmatched Items', index=False)

        return output.getvalue()
def main():
    st.title("Financial Analysis Suite")

    tab1, tab2 = st.tabs(["Financial Statement Analysis", "Fuzzy Matching"])

    with tab1:
        st.header("Financial Statement Analysis")
        
        analyzer = FinancialInsightsAnalyzer()
        
        tickers_input = st.text_area(
            "Enter ticker symbols (separated by semicolons)",
            "AAPL; MSFT; GOOGL"
        )
        
        tickers = [t.strip() for t in tickers_input.split(";") if t.strip()]
        
        selected_statements = st.multiselect(
            "Select Financial Statements",
            ["Balance Sheet", "Income Statement", "Cash Flow Statement"],
            default=["Balance Sheet"]
        )
        
        analysis_type = st.selectbox(
            "Select Analysis Type",
            ["Original", "Common Size"]
        )
        
        if st.button("Fetch and Analyze"):
            if not tickers:
                st.error("Please enter at least one ticker symbol")
                return
                
            if not selected_statements:
                st.error("Please select at least one statement type")
                return
                
            with st.spinner("Fetching financial data..."):
                results = analyzer.fetch_financial_data(tickers, selected_statements)
                insights = analyzer.generate_insights(results, analysis_type)
                
                for ticker in tickers:
                    if ticker in results:
                        st.header(f"{results[ticker]['name']} ({ticker})")
                        
                        for statement in selected_statements:
                            if statement in results[ticker]['data']:
                                st.subheader(statement)
                                df = results[ticker]['data'][statement]
                                
                                if analysis_type == "Common Size":
                                    df = analyzer.perform_common_size_analysis(df, statement)
                                    
                                if df is not None:
                                    st.dataframe(df)
                                else:
                                    st.warning(f"No {statement} data available for {ticker}")
                
                excel_data = analyzer.export_to_excel(results, insights, analysis_type)
                
                st.download_button(
                    label="Download Complete Analysis",
                    data=excel_data,
                    file_name=f"financial_analysis_{datetime.now().strftime('%Y%m%d')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

    with tab2:
        st.header("Fuzzy Matching Analysis")
        
        matcher = FuzzyMatcher()

        # Create two columns for file upload
        upload_col1, upload_col2 = st.columns(2)
        
        with upload_col1:
            st.subheader("First Dataset")
            file1 = st.file_uploader("Upload first CSV file", type=['csv', 'xlsx'], key="file1")
            
        with upload_col2:
            st.subheader("Second Dataset")
            file2 = st.file_uploader("Upload second CSV file", type=['csv', 'xlsx'], key="file2")

        if file1 and file2:
            try:
                if file1.name.endswith('.csv'):
                    df1 = pd.read_csv(file1)
                else:
                    df1 = pd.read_excel(file1)

                if file2.name.endswith('.csv'):
                    df2 = pd.read_csv(file2)
                else:
                    df2 = pd.read_excel(file2)

                # Create two columns for column selection
                select_col1, select_col2 = st.columns(2)
                
                with select_col1:
                    match_col1 = st.selectbox("Select matching column from first dataset", df1.columns)
                    
                with select_col2:
                    match_col2 = st.selectbox("Select matching column from second dataset", df2.columns)

                threshold = st.slider("Matching threshold (%)", 0, 100, 80)

                if st.button("Perform Matching"):
                    with st.spinner("Performing fuzzy matching..."):
                        matches_df, unmatched_df = matcher.perform_fuzzy_match(
                            df1, df2, match_col1, match_col2, threshold
                        )

                        st.subheader("Matching Results")
                        
                        if not matches_df.empty:
                            st.write(f"Found {len(matches_df)} matches")
                            st.dataframe(matches_df)
                            
                            excel_data = matcher.export_matches_to_excel(
                                matches_df, unmatched_df, df1, df2, match_col1, match_col2
                            )
                            
                            st.download_button(
                                label="Download Matching Results (Excel)",
                                data=excel_data,
                                file_name="fuzzy_matching_results.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )
                        
                        if not unmatched_df.empty:
                            st.subheader("Unmatched Items")
                            st.write(f"Found {len(unmatched_df)} unmatched items")
                            st.dataframe(unmatched_df)

            except Exception as e:
                st.error(f"Error processing files: {str(e)}")

if __name__ == "__main__":
    main()