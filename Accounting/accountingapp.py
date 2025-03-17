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
import logging
import time
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set Streamlit page configuration
st.set_page_config(
    page_title="Financial Analysis Suite",
    page_icon="📊",
    layout="wide"
)

class FinancialDataAnalyzer:
    def __init__(self):
        self.statement_types = {
            'Balance Sheet': 'balance_sheet',
            'Income Statement': 'income_stmt',
            'Cash Flow Statement': 'cash_flow'
        }

    def get_company_name(self, ticker):
        """Retrieve the full company name from a ticker symbol with a fallback."""
        try:
            company = yf.Ticker(ticker)
            info = company.info
            return info.get('longName', ticker)
        except Exception as e:
            logger.warning(f"Error getting company name for {ticker}: {str(e)}")
            return ticker

    def find_best_comparison_date(self, companies_data, statement_type):
        """Find the most recent common date across companies' financial statements."""
        all_dates = set()
        company_dates = {}

        for company, data in companies_data.items():
            if statement_type in data['data']:
                df = data['data'][statement_type]
                if df is not None and not df.empty:
                    dates = df.columns
                    company_dates[company] = dates
                    all_dates.update(dates)
                else:
                    logger.warning(f"No data available for {statement_type} of {company}")

        if not all_dates:
            logger.warning(f"No dates available for {statement_type} across companies")
            return {}

        all_dates = sorted(list(all_dates), reverse=True)
        common_date = None
        for date in all_dates:
            companies_with_date = sum(1 for dates in company_dates.values() if date in dates)
            if companies_with_date >= 2:  # At least two companies must share the date
                common_date = date
                break

        best_dates = {}
        for company, dates in company_dates.items():
            if common_date and common_date in dates:
                best_dates[company] = common_date
            elif dates.size > 0:
                best_dates[company] = dates[0]  # Most recent date for the company
            else:
                best_dates[company] = None
                logger.warning(f"No valid dates for {company} in {statement_type}")

        return best_dates

    def get_base_value(self, df, statement_type, date_col):
        """Determine the base value for common size analysis with fallbacks."""
        try:
            if statement_type == 'Balance Sheet':
                for key in ['Total Assets', 'Total assets']:
                    if key in df.index:
                        return df.loc[key, date_col]
                assets_rows = [i for i in df.index if 'asset' in str(i).lower()]
                if assets_rows:
                    return df.loc[assets_rows[0], date_col]
            elif statement_type == 'Income Statement':
                for key in ['Total Revenue', 'Revenue']:
                    if key in df.index:
                        return df.loc[key, date_col]
                revenue_rows = [i for i in df.index if 'revenue' in str(i).lower() or 'sales' in str(i).lower()]
                if revenue_rows:
                    return df.loc[revenue_rows[0], date_col]
            else:  # Cash Flow Statement
                for key in ['Operating Cash Flow', 'Cash from Operations']:
                    if key in df.index:
                        return df.loc[key, date_col]
                cf_rows = [i for i in df.index if 'operating' in str(i).lower() and 'cash' in str(i).lower()]
                if cf_rows:
                    return df.loc[cf_rows[0], date_col]
            logger.warning(f"No base value found for {statement_type} on {date_col}, using 1 as fallback")
            return 1
        except Exception as e:
            logger.error(f"Error getting base value for {statement_type}: {str(e)}")
            return 1

    def create_comparative_statements(self, companies_data, analysis_type="Original"):
        """Generate comparative financial statements for multiple companies."""
        comparative_statements = {}

        for statement_name in self.statement_types:
            best_dates = self.find_best_comparison_date(companies_data, statement_name)
            if not best_dates:
                continue

            comparative_data = {}
            for company, data in companies_data.items():
                if statement_name in data['data'] and company in best_dates and best_dates[company]:
                    df = data['data'][statement_name]
                    if df is not None and not df.empty:
                        date_col = best_dates[company]
                        try:
                            company_data = df[date_col].copy()
                            if analysis_type == "Common Size":
                                base = self.get_base_value(df, statement_name, date_col)
                                if base != 0 and base is not None:
                                    company_data = (company_data / abs(base) * 100).round(2)
                                else:
                                    logger.warning(f"Base value is zero or None for {company} in {statement_name}, using original data")
                            date_str = pd.to_datetime(date_col).strftime('%Y-%m-%d')
                            comparative_data[f"{company} ({date_str})"] = company_data
                        except Exception as e:
                            logger.error(f"Error processing {company} data for {statement_name}: {str(e)}")
                    else:
                        logger.warning(f"Data unavailable for {company} in {statement_name}")

            if comparative_data:
                comparative_statements[statement_name] = pd.DataFrame(comparative_data)
            else:
                logger.warning(f"No comparative data generated for {statement_name}")

        return comparative_statements

    def fetch_financial_data(self, tickers, statements, max_retries=3):
        """Fetch financial data for multiple tickers with retry logic."""
        results = {}
        for ticker in tickers:
            ticker = ticker.strip().upper()
            results[ticker] = {'name': self.get_company_name(ticker), 'data': {}}
            company = yf.Ticker(ticker)
            for statement in statements:
                for attempt in range(max_retries):
                    try:
                        if statement == 'Balance Sheet':
                            df = company.balance_sheet
                        elif statement == 'Income Statement':
                            df = company.income_stmt
                        else:  # Cash Flow Statement
                            df = company.cash_flow
                        if df is not None and not df.empty:
                            df.columns = pd.to_datetime(df.columns)
                            results[ticker]['data'][statement] = df
                            logger.info(f"Fetched {statement} for {ticker} on attempt {attempt + 1}")
                            break
                        else:
                            raise ValueError("Data is empty")
                    except Exception as e:
                        logger.error(f"Attempt {attempt + 1} failed for {ticker} {statement}: {str(e)}")
                        if attempt == max_retries - 1:
                            st.warning(f"Failed to fetch {statement} for {ticker} after {max_retries} attempts")
                        time.sleep(2 ** attempt)  # Exponential backoff
        return results

    def export_to_excel(self, data, analysis_type="Original"):
        """Export financial data to an Excel file."""
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            sheet_written = False
            comparative_statements = self.create_comparative_statements(data, analysis_type)
            
            for statement_type, df in comparative_statements.items():
                sheet_name = f"Comparative {statement_type}"[:31]
                df.to_excel(writer, sheet_name=sheet_name)
                sheet_written = True
                worksheet = writer.sheets[sheet_name]
                for idx, col in enumerate(df.columns, 1):
                    max_length = max(df[col].astype(str).map(len).max(), len(str(col))) + 2
                    worksheet.column_dimensions[get_column_letter(idx + 1)].width = min(max_length, 50)
                worksheet.freeze_panes = 'B2'

            for ticker, company_data in data.items():
                company_name = company_data['name']
                for statement_type, df in company_data['data'].items():
                    if df is not None and not df.empty:
                        sheet_name = f"{company_name[:20]} {statement_type[:10]}".replace('/', '-').replace(':', '-')[:31]
                        export_df = df.copy()
                        if analysis_type == "Common Size":
                            for date_col in export_df.columns:
                                base = self.get_base_value(export_df, statement_type, date_col)
                                if base != 0 and base is not None:
                                    export_df[date_col] = (export_df[date_col] / abs(base) * 100).round(2)
                        export_df.to_excel(writer, sheet_name=sheet_name)
                        sheet_written = True
                        worksheet = writer.sheets[sheet_name]
                        for idx, col in enumerate(export_df.columns, 1):
                            max_length = max(export_df[col].astype(str).map(len).max(), len(str(col))) + 2
                            worksheet.column_dimensions[get_column_letter(idx + 1)].width = min(max_length, 50)
                        worksheet.freeze_panes = 'B2'

            if not sheet_written:
                pd.DataFrame({"Message": ["No data available"]}).to_excel(writer, sheet_name="No Data", index=False)

        return output.getvalue()

class FuzzyMatcher:
    def __init__(self):
        self.default_threshold = 80

    def preprocess_text(self, text):
        """Preprocess text for fuzzy matching."""
        return str(text).lower().strip() if pd.notna(text) else ""

    def perform_fuzzy_match(self, df1, df2, col1, col2, threshold=None):
        """Perform fuzzy matching between two datasets."""
        threshold = threshold if threshold is not None else self.default_threshold
        matches = []
        unmatched = []

        df1_copy = df1.copy()
        df2_copy = df2.copy()
        df1_copy[col1] = df1_copy[col1].apply(self.preprocess_text)
        df2_copy[col2] = df2_copy[col2].apply(self.preprocess_text)

        df1_copy = df1_copy[df1_copy[col1] != ""].reset_index(drop=True)
        df2_copy = df2_copy[df2_copy[col2] != ""].reset_index(drop=True)

        if df1_copy.empty or df2_copy.empty:
            st.warning("One or both datasets are empty after preprocessing.")
            return pd.DataFrame(), pd.DataFrame()

        lookup_dict = dict(zip(df2_copy[col2], range(len(df2_copy))))
        for idx1, row1 in df1_copy.iterrows():
            try:
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
                    unmatched.append({'Source': row1[col1], 'Source_Row': idx1})
            except Exception as e:
                logger.error(f"Error matching row {idx1}: {str(e)}")
                unmatched.append({'Source': row1[col1], 'Source_Row': idx1})

        return pd.DataFrame(matches), pd.DataFrame(unmatched)

    def export_matches_to_excel(self, matches_df, unmatched_df, df1, df2, col1, col2):
        """Export fuzzy matching results to Excel."""
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            if not matches_df.empty:
                detailed_matches = []
                for _, match in matches_df.iterrows():
                    source_row = int(match['Source_Row'])
                    match_row = int(match['Match_Row'])
                    if 0 <= source_row < len(df1) and 0 <= match_row < len(df2):
                        source_data = df1.iloc[source_row].to_dict()
                        match_data = df2.iloc[match_row].to_dict()
                        detailed_matches.append({
                            'Similarity': match['Similarity'],
                            **{f'Source_{k}': v for k, v in source_data.items()},
                            **{f'Match_{k}': v for k, v in match_data.items()}
                        })
                if detailed_matches:
                    pd.DataFrame(detailed_matches).to_excel(writer, sheet_name='Detailed Matches', index=False)

            matches_df.to_excel(writer, sheet_name='Matches Summary', index=False)
            unmatched_df.to_excel(writer, sheet_name='Unmatched Items', index=False)

            df1_export = df1.copy()
            df2_export = df2.copy()
            df1_export['Row Number'] = range(1, len(df1_export) + 1)
            df2_export['Row Number'] = range(1, len(df2_export) + 1)
            df1_export.to_excel(writer, sheet_name='Original Dataset 1', index=False)
            df2_export.to_excel(writer, sheet_name='Original Dataset 2', index=False)

        return output.getvalue()

def read_file(file):
    """Read uploaded file with error handling."""
    try:
        if file.name.lower().endswith('.csv'):
            return pd.read_csv(file)
        elif file.name.lower().endswith(('.xls', '.xlsx')):
            return pd.read_excel(file)
        else:
            st.error(f"Unsupported file format: {file.name}")
            return None
    except Exception as e:
        st.error(f"Error reading file {file.name}: {str(e)}")
        return None

def main():
    st.title("Financial Analysis Suite")
    st.write("Analyze financial statements and perform fuzzy matching on datasets.")

    tab1, tab2 = st.tabs(["Financial Statement Analysis", "Fuzzy Matching"])

    with tab1:
        st.header("Financial Statement Analysis")
        analyzer = FinancialDataAnalyzer()

        tickers_input = st.text_area(
            "Enter ticker symbols (separated by semicolons or commas)",
            "AAPL;MSFT;GOOGL"
        )
        tickers = [t.strip().upper() for t in re.split('[,;]', tickers_input) if t.strip()]
        if not tickers:
            st.error("Please enter at least one ticker symbol (e.g., AAPL;MSFT)")
            return

        st.write("Selected tickers:", ", ".join(tickers))
        selected_statements = st.multiselect(
            "Select Financial Statements",
            ["Balance Sheet", "Income Statement", "Cash Flow Statement"],
            default=["Income Statement"]
        )
        analysis_type = st.selectbox("Select Analysis Type", ["Original", "Common Size"])

        if st.button("Fetch and Analyze"):
            if not selected_statements:
                st.error("Please select at least one statement type.")
                return
            with st.spinner("Fetching financial data..."):
                results = analyzer.fetch_financial_data(tickers, selected_statements)
                has_data = any(company_data['data'] for company_data in results.values())
                failed_tickers = [t for t, d in results.items() if not d['data']]

                if not has_data:
                    st.error("No data fetched. Check ticker symbols or internet connection.")
                    if failed_tickers:
                        st.warning(f"Failed tickers: {', '.join(failed_tickers)}")
                    return

                comparative_statements = analyzer.create_comparative_statements(results, analysis_type)
                if comparative_statements:
                    for statement_type, df in comparative_statements.items():
                        st.subheader(f"Comparative {statement_type}")
                        st.dataframe(df)
                else:
                    st.warning("No comparative statements generated.")

                excel_data = analyzer.export_to_excel(results, analysis_type)
                st.download_button(
                    label="Download Analysis",
                    data=excel_data,
                    file_name=f"financial_analysis_{datetime.now().strftime('%Y%m%d')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
                st.success("Analysis complete!")

    with tab2:
        st.header("Fuzzy Matching Analysis")
        matcher = FuzzyMatcher()

        col1, col2 = st.columns(2)
        with col1:
            file1 = st.file_uploader("Upload first file", type=['csv', 'xlsx', 'xls'], key="file1")
        with col2:
            file2 = st.file_uploader("Upload second file", type=['csv', 'xlsx', 'xls'], key="file2")

        if file1 and file2:
            df1 = read_file(file1)
            df2 = read_file(file2)
            if df1 is not None and df2 is not None and not df1.empty and not df2.empty:
                col1, col2 = st.columns(2)
                with col1:
                    match_col1 = st.selectbox("Column from first dataset", df1.columns)
                with col2:
                    match_col2 = st.selectbox("Column from second dataset", df2.columns)

                threshold = st.slider("Matching threshold (%)", 0, 100, 80)
                if st.button("Perform Matching"):
                    with st.spinner("Performing fuzzy matching..."):
                        matches_df, unmatched_df = matcher.perform_fuzzy_match(df1, df2, match_col1, match_col2, threshold)
                        if not matches_df.empty:
                            st.subheader("Matches")
                            st.write(f"Found {len(matches_df)} matches")
                            st.dataframe(matches_df)
                        else:
                            st.info("No matches found. Try lowering the threshold.")

                        if not unmatched_df.empty:
                            st.subheader("Unmatched Items")
                            st.write(f"Found {len(unmatched_df)} unmatched items")
                            st.dataframe(unmatched_df)

                        excel_data = matcher.export_matches_to_excel(matches_df, unmatched_df, df1, df2, match_col1, match_col2)
                        st.download_button(
                            label="Download Matching Results",
                            data=excel_data,
                            file_name=f"fuzzy_matching_{datetime.now().strftime('%Y%m%d')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                        st.success("Matching complete!")
            elif df1 is not None and df1.empty:
                st.error("First dataset is empty.")
            elif df2 is not None and df2.empty:
                st.error("Second dataset is empty.")

if __name__ == "__main__":
    main()
