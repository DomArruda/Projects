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

class FinancialDataAnalyzer:
    def __init__(self):
        self.statement_types = {
            'Balance Sheet': 'balance_sheet',
            'Income Statement': 'income_stmt',
            'Cash Flow Statement': 'cash_flow'
        }

    def get_company_name(self, ticker):
        """Get full company name from ticker"""
        try:
            company = yf.Ticker(ticker)
            return company.info.get('longName', ticker)
        except:
            return ticker

    def find_best_comparison_date(self, companies_data, statement_type):
        """
        Find the most appropriate comparison date across companies
        """
        all_dates = set()
        company_dates = {}
        
        # Collect all dates and map companies to their available dates
        for company, data in companies_data.items():
            if statement_type in data['data']:
                df = data['data'][statement_type]
                if df is not None and not df.empty:
                    dates = df.columns
                    company_dates[company] = dates
                    all_dates.update(dates)
        
        # Convert to list and sort (most recent first)
        all_dates = sorted(list(all_dates), reverse=True)
        
        if not all_dates:
            return {}
            
        # Find most recent common date
        common_date = None
        for date in all_dates:
            companies_with_date = sum(1 for dates in company_dates.values() if date in dates)
            if companies_with_date > 1:
                common_date = date
                break
        
        # Assign dates for each company
        best_dates = {}
        for company, dates in company_dates.items():
            if common_date and common_date in dates:
                best_dates[company] = common_date
            else:
                # Use most recent date for this company
                best_dates[company] = dates[0] if len(dates) > 0 else None
                
        return best_dates

    def create_comparative_statements(self, companies_data, analysis_type="Original"):
        """
        Create comparative financial statements with companies side by side
        """
        comparative_statements = {}
        
        for statement_type in self.statement_types.keys():
            # Find best comparison dates for this statement
            best_dates = self.find_best_comparison_date(companies_data, statement_type)
            if not best_dates:
                continue
                
            # Create comparative dataframe
            comparative_data = {}
            for company, data in companies_data.items():
                if statement_type in data['data'] and company in best_dates:
                    df = data['data'][statement_type]
                    if df is not None and not df.empty and best_dates[company] is not None:
                        # Extract data for the best date
                        company_data = df[best_dates[company]]
                        
                        # Perform common size analysis if requested
                        if analysis_type == "Common Size":
                            if statement_type == 'Balance Sheet':
                                base = company_data.get('Total Assets', 1)
                            elif statement_type == 'Income Statement':
                                base = company_data.get('Total Revenue', 1)
                            else:  # Cash Flow
                                base = company_data.get('Operating Cash Flow', 1)
                            
                            company_data = (company_data / abs(base) * 100).round(2)
                        
                        # Add date information to column name
                        date_str = pd.to_datetime(best_dates[company]).strftime('%Y-%m-%d')
                        comparative_data[f"{company} ({date_str})"] = company_data
            
            if comparative_data:
                comparative_statements[statement_type] = pd.DataFrame(comparative_data)
        
        return comparative_statements

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

    def export_to_excel(self, data, analysis_type="Original"):
        """Export to Excel with comparative statements"""
        output = BytesIO()
        
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            # Create comparative statements
            comparative_statements = self.create_comparative_statements(data, analysis_type)
            
            # Write comparative statements first
            for statement_type, comp_df in comparative_statements.items():
                sheet_name = f"Comparative {statement_type}"
                comp_df.to_excel(writer, sheet_name=sheet_name)
                
                # Format the sheet
                worksheet = writer.sheets[sheet_name]
                
                # Auto-adjust column widths
                for idx, col in enumerate(comp_df.columns):
                    max_length = max(
                        comp_df.index.astype(str).map(len).max(),
                        len(str(col)),
                        comp_df[col].astype(str).map(len).max()
                    )
                    worksheet.column_dimensions[get_column_letter(idx + 2)].width = max_length + 5
                
                # Freeze panes
                worksheet.freeze_panes = 'B2'
            
            # Write individual company sheets
            for ticker, company_data in data.items():
                company_name = company_data['name']
                
                for statement_type, df in company_data['data'].items():
                    if df is not None and not df.empty:
                        sheet_name = f"{company_name[:30]} {statement_type[:15]}"
                        sheet_name = sheet_name.replace('/', '-')
                        
                        if analysis_type == "Common Size":
                            if statement_type == 'Balance Sheet':
                                base = df.loc['Total Assets']
                            elif statement_type == 'Income Statement':
                                base = df.loc['Total Revenue']
                            else:  # Cash Flow
                                base = df.loc['Operating Cash Flow']
                            df = (df.div(base.abs()) * 100).round(2)


                        df = df.sort_values(df.columns[0], ascending = False)
                        
                        df.to_excel(writer, sheet_name=sheet_name)
                    
                        
                        # Format individual sheets
                        worksheet = writer.sheets[sheet_name]
                        for idx, col in enumerate(df.columns):
                            max_length = max(
                                df.index.astype(str).map(len).max(),
                                len(str(col)),
                                df[col].astype(str).map(len).max()
                            )
                            worksheet.column_dimensions[get_column_letter(idx + 2)].width = max_length + 5
                        
                        worksheet.freeze_panes = 'B2'
        
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

            df1['Row Number'] = df1.index + 1
            df2['Row Number'] = df2.index + 1
            df1.to_excel(writer, sheet_name = 'Original Dataset 1', index = False)
            df2.to_excel(writer, sheet_name = 'Origial Dataset 2', index = False)

        return output.getvalue()

def main():
    st.title("Financial Analysis Suite")

    tab1, tab2 = st.tabs(["Financial Statement Analysis", "Fuzzy Matching"])

    with tab1:
        st.header("Financial Statement Analysis")
        
        analyzer = FinancialDataAnalyzer()
        
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
                
                # Display comparative statements
                comparative_statements = analyzer.create_comparative_statements(results, analysis_type)
                
                for statement_type, comp_df in comparative_statements.items():
                    st.subheader(f"Comparative {statement_type}")
                    st.dataframe(comp_df)
                
                # Create Excel export
                excel_data = analyzer.export_to_excel(results, analysis_type)
                
                st.download_button(
                    label="Download Complete Analysis",
                    data=excel_data,
                    file_name=f"financial_analysis_{datetime.now().strftime('%Y%m%d')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

    with tab2:
        st.header("Fuzzy Matching Analysis")
        
        matcher = FuzzyMatcher()

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
