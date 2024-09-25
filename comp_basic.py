import streamlit as st
import yfinance as yf
import pandas as pd

# Define industries and sample stocks
industries = {
    'Technology': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'INTC', 'AMD', 'CSCO'],
    'Finance': ['JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'USB', 'TFC', 'STT', 'BK'],
    'Healthcare': ['JNJ', 'PFE', 'MRK', 'ABT', 'LLY', 'GILD', 'AMGN', 'BIIB', 'CVS', 'MDT']
}


def fetch_company_info(ticker):
    """
    This function is to retrieve company's basic information.
    :param ticker: A list of stock symbols
    :return:
    """
    data = {}

    # initializes a yfinance Ticker object for the given stock ticker symbol.
    stock = yf.Ticker(ticker)
    info = stock.info
    data[ticker] = {
        'Name': info.get('shortName', 'N/A'),
        'Sector': info.get('sector', 'N/A'),
        'Industry': info.get('industry', 'N/A'),
        'Country': info.get('country', 'N/A'),
        'City': info.get('city', 'N/A'),
        'State': info.get('state', 'N/A'),
        'Offical_WebSite': info.get('website', 'N/A'),
        'Business Summary': info.get('longBusinessSummary', 'N/A')
    }
    return pd.DataFrame(data).T


# Streamlit app layout
def display_company_info(comp_info):

    # Extract the first row of the DataFrame as simple strings
    name = comp_info['Name'].values[0]
    sector = comp_info['Sector'].values[0]
    industry = comp_info['Industry'].values[0]
    country = comp_info['Country'].values[0]
    city = comp_info['City'].values[0]
    state = comp_info['State'].values[0]
    website = comp_info['Offical_WebSite'].values[0]
    business_summary = comp_info['Business Summary'].values[0]

    # Create a charming layout using Streamlit columns
    st.markdown("### Company Information")

    # Use Streamlit columns to display data side-by-side
    col1, col2 = st.columns(2)

    with col1:
        st.write(f"**Name:** {name}")
        st.write(f"**Sector:** {sector}")
        st.write(f"**Industry:** {industry}")

    with col2:
        st.write(f"**Country:** {country}")
        st.write(f"**City:** {city}")
        st.write(f"**State:** {state}")
        st.write(f"**Website:** {website}")

    st.markdown("### Business Summary")
    st.write(business_summary)


def main():
    st.title("Basic Information")

    # Sidebar for selecting an industry
    selected_industry = st.sidebar.selectbox("Select Industry", list(industries.keys()))

    # Sidebar for choosing a company
    st.session_state['ticker'] = st.sidebar.selectbox("Select Ticker", industries[selected_industry])

    # Fetch and display stock information
    if 'ticker' in st.session_state:
        company_info = fetch_company_info(st.session_state['ticker'])
        display_company_info(company_info)
    else:
        st.warning('No ticker selected')


main()

