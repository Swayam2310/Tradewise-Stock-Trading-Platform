import streamlit as st

# st.Page() is a function in Streamlit used to define a page in a multipage app.
# The first and only required argument defines page source, which can be a Python file or function
# 'title': The title of the page. If it's None(default), the page title(in the browser tab) and label
#   (in the navigation menu) will be inferrred from the filename or callable name in page
comp_basic = st.Page("comp_basic.py",
                     title="Company Info",
                     url_path="home",
                     icon="🏢",
                     default=True)

retrieve_price = st.Page("retrieve_price.py",
                         title="Stock Price",
                         url_path="price",
                         icon="💹")

desc_stat = st.Page("descriptive_stat.py",
                    title="Descriptive Statistic",
                    url_path="desc",
                    icon="0️⃣")

analyze_corr = st.Page("analyze_correlation.py",
                        title="Correlation Analysis",
                        url_path="corr",
                        icon="🌐")

sentiment_price = st.Page("analyze_sentiment.py",
                          title="Sentiment Analysis",
                          url_path="senti",
                          icon="😊")

analyze_seasonality = st.Page("analyze_seasonality.py",
                          title="Seasonality Analysis",
                          url_path="seaso",
                          icon="🌱")
arima_model = st.Page("model.py",  
                      title="ARIMA Predictions",  
                      url_path="prediction",  
                      icon="📈")  

page_list = [comp_basic, retrieve_price, desc_stat, analyze_corr, sentiment_price, analyze_seasonality, model]
# Configure the available pages in a multi-page app.
pg = st.navigation(page_list)

# executing and rendering the multi-page navigation
pg.run()

# Visualize the footer
st.markdown("""
    <style>
        .footer {
            font-size: 12px;
            text-align: center;
            padding: 10px;
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            background-color: #f9f9f9;
            color: #555;
        }
    </style>
    <div class="footer">
        <p>TradeWise &copy; 2024</p>
    </div>
""", unsafe_allow_html=True)
