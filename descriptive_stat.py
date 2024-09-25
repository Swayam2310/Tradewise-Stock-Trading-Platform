import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


def desc_stats(stock_data):
    st.markdown(f"### Descriptive Stat of {st.session_state['ticker']} in Table")
    # Calculate descriptive statistics
    stat = stock_data.describe()
    stat = stat.loc[['min', 'std', '25%', '50%', '75%', 'mean', 'max']]
    return stat


# Function to plot descriptive statistics
def plot_descriptive_stats(stats):
    st.markdown(f"### Descriptive Stat of {st.session_state['ticker']} in Box Chart")

    stats = stats[['Open', 'High', 'Low', 'Close']]
    # Convert DataFrame to long format for Plotly
    long_data = stats.melt(var_name='The Type of Price',
                           value_name='The Amount of Price')

    # Create the box plot
    fig = px.box(long_data,
                 x='The Type of Price',
                 y='The Amount of Price',
                 title='Box Plot of Stock Price')

    # Add annotations
    for attribute in ['Open', 'High', 'Low', 'Close']:
        attribute_data = stats[attribute]

        # Get descriptive statistics
        min_val = attribute_data.loc['min']
        std_val = attribute_data.loc['std']
        p25 = attribute_data.loc['25%']
        median_val = attribute_data.loc['50%']
        p75 = attribute_data.loc['75%']
        mean_val = attribute_data.loc['mean']
        max_val = attribute_data.loc['max']

        # Add text annotations
        fig.add_annotation(
            x=attribute,
            y=min_val,
            text=f"Min: {min_val:.2f}",
            showarrow=True,
            arrowhead=1,
            xref='x',
            yref='y'
        )
        fig.add_annotation(
            x=attribute,
            y=mean_val,
            text=f"Mean: {mean_val:.2f}",
            showarrow=True,
            arrowhead=1,
            xref='x',
            yref='y'
        )
        fig.add_annotation(
            x=attribute,
            y=max_val,
            text=f"Max: {max_val:.2f}",
            showarrow=True,
            arrowhead=1,
            xref='x',
            yref='y'
        )

    # Customize layout
    fig.update_layout(
        xaxis_title="The Type of Price",
        yaxis_title="Price Value",
        boxmode="group"  # Group boxes together for each attribute
    )

    # Display the plot in Streamlit
    st.plotly_chart(fig)


stat = desc_stats(st.session_state['retrieved_data'])
st.dataframe(stat, use_container_width=True)
plot_descriptive_stats(stat)