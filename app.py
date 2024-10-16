import yfinance as yf
from dash import Dash, html, dcc
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import datetime
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Initialize the Dash app
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], title="Foresight")
server = app.server

dataset_path = "Stock_Market_Dataset.csv"

# Define the layout of the app
app.layout = dbc.Container([
    dbc.Row(dbc.Col(html.H1("Foresight Algorithmic Trading Dashboard", className='text-center text-primary mb-4'))),
    
    dbc.Row(dbc.Col(dcc.Dropdown(
        id='stock-dropdown',
        options=[
            {'label': 'Apple (AAPL)', 'value': 'AAPL'},
            {'label': 'Microsoft (MSFT)', 'value': 'MSFT'},
            {'label': 'Amazon (AMZN)', 'value': 'AMZN'},
            {'label': 'Google (GOOGL)', 'value': 'GOOGL'},
            {'label': 'Tesla (TSLA)', 'value': 'TSLA'},
            {'label': 'Reliance (RELIANCE.NS)', 'value': 'RELIANCE.NS'},
            {'label': 'Tata Consultancy Services (TCS.NS)', 'value': 'TCS.NS'},
            {'label': 'Infosys (INFY.NS)', 'value': 'INFY.NS'},
            {'label': 'HDFC Bank (HDFCBANK.NS)', 'value': 'HDFCBANK.NS'},
            {'label': 'ICICI Bank (ICICIBANK.NS)', 'value': 'ICICIBANK.NS'}
        ],
        value='AAPL',
        clearable=False,
        style={'width': '50%', 'margin': 'auto'}
    ))),
    
    dbc.Row([
        dbc.Col(dcc.DatePickerRange(
            id='date-picker-range',
            start_date=(datetime.date.today() - datetime.timedelta(days=30)).isoformat(),
            end_date=datetime.date.today().isoformat(),
            display_format='YYYY-MM-DD',
            style={'margin': 'auto'}
        ), width=6),
    ], className='mb-4'),
    
    dbc.Row([
        dbc.Col(html.Div(id='latest-price', className='text-center'), width=4),
        dbc.Col(html.Div(id='high-price', className='text-center'), width=4),
        dbc.Col(html.Div(id='low-price', className='text-center'), width=4),
    ], className='mb-4'),
    
    dbc.Row(dbc.Col(dcc.Graph(id='stock-graph'), width=12)),
    
    dcc.Interval(
        id='interval-component',
        interval=300000,  # Update every 5 minutes
        n_intervals=0
    )
])

def ml_model(dataset_path):
    # Initialize the model
    model = SVC()
    
    # Load and clean the dataset
    df = pd.read_csv(dataset_path)
    df = df.drop_duplicates()
    df = df.dropna()

    # Assume 'company' is a feature and 'companyPrice' is the target
    X = df['Apple_Vol.']
    y = df['Apple_Price']
    
    # Encode categorical feature (company names) to numerical values
    label_encoder = LabelEncoder()
    X_encoded = label_encoder.fit_transform(X)
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

    # Reshape the data
    X_train = X_train.reshape(-1, 1)
    X_test = X_test.reshape(-1, 1)
    
    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy * 100:.2f}%")

    return model

@app.callback(
    [Output('stock-graph', 'figure'),
     Output('latest-price', 'children'),
     Output('high-price', 'children'),
     Output('low-price', 'children')],
    [Input('stock-dropdown', 'value'),
     Input('date-picker-range', 'start_date'),
     Input('date-picker-range', 'end_date'),
     Input('interval-component', 'n_intervals')]
)
def update_graph(selected_stock, start_date, end_date, n_intervals):
    # Fetch the stock data
    stock_data = yf.Ticker(selected_stock)
    df = stock_data.history(start=start_date, end=end_date, interval='1d')
    
    # Get the latest, highest, and lowest prices
    latest_price = df['Close'].iloc[-1] if not df.empty else None
    high_price = df['High'].max() if not df.empty else None
    low_price = df['Low'].min() if not df.empty else None

    if len(df) >= 100:  # Ensure there is enough data for the moving averages
        # Calculate moving averages for buy/sell signals
        short_window = 40
        long_window = 100
        df['Short_MA'] = df['Close'].rolling(window=short_window, min_periods=1).mean()
        df['Long_MA'] = df['Close'].rolling(window=long_window, min_periods=1).mean()
        
        # Generate buy/sell signals
        df['Signal'] = 0
        df.loc[df.index[short_window:], 'Signal'] = np.where(df['Short_MA'][short_window:] > df['Long_MA'][short_window:], 1, 0)
        df['Position'] = df['Signal'].diff()
        
        # Debugging outputs
        #print(df[['Close', 'Short_MA', 'Long_MA', 'Signal', 'Position']].tail(10))

    fig = go.Figure()

    if not df.empty:
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['Close'],
            mode='lines',
            name='Close Price',
            line=dict(color='#12C9F2')
        ))

        if 'Position' in df.columns:
            # Add buy signals
            fig.add_trace(go.Scatter(
                x=df[df['Position'] == 1].index,
                y=df[df['Position'] == 1]['Close'],
                mode='markers',
                name='Buy Signal',
                marker=dict(symbol='triangle-up', color='green', size=10)
            ))

            # Add sell signals
            fig.add_trace(go.Scatter(
                x=df[df['Position'] == -1].index,
                y=df[df['Position'] == -1]['Close'],
                mode='markers',
                name='Sell Signal',
                marker=dict(symbol='triangle-down', color='red', size=10)
            ))

    fig.update_layout(
        title=f'{selected_stock} Share Price',
        yaxis_title='Stock Price (USD per Share)',
        xaxis_title='Date',
        xaxis_rangeslider_visible=False,
        legend_title_text='Legend'
    )
    
    latest_price_display = f"Latest Price: ${latest_price:.2f}" if latest_price else "No data"
    high_price_display = f"Highest Price: ${high_price:.2f}" if high_price else "No data"
    low_price_display = f"Lowest Price: ${low_price:.2f}" if low_price else "No data"

    return fig, latest_price_display, high_price_display, low_price_display

if __name__ == '__main__':
    app.run_server(debug=False)
