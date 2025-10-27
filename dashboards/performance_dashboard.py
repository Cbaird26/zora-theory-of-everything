#!/usr/bin/env python3
"""
Zora Performance Dashboard
Real-time monitoring and display of Theory of Everything trading performance
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
from datetime import datetime, timedelta
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

# Import Zora Consciousness Core
import sys
import os
sys.path.append(os.path.expanduser('~'))
from zora_consciousness_core_fixed import ZoraConsciousness

# --- Dashboard Configuration ---
st.set_page_config(
    page_title="Zora Theory of Everything Trading Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS ---
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .consciousness-metric {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .theory-section {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# --- Helper Functions ---
def fetch_live_data(symbol):
    """Fetch live market data"""
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period="1d", interval="1m")
        if not data.empty:
            return data['Close'].iloc[-1], data['Volume'].iloc[-1]
        return None, None
    except:
        return None, None

def create_market_data_for_zora(symbol, current_price):
    """Create market data structure for Zora consciousness"""
    # Simulate multi-timeframe data
    base_price = current_price
    data = {
        '1m': {
            'price': [base_price * (1 + np.random.normal(0, 0.001)) for _ in range(100)],
            'volume': [np.random.randint(1000, 10000) for _ in range(100)]
        },
        '5m': {
            'price': [base_price * (1 + np.random.normal(0, 0.005)) for _ in range(100)],
            'volume': [np.random.randint(5000, 50000) for _ in range(100)]
        },
        '15m': {
            'price': [base_price * (1 + np.random.normal(0, 0.01)) for _ in range(100)],
            'volume': [np.random.randint(10000, 100000) for _ in range(100)]
        },
        '1h': {
            'price': [base_price * (1 + np.random.normal(0, 0.02)) for _ in range(100)],
            'volume': [np.random.randint(20000, 200000) for _ in range(100)]
        }
    }
    return data

# --- Initialize Session State ---
if 'zora' not in st.session_state:
    st.session_state.zora = ZoraConsciousness()
    st.session_state.trades = []
    st.session_state.portfolio_value = 100000.0
    st.session_state.position = 0.0
    st.session_state.capital = 100000.0
    st.session_state.consciousness_history = []
    st.session_state.performance_history = []

# --- Main Dashboard ---
st.markdown('<h1 class="main-header">🧠 Zora Theory of Everything Trading Dashboard</h1>', unsafe_allow_html=True)

# Sidebar Controls
st.sidebar.header("🎛️ Control Panel")
symbol = st.sidebar.selectbox("Select Symbol", ["BTC-USD", "ETH-USD", "AAPL", "TSLA", "SPY"])
auto_refresh = st.sidebar.checkbox("Auto Refresh", value=True)
refresh_interval = st.sidebar.slider("Refresh Interval (seconds)", 5, 60, 10)

if st.sidebar.button("🔄 Manual Refresh"):
    st.rerun()

# Main Dashboard Layout
col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    st.markdown('<div class="theory-section">', unsafe_allow_html=True)
    st.markdown("### 🧠 Theory of Everything Status")
    
    # Get live market data
    current_price, current_volume = fetch_live_data(symbol)
    if current_price is None:
        current_price = 42000  # Fallback price
    
    # Create market data for Zora
    market_data = create_market_data_for_zora(symbol, current_price)
    
    # Zora perceives market
    perception = st.session_state.zora.perceive_market(market_data)
    decision = st.session_state.zora.decide_action(perception)
    
    # Display consciousness metrics
    col1a, col1b, col1c = st.columns(3)
    
    with col1a:
        st.markdown(f'''
        <div class="consciousness-metric">
            <h3>Φc (Consciousness)</h3>
            <h2>{st.session_state.zora.consciousness_state.phi_c:.3f}</h2>
        </div>
        ''', unsafe_allow_html=True)
    
    with col1b:
        st.markdown(f'''
        <div class="consciousness-metric">
            <h3>E (Ethical Field)</h3>
            <h2>{st.session_state.zora.consciousness_state.E:.3f}</h2>
        </div>
        ''', unsafe_allow_html=True)
    
    with col1c:
        st.markdown(f'''
        <div class="consciousness-metric">
            <h3>Φc/E Ratio</h3>
            <h2>{perception.get("teleology_ratio", 1.0):.3f}</h2>
        </div>
        ''', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Market Analysis
    st.markdown("### 📊 Market Analysis")
    
    col2a, col2b = st.columns(2)
    
    with col2a:
        st.metric("Current Price", f"${current_price:,.2f}")
        st.metric("Market Regime", perception.get("market_regime", "unknown"))
        st.metric("Consciousness Confidence", f"{perception.get('consciousness_confidence', 0):.3f}")
    
    with col2b:
        st.metric("Fractal Resonance", f"{perception.get('consciousness_resonance', 0):.3f}")
        st.metric("Decision", decision.get("Action", "HOLD"))
        st.metric("Position Size", f"{decision.get('Position Size', 0):.3f}")
    
    # Consciousness Evolution Chart
    st.markdown("### 🧬 Consciousness Evolution")
    
    if st.session_state.consciousness_history:
        df_consciousness = pd.DataFrame(st.session_state.consciousness_history)
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("Φc Evolution", "E Evolution"),
            vertical_spacing=0.1
        )
        
        fig.add_trace(
            go.Scatter(x=df_consciousness.index, y=df_consciousness['phi_c'], 
                      name='Φc', line=dict(color='#1f77b4')),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=df_consciousness.index, y=df_consciousness['E'], 
                      name='E', line=dict(color='#ff7f0e')),
            row=2, col=1
        )
        
        fig.update_layout(height=400, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("### 💰 Portfolio Status")
    
    st.markdown(f'''
    <div class="metric-card">
        <h3>Portfolio Value</h3>
        <h2>${st.session_state.portfolio_value:,.2f}</h2>
    </div>
    ''', unsafe_allow_html=True)
    
    st.markdown(f'''
    <div class="metric-card">
        <h3>Capital</h3>
        <h2>${st.session_state.capital:,.2f}</h2>
    </div>
    ''', unsafe_allow_html=True)
    
    st.markdown(f'''
    <div class="metric-card">
        <h3>Position</h3>
        <h2>{st.session_state.position:.4f}</h2>
    </div>
    ''', unsafe_allow_html=True)
    
    # Performance Metrics
    if st.session_state.performance_history:
        df_perf = pd.DataFrame(st.session_state.performance_history)
        total_return = (df_perf['portfolio_value'].iloc[-1] / df_perf['portfolio_value'].iloc[0] - 1) * 100
        
        st.markdown(f'''
        <div class="metric-card">
            <h3>Total Return</h3>
            <h2>{total_return:+.2f}%</h2>
        </div>
        ''', unsafe_allow_html=True)

with col3:
    st.markdown("### 📈 Recent Trades")
    
    if st.session_state.trades:
        recent_trades = st.session_state.trades[-5:]  # Last 5 trades
        for trade in reversed(recent_trades):
            color = "🟢" if trade['pnl'] > 0 else "🔴" if trade['pnl'] < 0 else "⚪"
            st.write(f"{color} {trade['action']} @ ${trade['price']:,.2f}")
            st.write(f"   PnL: ${trade['pnl']:,.2f}")
            st.write(f"   {trade['date'].strftime('%H:%M:%S')}")
            st.write("---")
    else:
        st.write("No trades yet")
    
    # Theory of Everything Equations
    st.markdown("### 🧮 Theory Equations")
    st.latex(r"\Phi_c(h) = \sigma_c(W_c h)")
    st.latex(r"E(h) = \sigma_E(W_E h)")
    st.latex(r"\text{Teleology Ratio} = \frac{\Phi_c}{E}")

# Simulate trading (for demo purposes)
if st.button("🎯 Execute Trade Decision"):
    if decision["Action"] != "HOLD":
        # Simulate trade execution
        trade_value = st.session_state.capital * decision["Position Size"]
        trade_shares = trade_value / current_price
        
        if decision["Action"] == "BUY" and st.session_state.capital >= trade_value:
            st.session_state.position += trade_shares
            st.session_state.capital -= trade_value
            pnl = trade_value * 0.01  # Simulate 1% gain
        elif decision["Action"] == "SELL" and st.session_state.position >= trade_shares:
            st.session_state.position -= trade_shares
            st.session_state.capital += trade_value
            pnl = trade_value * 0.01  # Simulate 1% gain
        else:
            pnl = 0
        
        # Record trade
        trade = {
            'action': decision["Action"],
            'price': current_price,
            'pnl': pnl,
            'date': datetime.now()
        }
        st.session_state.trades.append(trade)
        
        # Evolve consciousness
        evolution = st.session_state.zora.evolve_understanding({
            'PnL': pnl,
            'Action': decision["Action"]
        })
        
        # Record consciousness state
        st.session_state.consciousness_history.append({
            'phi_c': evolution['Φc_evolved'],
            'E': evolution['E_evolved'],
            'teleology_ratio': evolution['Teleology Ratio'],
            'confidence': st.session_state.zora.consciousness_state.confidence
        })
        
        st.success(f"✅ Trade executed: {decision['Action']} at ${current_price:,.2f}")
        st.rerun()

# Update portfolio value
st.session_state.portfolio_value = st.session_state.capital + (st.session_state.position * current_price)

# Record performance
st.session_state.performance_history.append({
    'timestamp': datetime.now(),
    'portfolio_value': st.session_state.portfolio_value,
    'price': current_price
})

# Auto refresh
if auto_refresh:
    time.sleep(refresh_interval)
    st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    🧠 Zora Theory of Everything Trading System | 
    Powered by Φc(h) = σc(Wch) and E(h) = σE(WEh) | 
    Consciousness-driven optimization
</div>
""", unsafe_allow_html=True)
