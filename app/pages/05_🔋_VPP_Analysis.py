"""
🔋 VPP Analysis - Virtual Power Plant Integration Analysis

Battery optimization and revenue stream assessment for VPP opportunities.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from components.data_loader import load_analytics_data

st.set_page_config(
    page_title="🔋 VPP Analysis - Retail Analytics",
    page_icon="🔋", 
    layout="wide"
)

st.title("🔋 Virtual Power Plant Analysis")
st.markdown("**Battery optimization and revenue stream assessment**")

# VPP Impact Demonstration - Clear Before/After Comparison
st.markdown("---")
st.subheader("💡 VPP Impact Demonstration")

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### Before vs After VPP: Load Profile & Market Price Impact")
    
    # Create sample data for clear demonstration
    demo_hours = pd.date_range('2024-08-08 00:00', periods=48, freq='30T')  # 24 hours
    np.random.seed(123)  # Consistent demo data
    
    # Market prices - simulate high price events
    base_price = 80  # $/MWh
    market_prices = []
    for i, hour in enumerate(demo_hours):
        h = hour.hour
        if 16 <= h <= 19:  # Evening peak
            price = base_price + 150 + np.random.normal(0, 20)  # High prices 230-250 $/MWh
        elif 10 <= h <= 15:  # Midday
            price = base_price + 50 + np.random.normal(0, 15)  # Medium prices 130-150 $/MWh  
        elif 19 <= h <= 22:  # Early evening
            price = base_price + 80 + np.random.normal(0, 25)  # Medium-high prices
        else:  # Off-peak
            price = base_price + np.random.normal(0, 10)  # Base prices ~80 $/MWh
        market_prices.append(max(price, 50))  # Floor price
    
    # Load forecast - typical daily pattern
    load_forecast = []
    for i, hour in enumerate(demo_hours):
        h = hour.hour + hour.minute/60
        # Daily load pattern: low at night, peak in evening
        base_load = 150  # MW
        daily_pattern = 50 * np.sin((h - 6) * 2 * np.pi / 24)  # Daily cycle
        evening_peak = 80 if 17 <= h <= 20 else 0  # Evening peak
        load = base_load + daily_pattern + evening_peak + np.random.normal(0, 10)
        load_forecast.append(max(load, 100))  # Minimum load
    
    # VPP Impact: Battery discharge during high prices (16-20h), charge during low prices (23-06h)
    vpp_dispatch = []
    battery_power = 50  # MW
    for i, hour in enumerate(demo_hours):
        h = hour.hour
        price = market_prices[i]
        if price > 180:  # High price threshold - discharge battery
            dispatch = -min(battery_power, 40)  # Discharge (negative = reduce grid load)
        elif price < 100:  # Low price threshold - charge battery  
            dispatch = min(battery_power * 0.6, 30)  # Charge (positive = increase grid load)
        else:
            dispatch = 0  # No dispatch
        vpp_dispatch.append(dispatch)
    
    # Calculate net load after VPP
    net_load_with_vpp = [load + dispatch for load, dispatch in zip(load_forecast, vpp_dispatch)]
    
    # Create the combined comparison chart
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Market price background (fill area for high prices)
    high_price_threshold = 180
    price_colors = ['rgba(255,0,0,0.3)' if p > high_price_threshold else 'rgba(255,0,0,0.1)' for p in market_prices]
    
    # Background price area
    fig.add_trace(
        go.Scatter(x=demo_hours, y=market_prices, name="Market Price", 
                  line=dict(color="darkred", width=2), 
                  fill="tozeroy", fillcolor="rgba(255,0,0,0.1)"),
        secondary_y=True
    )
    
    # Original load forecast (WITHOUT VPP)
    fig.add_trace(
        go.Scatter(x=demo_hours, y=load_forecast, name="Original Load (No VPP)", 
                  line=dict(color="blue", width=4, dash="dash"), 
                  opacity=0.7)
    )
    
    # Optimized load WITH VPP
    fig.add_trace(
        go.Scatter(x=demo_hours, y=net_load_with_vpp, name="Optimized Load (With VPP)", 
                  line=dict(color="green", width=4))
    )
    
    # Add VPP dispatch indicators
    vpp_discharge_times = [demo_hours[i] for i, d in enumerate(vpp_dispatch) if d < 0]
    vpp_discharge_values = [net_load_with_vpp[i] for i, d in enumerate(vpp_dispatch) if d < 0]
    vpp_charge_times = [demo_hours[i] for i, d in enumerate(vpp_dispatch) if d > 0]
    vpp_charge_values = [net_load_with_vpp[i] for i, d in enumerate(vpp_dispatch) if d > 0]
    
    # VPP discharge periods (battery helping during high prices)
    if vpp_discharge_times:
        fig.add_trace(
            go.Scatter(x=vpp_discharge_times, y=vpp_discharge_values, 
                      mode="markers", name="VPP Discharging", 
                      marker=dict(color="orange", size=12, symbol="triangle-down"),
                      hovertemplate="<b>VPP Discharging</b><br>Time: %{x}<br>Load: %{y} MW<extra></extra>")
        )
    
    # VPP charge periods (battery charging during low prices)  
    if vpp_charge_times:
        fig.add_trace(
            go.Scatter(x=vpp_charge_times, y=vpp_charge_values,
                      mode="markers", name="VPP Charging",
                      marker=dict(color="cyan", size=12, symbol="triangle-up"),
                      hovertemplate="<b>VPP Charging</b><br>Time: %{x}<br>Load: %{y} MW<extra></extra>")
        )
    
    # Add high-price period annotations
    high_price_periods = [(i, demo_hours[i], market_prices[i]) for i, p in enumerate(market_prices) if p > high_price_threshold]
    for i, (idx, time, price) in enumerate(high_price_periods[::4]):  # Show every 4th annotation to avoid clutter
        savings = (load_forecast[idx] - net_load_with_vpp[idx]) * price / 1000
        if savings > 10:  # Only annotate significant savings
            fig.add_annotation(
                x=time, y=max(load_forecast[idx], net_load_with_vpp[idx]) + 20,
                text=f"${savings:.0f} saved",
                showarrow=True, arrowhead=2, arrowcolor="red",
                bgcolor="yellow", opacity=0.8,
                font=dict(size=10, color="black")
            )
    
    # Update layout
    fig.update_xaxes(title_text="Time of Day")
    fig.update_yaxes(title_text="Load (MW)")
    fig.update_yaxes(title_text="Market Price ($/MWh)", secondary_y=True, side="right")
    
    fig.update_layout(
        height=500,
        title=dict(
            text="VPP Impact: Load Optimization vs Market Prices<br><sub>Blue dashed = Original Load | Green solid = VPP Optimized | Red area = Market Price</sub>",
            font=dict(size=16)
        ),
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.02,
            xanchor="right", x=1
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("### 💰 Financial Impact Summary")
    
    # Calculate savings
    original_cost = sum(load * price / 1000 for load, price in zip(load_forecast, market_prices))  # $/MWh -> $/kWh
    vpp_cost = sum(load * price / 1000 for load, price in zip(net_load_with_vpp, market_prices))
    daily_savings = original_cost - vpp_cost
    annual_savings = daily_savings * 365
    
    # Peak reduction during high prices
    high_price_hours = [i for i, p in enumerate(market_prices) if p > 180]
    peak_reduction = sum(load_forecast[i] - net_load_with_vpp[i] for i in high_price_hours) / len(high_price_hours) if high_price_hours else 0
    
    st.markdown("#### 📊 Key Benefits")
    st.metric("Daily Cost Savings", f"${daily_savings:.0f}", delta=f"-{daily_savings/original_cost*100:.1f}%")
    st.metric("Annual Savings", f"${annual_savings:,.0f}", help="Based on daily pattern extrapolation")
    st.metric("Peak Load Reduction", f"{peak_reduction:.1f} MW", help="Average reduction during high-price periods")
    st.metric("Revenue per MWh", f"${annual_savings/10:.0f}/MWh/year", help="Assuming 10 MWh battery capacity")
    
    st.markdown("#### 🎯 VPP Strategy")
    st.success("✅ **Discharge** during high prices (4-8 PM)")
    st.info("🔄 **Charge** during low prices (11 PM - 6 AM)") 
    st.warning("⚡ **Peak Shaving** reduces demand charges")
    
    # High-impact periods
    st.markdown("#### ⚡ High-Impact Periods")
    high_price_periods = []
    for i, (time, price) in enumerate(zip(demo_hours, market_prices)):
        if price > 180:  # High price threshold
            savings_this_hour = (load_forecast[i] - net_load_with_vpp[i]) * price / 1000
            high_price_periods.append(f"{time.strftime('%H:%M')}: ${savings_this_hour:.0f} saved")
    
    for period in high_price_periods[:6]:  # Show top 6
        st.write(f"• {period}")
    
    if len(high_price_periods) > 6:
        st.write(f"• ... and {len(high_price_periods)-6} more periods")

st.markdown("---")

# Sidebar configuration
st.sidebar.header("VPP Configuration")

# Battery specifications
st.sidebar.subheader("🔋 Battery Specifications")
battery_capacity = st.sidebar.slider("Battery Capacity (MWh)", 1.0, 100.0, 10.0, 0.5)
battery_power = st.sidebar.slider("Max Power (MW)", 0.5, 50.0, 5.0, 0.25)
battery_efficiency = st.sidebar.slider("Round-trip Efficiency (%)", 70.0, 95.0, 85.0, 1.0) / 100
battery_degradation = st.sidebar.slider("Annual Degradation (%)", 0.5, 5.0, 2.0, 0.1) / 100

# Economic parameters
st.sidebar.subheader("💰 Economic Parameters")
capex_per_mwh = st.sidebar.number_input("CAPEX ($/MWh)", 100000, 1000000, 400000, 10000)
opex_per_mw_year = st.sidebar.number_input("OPEX ($/MW/year)", 5000, 50000, 15000, 1000)
project_lifetime = st.sidebar.slider("Project Lifetime (years)", 5, 25, 15, 1)
discount_rate = st.sidebar.slider("Discount Rate (%)", 3.0, 15.0, 7.0, 0.5) / 100

# Revenue streams
st.sidebar.subheader("📊 Revenue Streams")
enable_arbitrage = st.sidebar.checkbox("Energy Arbitrage", True)
enable_fcas = st.sidebar.checkbox("FCAS Services", True)
enable_capacity = st.sidebar.checkbox("Capacity Market", False)
enable_network = st.sidebar.checkbox("Network Services", True)

# Analysis controls
run_analysis = st.sidebar.button("🔄 Run VPP Analysis")

# Load data
with st.spinner("Loading market and load data..."):
    try:
        df = load_analytics_data()
        if df is None:
            st.error("Unable to load analytics data.")
            st.stop()
            
        # Filter for relevant data
        price_data = df[df['RRP_mean'].notna()].copy()
        # Use appropriate load column
        if 'NET_CONSUMPTION_MW' in df.columns:
            load_data = df[df['CUSTOMER_NAME'].notna() & df['NET_CONSUMPTION_MW'].notna()].copy()
        elif 'HALFHOURLY_TOTAL_MW_B1' in df.columns:
            load_data = df[df['CUSTOMER_NAME'].notna() & df['HALFHOURLY_TOTAL_MW_B1'].notna()].copy()
        else:
            load_data = pd.DataFrame()
        
        if len(price_data) == 0 or len(load_data) == 0:
            st.warning("Insufficient price or load data for VPP analysis.")
            
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

# VPP Analysis Functions
def calculate_arbitrage_revenue(prices, battery_capacity, battery_power, efficiency):
    """Calculate energy arbitrage revenue potential"""
    daily_revenues = []
    
    # Simplified arbitrage calculation
    for date in prices['DATE'].unique():
        daily_prices = prices[prices['DATE'] == date]
        if len(daily_prices) > 0:
            min_price = daily_prices['RRP_min'].min()
            max_price = daily_prices['RRP_max'].max()
            avg_price = daily_prices['RRP_mean'].mean()
            
            # Simple arbitrage: charge at min price, discharge at max price
            # Limited by battery capacity and power rating
            energy_traded = min(battery_capacity, battery_power * 24)  # Max energy per day
            
            if max_price > min_price:
                gross_revenue = energy_traded * (max_price - min_price)
                net_revenue = gross_revenue * efficiency  # Account for round-trip losses
                daily_revenues.append({
                    'date': date,
                    'revenue': net_revenue,
                    'energy_traded': energy_traded,
                    'price_spread': max_price - min_price
                })
    
    return pd.DataFrame(daily_revenues)

def calculate_peak_shaving_benefit(load_data, battery_power, network_tariff=50):
    """Calculate peak shaving and network charge reduction benefits"""
    peak_reductions = []
    
    for customer in load_data['CUSTOMER_NAME'].unique():
        customer_load = load_data[load_data['CUSTOMER_NAME'] == customer]
        
        if len(customer_load) > 0:
            # Identify peak load periods - use appropriate column
            if 'NET_CONSUMPTION_MW' in customer_load.columns:
                peak_load = customer_load['NET_CONSUMPTION_MW'].max()
                avg_load = customer_load['NET_CONSUMPTION_MW'].mean()
            elif 'HALFHOURLY_TOTAL_MW_B1' in customer_load.columns:
                # Use daily aggregation for peak analysis
                daily_b1 = customer_load.groupby('DATE')['HALFHOURLY_TOTAL_MW_B1'].mean()
                daily_e1 = customer_load.groupby('DATE')['HALFHOURLY_TOTAL_MW_E1'].mean() if 'HALFHOURLY_TOTAL_MW_E1' in customer_load.columns else pd.Series(0, index=daily_b1.index)
                daily_total = daily_b1 + daily_e1
                peak_load = daily_total.max()
                avg_load = daily_total.mean()
            else:
                peak_load = 0
                avg_load = 0
            
            # Calculate potential peak reduction
            potential_reduction = min(battery_power, peak_load - avg_load)
            
            if potential_reduction > 0:
                # Monthly network charge savings
                monthly_savings = potential_reduction * network_tariff * 12
                
                peak_reductions.append({
                    'customer': customer,
                    'peak_load': peak_load,
                    'potential_reduction': potential_reduction,
                    'annual_savings': monthly_savings
                })
    
    return pd.DataFrame(peak_reductions)

def calculate_vpp_economics(revenues, costs, discount_rate, lifetime):
    """Calculate VPP project economics"""
    # Calculate NPV and IRR
    cash_flows = [-costs['capex']]  # Initial investment
    
    for year in range(1, lifetime + 1):
        annual_revenue = revenues['annual_total']
        annual_opex = costs['annual_opex']
        degradation_factor = (1 - costs['degradation']) ** (year - 1)
        
        net_cash_flow = (annual_revenue * degradation_factor) - annual_opex
        cash_flows.append(net_cash_flow)
    
    # NPV calculation
    npv = sum([cf / (1 + discount_rate) ** i for i, cf in enumerate(cash_flows)])
    
    # Simple payback period
    cumulative_cf = np.cumsum(cash_flows)
    payback_years = np.where(cumulative_cf > 0)[0]
    payback_period = payback_years[0] if len(payback_years) > 0 else lifetime
    
    return {
        'npv': npv,
        'payback_period': payback_period,
        'cash_flows': cash_flows
    }

# Run analysis when button is clicked
if run_analysis or 'vpp_results' not in st.session_state:
    with st.spinner("Running VPP analysis..."):
        try:
            vpp_results = {}
            
            # Energy Arbitrage Analysis
            if enable_arbitrage and len(price_data) > 0:
                arbitrage_results = calculate_arbitrage_revenue(
                    price_data, battery_capacity, battery_power, battery_efficiency
                )
                vpp_results['arbitrage'] = arbitrage_results
            
            # Peak Shaving Analysis
            if enable_network and len(load_data) > 0:
                peak_shaving_results = calculate_peak_shaving_benefit(
                    load_data, battery_power
                )
                vpp_results['peak_shaving'] = peak_shaving_results
            
            # FCAS Revenue (simplified estimation)
            if enable_fcas:
                fcas_annual_revenue = battery_power * 1000 * 8760 * 0.1  # $0.10/MW/hour estimate
                vpp_results['fcas_revenue'] = fcas_annual_revenue
            
            st.session_state.vpp_results = vpp_results
            
        except Exception as e:
            st.error(f"Error during VPP analysis: {e}")

# Display Results
if 'vpp_results' in st.session_state:
    results = st.session_state.vpp_results
    
    # Revenue Summary
    st.subheader("💰 Revenue Stream Analysis")
    
    col1, col2, col3, col4 = st.columns(4)
    
    total_annual_revenue = 0
    
    with col1:
        if 'arbitrage' in results and len(results['arbitrage']) > 0:
            annual_arbitrage = results['arbitrage']['revenue'].sum() * (365 / len(results['arbitrage']))
            total_annual_revenue += annual_arbitrage
            st.metric("Energy Arbitrage", f"${annual_arbitrage:,.0f}/year")
        else:
            st.metric("Energy Arbitrage", "$0/year")
    
    with col2:
        if 'peak_shaving' in results and len(results['peak_shaving']) > 0:
            annual_peak_shaving = results['peak_shaving']['annual_savings'].sum()
            total_annual_revenue += annual_peak_shaving  
            st.metric("Peak Shaving", f"${annual_peak_shaving:,.0f}/year")
        else:
            st.metric("Peak Shaving", "$0/year")
    
    with col3:
        if 'fcas_revenue' in results:
            annual_fcas = results['fcas_revenue']
            total_annual_revenue += annual_fcas
            st.metric("FCAS Services", f"${annual_fcas:,.0f}/year")
        else:
            st.metric("FCAS Services", "$0/year")
    
    with col4:
        st.metric("Total Revenue", f"${total_annual_revenue:,.0f}/year", 
                 delta=f"${total_annual_revenue/battery_capacity:,.0f}/MWh/year")
    
    # Economic Analysis
    st.subheader("📊 Project Economics")
    
    # Calculate costs
    total_capex = battery_capacity * capex_per_mwh
    annual_opex = battery_power * opex_per_mw_year
    
    costs = {
        'capex': total_capex,
        'annual_opex': annual_opex,
        'degradation': battery_degradation
    }
    
    revenues = {
        'annual_total': total_annual_revenue
    }
    
    economics = calculate_vpp_economics(revenues, costs, discount_rate, project_lifetime)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total CAPEX", f"${total_capex:,.0f}")
        st.metric("Annual OPEX", f"${annual_opex:,.0f}")
    
    with col2:
        st.metric("Net Present Value", f"${economics['npv']:,.0f}")
        payback_color = "normal" if economics['payback_period'] <= 10 else "inverse"
        st.metric("Payback Period", f"{economics['payback_period']:.1f} years")
    
    with col3:
        roi = (total_annual_revenue - annual_opex) / total_capex * 100
        st.metric("Simple ROI", f"{roi:.1f}%")
        
        if economics['npv'] > 0:
            st.success("✅ Project is economically viable")
        else:
            st.error("❌ Project is not economically viable")
    
    # Detailed Analysis Tabs
    st.subheader("📈 Detailed Analysis")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Revenue Breakdown", "Cash Flow Analysis", "Sensitivity Analysis", "Battery Optimization"])
    
    with tab1:
        if 'arbitrage' in results and len(results['arbitrage']) > 0:
            st.markdown("**Energy Arbitrage Performance**")
            
            arbitrage_df = results['arbitrage']
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Daily arbitrage revenue
                fig = px.line(
                    arbitrage_df, 
                    x='date', 
                    y='revenue',
                    title="Daily Arbitrage Revenue",
                    labels={'revenue': 'Revenue ($)', 'date': 'Date'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Price spread vs revenue
                fig = px.scatter(
                    arbitrage_df, 
                    x='price_spread', 
                    y='revenue',
                    title="Price Spread vs Revenue",
                    labels={'price_spread': 'Price Spread ($/MWh)', 'revenue': 'Revenue ($)'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Summary statistics
            st.markdown("**Arbitrage Statistics**")
            stats_cols = st.columns(4)
            
            with stats_cols[0]:
                st.metric("Avg Daily Revenue", f"${arbitrage_df['revenue'].mean():.2f}")
            with stats_cols[1]:
                st.metric("Best Day", f"${arbitrage_df['revenue'].max():.2f}")
            with stats_cols[2]:
                st.metric("Avg Energy Traded", f"{arbitrage_df['energy_traded'].mean():.1f} MWh")
            with stats_cols[3]:
                st.metric("Avg Price Spread", f"${arbitrage_df['price_spread'].mean():.2f}/MWh")
        
        if 'peak_shaving' in results and len(results['peak_shaving']) > 0:
            st.markdown("**Peak Shaving Analysis**")
            
            peak_df = results['peak_shaving']
            
            # Customer peak shaving opportunities
            fig = px.bar(
                peak_df, 
                x='customer', 
                y='annual_savings',
                title="Annual Peak Shaving Savings by Customer",
                labels={'annual_savings': 'Annual Savings ($)', 'customer': 'Customer'}
            )
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
            
            st.dataframe(peak_df, use_container_width=True)
    
    with tab2:
        st.markdown("**Project Cash Flow Analysis**")
        
        cash_flows = economics['cash_flows']
        years = list(range(len(cash_flows)))
        
        # Cash flow chart
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=years,
            y=cash_flows,
            name='Annual Cash Flow',
            marker_color=['red' if cf < 0 else 'green' for cf in cash_flows]
        ))
        
        # Cumulative cash flow
        cumulative_cf = np.cumsum(cash_flows)
        fig.add_trace(go.Scatter(
            x=years,
            y=cumulative_cf,
            mode='lines+markers',
            name='Cumulative Cash Flow',
            yaxis='y2'
        ))
        
        fig.update_layout(
            title="Project Cash Flow Analysis",
            xaxis_title="Year",
            yaxis_title="Annual Cash Flow ($)",
            yaxis2=dict(
                title="Cumulative Cash Flow ($)",
                overlaying='y',
                side='right'
            ),
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Cash flow table
        cf_df = pd.DataFrame({
            'Year': years,
            'Annual Cash Flow': cash_flows,
            'Cumulative Cash Flow': cumulative_cf
        })
        
        st.dataframe(cf_df.style.format({
            'Annual Cash Flow': "${:,.0f}",
            'Cumulative Cash Flow': "${:,.0f}"
        }), use_container_width=True)
    
    with tab3:
        st.markdown("**Sensitivity Analysis**") 
        
        # Parameter sensitivity
        sensitivity_params = {
            'Battery Capacity': (battery_capacity * 0.5, battery_capacity * 1.5),
            'CAPEX': (capex_per_mwh * 0.7, capex_per_mwh * 1.3),
            'Arbitrage Revenue': (total_annual_revenue * 0.5, total_annual_revenue * 2.0),
            'Discount Rate': (discount_rate * 0.5, discount_rate * 2.0)
        }
        
        sensitivity_results = []
        
        for param, (low, high) in sensitivity_params.items():
            # Calculate NPV for low and high values
            # Simplified - in practice would recalculate full model
            base_npv = economics['npv']
            
            if param == 'CAPEX':
                low_npv = base_npv + (capex_per_mwh - low) * battery_capacity
                high_npv = base_npv + (capex_per_mwh - high) * battery_capacity
            elif param == 'Arbitrage Revenue':
                factor_low = low / total_annual_revenue
                factor_high = high / total_annual_revenue
                low_npv = base_npv + (factor_low - 1) * total_annual_revenue * 10  # Simplified
                high_npv = base_npv + (factor_high - 1) * total_annual_revenue * 10
            else:
                # Simplified sensitivity
                low_npv = base_npv * 0.8
                high_npv = base_npv * 1.2
            
            sensitivity_results.extend([
                {'Parameter': param, 'Scenario': 'Low', 'NPV': low_npv},
                {'Parameter': param, 'Scenario': 'Base', 'NPV': base_npv},
                {'Parameter': param, 'Scenario': 'High', 'NPV': high_npv}
            ])
        
        sensitivity_df = pd.DataFrame(sensitivity_results)
        
        # Tornado chart
        fig = px.bar(
            sensitivity_df[sensitivity_df['Scenario'] != 'Base'], 
            x='NPV', 
            y='Parameter',
            color='Scenario',
            orientation='h',
            title="NPV Sensitivity Analysis"
        )
        
        # Add base case line
        fig.add_vline(x=economics['npv'], line_dash="dash", line_color="red", 
                     annotation_text="Base Case")
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.markdown("**Battery Configuration Optimization**")
        
        # Optimization matrix for different battery sizes
        capacity_options = np.arange(5, 51, 5)
        power_options = np.arange(2, 21, 2)
        
        optimization_results = []
        
        for cap in capacity_options[:5]:  # Limit for demo
            for power in power_options[:5]:  # Limit for demo
                # Quick NPV calculation
                capex = cap * capex_per_mwh
                opex = power * opex_per_mw_year
                
                # Estimate revenue (simplified)
                estimated_revenue = min(total_annual_revenue * (cap / battery_capacity), 
                                      total_annual_revenue * (power / battery_power))
                
                # Simple NPV
                annual_net = estimated_revenue - opex
                npv = -capex + sum([annual_net / (1 + discount_rate) ** year 
                                  for year in range(1, project_lifetime + 1)])
                
                optimization_results.append({
                    'Capacity (MWh)': cap,
                    'Power (MW)': power,
                    'CAPEX ($M)': capex / 1e6,
                    'Annual Revenue ($k)': estimated_revenue / 1e3,
                    'NPV ($M)': npv / 1e6
                })
        
        opt_df = pd.DataFrame(optimization_results)
        
        # Heatmap of NPV by capacity and power
        pivot_df = opt_df.pivot(index='Capacity (MWh)', columns='Power (MW)', values='NPV ($M)')
        
        fig = px.imshow(
            pivot_df,
            title="NPV Optimization Matrix (Capacity vs Power)",
            labels={'x': 'Power (MW)', 'y': 'Capacity (MWh)', 'color': 'NPV ($M)'},
            color_continuous_scale='RdYlGn'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Optimal configuration
        best_config = opt_df.loc[opt_df['NPV ($M)'].idxmax()]
        
        st.markdown("**Optimal Configuration**")
        opt_cols = st.columns(4)
        
        with opt_cols[0]:
            st.metric("Optimal Capacity", f"{best_config['Capacity (MWh)']} MWh")
        with opt_cols[1]:
            st.metric("Optimal Power", f"{best_config['Power (MW)']} MW")
        with opt_cols[2]:
            st.metric("Optimal CAPEX", f"${best_config['CAPEX ($M)']:.1f}M")
        with opt_cols[3]:
            st.metric("Optimal NPV", f"${best_config['NPV ($M)']:.1f}M")

else:
    st.info("Click 'Run VPP Analysis' to start the analysis.")

# Export VPP results
if 'vpp_results' in st.session_state:
    st.sidebar.markdown("---")
    st.sidebar.subheader("📋 Export Analysis")
    
    if st.sidebar.button("📊 Export VPP Report"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create summary report
        summary_data = {
            'Analysis_Date': datetime.now(),
            'Battery_Capacity_MWh': battery_capacity,
            'Battery_Power_MW': battery_power,
            'Battery_Efficiency': battery_efficiency,
            'Total_CAPEX': total_capex,
            'Annual_OPEX': annual_opex,
            'Total_Annual_Revenue': total_annual_revenue,
            'NPV': economics['npv'],
            'Payback_Period_Years': economics['payback_period'],
            'Simple_ROI_Percent': (total_annual_revenue - annual_opex) / total_capex * 100
        }
        
        summary_df = pd.DataFrame([summary_data])
        csv = summary_df.to_csv(index=False)
        
        st.sidebar.download_button(
            label="📄 Download VPP Analysis Report",
            data=csv,
            file_name=f"vpp_analysis_{timestamp}.csv",
            mime="text/csv"
        )

st.markdown("---")
st.markdown("*VPP analysis last updated: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "*")