"""
💰 AI-Powered Customer Pricing Optimization

Transform customer acquisition and pricing strategy with intelligent risk assessment
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from components.data_loader import load_analytics_data, filter_customer_data, get_customer_list
from components.metrics import calculate_load_profile_characteristics, calculate_customer_metrics
from components.visualizations import create_load_trend_chart, create_load_duration_curve, create_seasonal_pattern_chart

st.set_page_config(
    page_title="💰 Customer Pricing Analysis",
    page_icon="💰",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .pricing-header {
        font-size: 2rem;
        font-weight: bold;
        color: #059669;
        margin-bottom: 1rem;
    }
    .tariff-card {
        background-color: #f0fdf4;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #10b981;
        margin-bottom: 1rem;
    }
    .risk-low {
        background-color: #dcfce7;
        border-left-color: #16a34a;
    }
    .risk-medium {
        background-color: #fef3c7;
        border-left-color: #d97706;
    }
    .risk-high {
        background-color: #fee2e2;
        border-left-color: #dc2626;
    }
    .recommendation-box {
        background-color: #eff6ff;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 2px solid #3b82f6;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def calculate_tariff_scenarios(load_profile):
    """Calculate different tariff scenarios"""
    if load_profile is None or load_profile.empty:
        return {}
    
    # Use appropriate load column
    if 'NET_CONSUMPTION_MW' in load_profile.columns:
        daily_load = load_profile.groupby('DATE')['NET_CONSUMPTION_MW'].mean()
    elif 'HALFHOURLY_TOTAL_MW_B1' in load_profile.columns:
        b1_load = load_profile.groupby('DATE')['HALFHOURLY_TOTAL_MW_B1'].mean()
        e1_load = load_profile.groupby('DATE')['HALFHOURLY_TOTAL_MW_E1'].mean() if 'HALFHOURLY_TOTAL_MW_E1' in load_profile.columns else pd.Series(0, index=b1_load.index)
        daily_load = b1_load + e1_load
    else:
        daily_load = pd.Series()
    
    # Check if we have valid data
    if daily_load.empty or len(daily_load) == 0:
        return {}
    
    # Basic calculations (annualized based on daily averages)
    annual_consumption_mwh = daily_load.sum() * (365 / len(daily_load)) * 24  # Convert MW*days to MWh (MW * days * 24h/day)
    peak_demand = daily_load.max()
    average_demand = daily_load.mean()
    load_factor = average_demand / peak_demand if peak_demand > 0 else 0
    
    # Tariff scenarios (based on QLD market data - more realistic rates)
    scenarios = {
        'flat_rate': {
            'name': 'Flat Rate',
            'energy_rate': 0.08,  # $/kWh (80 $/MWh, based on QLD RRP average)
            'daily_charge': 1.20,  # $/day
            'demand_charge': 0,
            'description': 'Simple flat energy rate with daily connection charge'
        },
        'time_of_use': {
            'name': 'Time of Use',
            'peak_rate': 0.12,     # $/kWh (120 $/MWh for peak periods)
            'off_peak_rate': 0.06, # $/kWh (60 $/MWh for off-peak)
            'shoulder_rate': 0.08,  # $/kWh (80 $/MWh for shoulder)
            'daily_charge': 1.20,
            'description': 'Different rates for peak, off-peak, and shoulder periods'
        },
        'demand_tariff': {
            'name': 'Demand Tariff',
            'energy_rate': 0.07,   # $/kWh (70 $/MWh base energy rate)
            'demand_charge': 15.00, # $/kW/month
            'daily_charge': 1.20,
            'description': 'Energy charge plus monthly demand charge based on peak kW'
        }
    }
    
    # Calculate annual costs (simplified)
    costs = {}
    
    # Convert MWh to kWh for cost calculations (energy rates are in $/kWh)
    annual_consumption_kwh = annual_consumption_mwh * 1000
    
    # Flat rate
    flat_annual_cost = (annual_consumption_kwh * scenarios['flat_rate']['energy_rate'] + 
                       scenarios['flat_rate']['daily_charge'] * 365)
    costs['flat_rate'] = flat_annual_cost
    
    # Time of use (assume 30% peak, 40% shoulder, 30% off-peak)
    tou_annual_cost = (annual_consumption_kwh * 0.30 * scenarios['time_of_use']['peak_rate'] +
                      annual_consumption_kwh * 0.40 * scenarios['time_of_use']['shoulder_rate'] +
                      annual_consumption_kwh * 0.30 * scenarios['time_of_use']['off_peak_rate'] +
                      scenarios['time_of_use']['daily_charge'] * 365)
    costs['time_of_use'] = tou_annual_cost
    
    # Demand tariff (peak_demand is in MW, need to convert to kW for demand charge)
    peak_demand_kw = peak_demand * 1000
    demand_annual_cost = (annual_consumption_kwh * scenarios['demand_tariff']['energy_rate'] +
                         peak_demand_kw * scenarios['demand_tariff']['demand_charge'] * 12 +
                         scenarios['demand_tariff']['daily_charge'] * 365)
    costs['demand_tariff'] = demand_annual_cost
    
    # Add costs to scenarios
    for key in scenarios:
        scenarios[key]['annual_cost'] = costs[key]
        scenarios[key]['cost_per_mwh'] = costs[key] / annual_consumption_mwh if annual_consumption_mwh > 0 else 0
    
    return scenarios

def assess_customer_risk(characteristics):
    """Assess customer risk profile"""
    if not characteristics:
        return {'level': 'Unknown', 'factors': [], 'score': 0}
    
    risk_factors = []
    risk_score = 0
    
    # Load factor assessment
    load_factor = characteristics.get('load_factor', 0)
    if load_factor < 0.3:
        risk_factors.append("Low load factor - high demand charges")
        risk_score += 3
    elif load_factor < 0.5:
        risk_factors.append("Moderate load factor")
        risk_score += 1
    
    # Volatility assessment
    volatility = characteristics.get('volatility', 0)
    if volatility > 0.5:
        risk_factors.append("High load volatility - unpredictable consumption")
        risk_score += 2
    elif volatility > 0.3:
        risk_factors.append("Moderate load volatility")
        risk_score += 1
    
    # Peak to average ratio
    peak_ratio = characteristics.get('peak_to_average_ratio', 1)
    if peak_ratio > 3:
        risk_factors.append("High peak-to-average ratio - demand charge exposure")
        risk_score += 2
    elif peak_ratio > 2:
        risk_factors.append("Moderate peak demand")
        risk_score += 1
    
    # Consumption level
    avg_load = characteristics.get('average_load', 0)
    if avg_load > 100:
        risk_factors.append("High consumption customer - price sensitivity")
        risk_score += 1
    
    # Determine risk level
    if risk_score <= 2:
        risk_level = 'Low'
        risk_class = 'risk-low'
    elif risk_score <= 4:
        risk_level = 'Medium'
        risk_class = 'risk-medium'
    else:
        risk_level = 'High'
        risk_class = 'risk-high'
    
    return {
        'level': risk_level,
        'score': risk_score,
        'factors': risk_factors,
        'class': risk_class
    }

def main():
    """Main pricing analysis page"""
    
    st.markdown('<h1 class="pricing-header">💰 Customer Pricing Analysis</h1>', unsafe_allow_html=True)
    st.markdown("**Automated tariff optimization and risk assessment for customers**")
    
    # Sidebar controls
    st.sidebar.title("⚙️ Pricing Configuration")
    
    # Customer selection
    customers = get_customer_list()
    if customers:
        selected_customer = st.sidebar.selectbox(
            "Select Customer",
            customers,
            index=0
        )
    else:
        st.error("No customer data available. Please run data preparation first.")
        return
    
    # Date range selection
    st.sidebar.subheader("Analysis Period")
    
    # Load customer data
    customer_data = filter_customer_data(selected_customer)
    
    if customer_data is None or customer_data.empty:
        st.error(f"No data available for customer: {selected_customer}")
        return
    
    # Date range from available data
    min_date = customer_data['DATE'].min().date()
    max_date = customer_data['DATE'].max().date()
    
    start_date = st.sidebar.date_input(
        "Start Date",
        value=min_date,
        min_value=min_date,
        max_value=max_date
    )
    
    end_date = st.sidebar.date_input(
        "End Date", 
        value=max_date,
        min_value=min_date,
        max_value=max_date
    )
    
    # Filter data by date range
    mask = (customer_data['DATE'].dt.date >= start_date) & (customer_data['DATE'].dt.date <= end_date)
    filtered_data = customer_data[mask]
    
    if filtered_data.empty:
        st.warning("No data available for selected date range.")
        return
    
    # Calculate metrics
    characteristics = calculate_load_profile_characteristics(filtered_data)
    customer_metrics = calculate_customer_metrics(filtered_data)
    risk_assessment = assess_customer_risk(characteristics)
    tariff_scenarios = calculate_tariff_scenarios(filtered_data)
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📊 Load Profile Analysis")
        
        # Load trend chart
        if 'NET_CONSUMPTION_MW' in filtered_data.columns:
            daily_load = filtered_data.groupby('DATE')['NET_CONSUMPTION_MW'].mean().reset_index()
            load_col = 'NET_CONSUMPTION_MW'
        elif 'HALFHOURLY_TOTAL_MW_B1' in filtered_data.columns:
            b1_load = filtered_data.groupby('DATE')['HALFHOURLY_TOTAL_MW_B1'].mean()
            e1_load = filtered_data.groupby('DATE')['HALFHOURLY_TOTAL_MW_E1'].mean() if 'HALFHOURLY_TOTAL_MW_E1' in filtered_data.columns else pd.Series(0, index=b1_load.index)
            daily_load = pd.DataFrame({
                'DATE': b1_load.index,
                'TOTAL_LOAD': b1_load + e1_load
            }).reset_index(drop=True)
            load_col = 'TOTAL_LOAD'
        else:
            daily_load = pd.DataFrame()
            load_col = 'TOTAL_LOAD'
            
        if not daily_load.empty:
            fig = create_load_trend_chart(daily_load.rename(columns={load_col: 'DAILY_TOTAL_MW'}), f"Daily Load Profile - {selected_customer}")
            st.plotly_chart(fig, use_container_width=True)
        
        # Load characteristics table
        st.subheader("📈 Load Characteristics")
        
        if characteristics:
            char_df = pd.DataFrame([
                ["Average Load", f"{characteristics.get('average_load', 0):.2f} MW"],
                ["Peak Load", f"{characteristics.get('peak_load', 0):.2f} MW"],
                ["Base Load", f"{characteristics.get('base_load', 0):.2f} MW"],
                ["Load Factor", f"{characteristics.get('load_factor', 0):.2%}"],
                ["Volatility", f"{characteristics.get('volatility', 0):.2%}"],
                ["Peak/Average Ratio", f"{characteristics.get('peak_to_average_ratio', 0):.2f}"]
            ], columns=["Metric", "Value"])
            
            st.dataframe(char_df, hide_index=True, use_container_width=True)
    
    with col2:
        st.subheader("⚠️ Risk Assessment")
        
        # Risk level display
        risk_class = risk_assessment.get('class', 'risk-low')
        st.markdown(f'''
        <div class="tariff-card {risk_class}">
            <h3>Risk Level: {risk_assessment.get('level', 'Unknown')}</h3>
            <p>Risk Score: {risk_assessment.get('score', 0)}/6</p>
        </div>
        ''', unsafe_allow_html=True)
        
        # Risk factors
        if risk_assessment.get('factors'):
            st.markdown("**Risk Factors:**")
            for factor in risk_assessment['factors']:
                st.markdown(f"• {factor}")
        else:
            st.success("✅ No significant risk factors identified")
        
        # Key metrics
        st.subheader("📊 Key Metrics")
        
        if customer_metrics:
            st.metric("Data Days", customer_metrics.get('data_days', 0))
            st.metric("Average Consumption", f"{customer_metrics.get('avg_daily_consumption', 0):.1f} MW")
            st.metric("Peak Demand", f"{customer_metrics.get('peak_daily_consumption', 0):.1f} MW")
    
    # Tariff scenarios comparison
    st.subheader("💵 Tariff Scenario Analysis")
    
    if tariff_scenarios:
        scenario_data = []
        for key, scenario in tariff_scenarios.items():
            scenario_data.append({
                'Tariff Type': scenario['name'],
                'Annual Cost ($)': f"${scenario['annual_cost']:,.0f}",
                'Cost per MWh ($)': f"${scenario['cost_per_mwh']:.2f}",
                'Description': scenario['description']
            })
        
        scenario_df = pd.DataFrame(scenario_data)
        st.dataframe(scenario_df, hide_index=True, use_container_width=True)
        
        # Cost comparison chart
        costs = [scenario['annual_cost'] for scenario in tariff_scenarios.values()]
        names = [scenario['name'] for scenario in tariff_scenarios.values()]
        
        fig = px.bar(
            x=names,
            y=costs,
            title="Annual Cost Comparison by Tariff Type",
            labels={'x': 'Tariff Type', 'y': 'Annual Cost ($)'}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Best option recommendation
        best_scenario_key = min(tariff_scenarios.keys(), key=lambda k: tariff_scenarios[k]['annual_cost'])
        best_scenario = tariff_scenarios[best_scenario_key]
        
        st.markdown(f'''
        <div class="recommendation-box">
            <h3>💡 Recommended Tariff</h3>
            <h4>{best_scenario['name']}</h4>
            <p><strong>Annual Cost:</strong> ${best_scenario['annual_cost']:,.0f}</p>
            <p><strong>Cost per MWh:</strong> ${best_scenario['cost_per_mwh']:.2f}</p>
            <p>{best_scenario['description']}</p>
        </div>
        ''', unsafe_allow_html=True)
    
    # Load duration curve
    st.subheader("📈 Load Duration Analysis")
    
    # Use appropriate load column for duration analysis
    if 'NET_CONSUMPTION_MW' in filtered_data.columns:
        daily_load_series = filtered_data.groupby('DATE')['NET_CONSUMPTION_MW'].mean()
    elif 'HALFHOURLY_TOTAL_MW_B1' in filtered_data.columns:
        b1_series = filtered_data.groupby('DATE')['HALFHOURLY_TOTAL_MW_B1'].mean()
        e1_series = filtered_data.groupby('DATE')['HALFHOURLY_TOTAL_MW_E1'].mean() if 'HALFHOURLY_TOTAL_MW_E1' in filtered_data.columns else pd.Series(0, index=b1_series.index)
        daily_load_series = b1_series + e1_series
    else:
        daily_load_series = pd.Series()
        
    if not daily_load_series.empty:
        fig = create_load_duration_curve(daily_load_series, f"Load Duration Curve - {selected_customer}")
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        **Load Duration Curve Insights:**
        - Shows load values sorted from highest to lowest
        - Useful for understanding demand patterns and capacity requirements
        - Steep curves indicate high variability; flat curves indicate consistent load
        """)
    
    # Seasonal analysis
    if 'seasonal_variation' in characteristics and characteristics['seasonal_variation']:
        st.subheader("🌅 Seasonal Load Patterns")
        
        seasonal_data = pd.DataFrame(
            list(characteristics['seasonal_variation'].items()),
            columns=['Season', 'Average_Load']
        )
        
        fig = px.bar(
            seasonal_data,
            x='Season',
            y='Average_Load',
            title="Average Load by Season",
            labels={'Average_Load': 'Average Load (MW)'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Export functionality
    st.subheader("📄 Export Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Export Load Profile"):
            # Create export data
            export_data = daily_load.copy()
            csv = export_data.to_csv(index=False)
            st.download_button(
                label="Download Load Profile CSV",
                data=csv,
                file_name=f"{selected_customer}_load_profile.csv",
                mime="text/csv"
            )
    
    with col2:
        if st.button("💰 Export Pricing Analysis"):
            # Create pricing summary
            pricing_summary = pd.DataFrame(scenario_data)
            csv = pricing_summary.to_csv(index=False)
            st.download_button(
                label="Download Pricing Analysis CSV",
                data=csv,
                file_name=f"{selected_customer}_pricing_analysis.csv",
                mime="text/csv"
            )
    
    with col3:
        if st.button("📋 Generate Quote"):
            st.info("Quote generation feature - integrate with your quoting system")
    
    # Footer
    st.markdown("---")
    st.markdown("*Pricing analysis based on historical load patterns. Actual costs may vary based on specific contract terms and market conditions.*")

if __name__ == "__main__":
    main()
