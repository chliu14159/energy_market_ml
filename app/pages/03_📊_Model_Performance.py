"""
📊 Model Performance - AI Model Comparison & Selection Guide

Compare forecasting models and get recommendations for your specific use case
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from components.data_loader import load_analytics_data

st.set_page_config(
    page_title="📊 Model Performance",
    page_icon="📊", 
    layout="wide"
)

# Custom CSS for model comparison
st.markdown("""
<style>
    .model-comparison-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 1rem;
        text-align: center;
        margin-bottom: 2rem;
    }
    .model-card {
        background-color: #f8fafc;
        padding: 1.5rem;
        border-radius: 0.75rem;
        border-left: 4px solid #3b82f6;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .best-model {
        border-left-color: #10b981;
        background-color: #ecfdf5;
    }
    .warning-model {
        border-left-color: #f59e0b;
        background-color: #fffbeb;
    }
    .recommendation-box {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def get_model_performance_data():
    """Generate comprehensive model comparison data"""
    
    # Define our model lineup with realistic performance metrics
    models = {
        'LSTM Neural Network': {
            'type': 'Deep Learning',
            'description': 'Advanced neural network optimized for time series forecasting',
            'mape': 3.8,
            'rmse': 2.1,
            'training_time_hours': 24,
            'prediction_speed_ms': 15,
            'data_requirements': 'High (>2 years)',
            'complexity': 'High',
            'interpretability': 'Low',
            'best_for': ['Long-term forecasting', 'Complex patterns', 'Large datasets'],
            'limitations': ['Requires extensive data', 'Black box', 'Long training time'],
            'accuracy_trend': np.array([4.2, 4.0, 3.9, 3.8, 3.8, 3.7]),  # Last 6 months
            'status': 'Production Ready',
            'recommended': True
        },
        'Gradient Boosting (XGBoost)': {
            'type': 'Ensemble Learning', 
            'description': 'Robust ensemble method with excellent performance across scenarios',
            'mape': 4.2,
            'rmse': 2.8,
            'training_time_hours': 8,
            'prediction_speed_ms': 5,
            'data_requirements': 'Medium (>1 year)',
            'complexity': 'Medium',
            'interpretability': 'High',
            'best_for': ['Feature-rich data', 'Interpretable results', 'Fast training'],
            'limitations': ['Prone to overfitting', 'Hyperparameter sensitive'],
            'accuracy_trend': np.array([4.8, 4.5, 4.3, 4.2, 4.1, 4.2]),
            'status': 'Production Ready',
            'recommended': True
        },
        'Random Forest': {
            'type': 'Ensemble Learning',
            'description': 'Stable baseline model with good interpretability',
            'mape': 5.1,
            'rmse': 3.4,
            'training_time_hours': 4,
            'prediction_speed_ms': 8,
            'data_requirements': 'Medium (>6 months)',
            'complexity': 'Low',
            'interpretability': 'High',
            'best_for': ['Baseline forecasting', 'Quick deployment', 'Stable results'],
            'limitations': ['Lower accuracy', 'Memory intensive'],
            'accuracy_trend': np.array([5.5, 5.3, 5.2, 5.1, 5.0, 5.1]),
            'status': 'Production Ready', 
            'recommended': False
        },
        'ARIMA + Weather': {
            'type': 'Statistical + ML',
            'description': 'Traditional time series with weather enhancement',
            'mape': 6.8,
            'rmse': 4.2,
            'training_time_hours': 2,
            'prediction_speed_ms': 3,
            'data_requirements': 'Low (>3 months)',
            'complexity': 'Low',
            'interpretability': 'Very High',
            'best_for': ['Quick deployment', 'Simple patterns', 'Limited data'],
            'limitations': ['Lower accuracy', 'Assumes stationarity'],
            'accuracy_trend': np.array([7.2, 7.0, 6.9, 6.8, 6.7, 6.8]),
            'status': 'Legacy',
            'recommended': False
        },
        'Naive Baseline': {
            'type': 'Statistical',
            'description': 'Simple persistence model for comparison',
            'mape': 12.5,
            'rmse': 8.9,
            'training_time_hours': 0.1,
            'prediction_speed_ms': 1,
            'data_requirements': 'Minimal (1 week)',
            'complexity': 'Very Low',
            'interpretability': 'Very High',
            'best_for': ['Benchmarking', 'Emergency fallback'],
            'limitations': ['Poor accuracy', 'No learning'],
            'accuracy_trend': np.array([12.8, 12.6, 12.5, 12.4, 12.5, 12.5]),
            'status': 'Reference Only',
            'recommended': False
        }
    }
    
    return models

def create_model_comparison_chart(models_data):
    """Create comprehensive model comparison visualization"""
    
    # Prepare data for plotting
    model_names = list(models_data.keys())
    mape_values = [models_data[name]['mape'] for name in model_names]
    rmse_values = [models_data[name]['rmse'] for name in model_names]
    speed_values = [models_data[name]['prediction_speed_ms'] for name in model_names]
    training_time = [models_data[name]['training_time_hours'] for name in model_names]
    
    # Color mapping based on recommendation
    colors = ['#10b981' if models_data[name]['recommended'] else '#6b7280' for name in model_names]
    
    # Create subplot with secondary y-axis
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Accuracy: MAPE (%)', 'Prediction Speed (ms)', 'Training Time (hours)', 'Accuracy vs Speed Trade-off'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # MAPE comparison
    fig.add_trace(
        go.Bar(x=model_names, y=mape_values, name='MAPE (%)', 
               marker_color=colors, text=[f'{v:.1f}%' for v in mape_values], textposition='auto'),
        row=1, col=1
    )
    
    # Prediction Speed
    fig.add_trace(
        go.Bar(x=model_names, y=speed_values, name='Speed (ms)',
               marker_color=colors, text=[f'{v}ms' for v in speed_values], textposition='auto'),
        row=1, col=2
    )
    
    # Training Time
    fig.add_trace(
        go.Bar(x=model_names, y=training_time, name='Training (hrs)',
               marker_color=colors, text=[f'{v}h' for v in training_time], textposition='auto'),
        row=2, col=1
    )
    
    # Accuracy vs Speed scatter
    fig.add_trace(
        go.Scatter(x=speed_values, y=mape_values, mode='markers+text',
                  text=model_names, textposition='top center',
                  marker=dict(size=15, color=colors, line=dict(width=2, color='white')),
                  name='Models'),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        height=700,
        showlegend=False,
        title_text="Model Performance Comparison Matrix",
        title_x=0.5
    )
    
    # Update axes
    fig.update_yaxes(title_text="MAPE (%)", row=1, col=1)
    fig.update_yaxes(title_text="Prediction Speed (ms)", row=1, col=2) 
    fig.update_yaxes(title_text="Training Time (hours)", row=2, col=1)
    fig.update_xaxes(title_text="Prediction Speed (ms)", row=2, col=2)
    fig.update_yaxes(title_text="MAPE (%)", row=2, col=2)
    
    # Rotate x-axis labels for readability
    fig.update_xaxes(tickangle=45)
    
    return fig

def create_accuracy_trend_chart(models_data):
    """Create accuracy trend chart over time"""
    
    months = ['6 months ago', '5 months ago', '4 months ago', '3 months ago', '2 months ago', 'Current']
    
    fig = go.Figure()
    
    for model_name, data in models_data.items():
        if data['recommended']:  # Only show recommended models
            fig.add_trace(go.Scatter(
                x=months,
                y=data['accuracy_trend'],
                mode='lines+markers',
                name=model_name,
                line=dict(width=3),
                marker=dict(size=8)
            ))
    
    # Add benchmark line
    fig.add_hline(y=5.0, line_dash="dash", line_color="orange", 
                 annotation_text="Industry Standard (5% MAPE)")
    fig.add_hline(y=10.0, line_dash="dash", line_color="red",
                 annotation_text="Acceptable Threshold (10% MAPE)")
    
    fig.update_layout(
        title="Model Accuracy Trends Over Time (Recommended Models Only)",
        xaxis_title="Time Period",
        yaxis_title="MAPE (%)",
        height=400,
        hovermode='x unified'
    )
    
    return fig

def get_model_recommendation(use_case):
    """Get model recommendation based on use case"""
    
    recommendations = {
        "High accuracy, long-term planning": {
            "model": "LSTM Neural Network",
            "reasoning": "Best overall accuracy (3.8% MAPE) with excellent long-term pattern recognition. Ideal for strategic planning and portfolio optimization.",
            "setup_time": "2-3 weeks",
            "confidence": "Very High"
        },
        "Balance of accuracy and speed": {
            "model": "Gradient Boosting (XGBoost)", 
            "reasoning": "Excellent balance of 4.2% MAPE accuracy with fast 5ms predictions. Great interpretability for business understanding.",
            "setup_time": "1 week",
            "confidence": "High"
        },
        "Quick deployment, limited data": {
            "model": "Random Forest",
            "reasoning": "Quick to deploy with acceptable 5.1% MAPE. Works well with limited historical data and provides stable results.",
            "setup_time": "2-3 days", 
            "confidence": "Medium"
        },
        "Research and benchmarking": {
            "model": "ARIMA + Weather",
            "reasoning": "Good baseline with high interpretability. Useful for understanding weather impact and model benchmarking.",
            "setup_time": "1 day",
            "confidence": "Medium"
        }
    }
    
    return recommendations.get(use_case, recommendations["Balance of accuracy and speed"])

def main():
    # Header
    st.markdown("""
    <div class="model-comparison-header">
        <h1>📊 AI Model Comparison & Selection Guide</h1>
        <p>Compare forecasting models and get recommendations for your specific use case</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Get model performance data
    models_data = get_model_performance_data()
    
    # Model Selection Guide
    st.subheader("🎯 Which Model Should You Use?")
    
    use_case = st.selectbox(
        "Select your primary use case:",
        ["High accuracy, long-term planning", 
         "Balance of accuracy and speed", 
         "Quick deployment, limited data",
         "Research and benchmarking"],
        help="Choose the option that best matches your forecasting requirements"
    )
    
    recommendation = get_model_recommendation(use_case)
    
    st.markdown(f"""
    <div class="recommendation-box">
        <h3>🏆 Recommended: {recommendation['model']}</h3>
        <p><strong>Why this model:</strong> {recommendation['reasoning']}</p>
        <p><strong>Setup Time:</strong> {recommendation['setup_time']} | <strong>Confidence:</strong> {recommendation['confidence']}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Model Comparison Overview
    st.subheader("📋 Model Performance Comparison")
    
    # Create comparison dataframe
    comparison_data = []
    for name, data in models_data.items():
        comparison_data.append({
            'Model': name,
            'Type': data['type'],
            'MAPE (%)': data['mape'],
            'RMSE (MW)': data['rmse'],
            'Speed (ms)': data['prediction_speed_ms'],
            'Training Time': f"{data['training_time_hours']}h",
            'Data Needs': data['data_requirements'],
            'Interpretability': data['interpretability'],
            'Status': data['status'],
            'Recommended': '✅' if data['recommended'] else '⭕'
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values('MAPE (%)', ascending=True)
    
    # Color code the dataframe
    def highlight_recommended(row):
        if row['Recommended'] == '✅':
            return ['background-color: #ecfdf5'] * len(row)
        elif row['Status'] == 'Legacy':
            return ['background-color: #fffbeb'] * len(row)
        else:
            return [''] * len(row)
    
    st.dataframe(
        comparison_df.style.apply(highlight_recommended, axis=1),
        use_container_width=True,
        hide_index=True
    )
    
    # Performance Charts
    st.subheader("📊 Detailed Performance Analysis")
    
    tab1, tab2, tab3 = st.tabs(["Model Comparison", "Accuracy Trends", "Model Details"])
    
    with tab1:
        # Main comparison chart
        comparison_fig = create_model_comparison_chart(models_data)
        st.plotly_chart(comparison_fig, use_container_width=True)
        
        st.markdown("""
        **Chart Interpretation:**
        - **Green bars**: Recommended models for production use
        - **Gray bars**: Available but not recommended for new deployments
        - **Lower MAPE** and **Lower RMSE** = Better accuracy
        - **Lower prediction speed** = Faster real-time performance
        """)
    
    with tab2:
        # Accuracy trends
        trend_fig = create_accuracy_trend_chart(models_data)
        st.plotly_chart(trend_fig, use_container_width=True)
        
        # Current month performance
        st.subheader("Current Month Performance Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        # Get current best performers
        best_accuracy = min(data['mape'] for data in models_data.values() if data['recommended'])
        best_speed = min(data['prediction_speed_ms'] for data in models_data.values() if data['recommended'])
        
        with col1:
            st.metric("Best Accuracy", f"{best_accuracy}%", delta="-0.1% vs last month")
        with col2:
            st.metric("Best Speed", f"{best_speed}ms", delta="0ms vs last month")
        with col3:
            st.metric("Models in Production", "2", help="LSTM and XGBoost")
        with col4:
            st.metric("Avg Uptime", "99.8%", delta="+0.1% vs last month")
    
    with tab3:
        # Detailed model cards
        st.subheader("🔍 Detailed Model Information")
        
        for model_name, data in models_data.items():
            card_class = "best-model" if data['recommended'] else ("warning-model" if data['status'] == 'Legacy' else "model-card")
            
            st.markdown(f"""
            <div class="{card_class}">
                <h4>{model_name} {('🏆' if data['recommended'] else '⚠️' if data['status'] == 'Legacy' else '')}</h4>
                <p><strong>Type:</strong> {data['type']} | <strong>Status:</strong> {data['status']}</p>
                <p>{data['description']}</p>
                
                <div style="display: flex; gap: 2rem; margin: 1rem 0;">
                    <div><strong>MAPE:</strong> {data['mape']}%</div>
                    <div><strong>RMSE:</strong> {data['rmse']} MW</div>
                    <div><strong>Speed:</strong> {data['prediction_speed_ms']}ms</div>
                    <div><strong>Complexity:</strong> {data['complexity']}</div>
                </div>
                
                <details style="margin-top: 1rem;">
                    <summary style="cursor: pointer; font-weight: bold;">View Details</summary>
                    <div style="margin-top: 0.5rem;">
                        <p><strong>Best for:</strong> {', '.join(data['best_for'])}</p>
                        <p><strong>Limitations:</strong> {', '.join(data['limitations'])}</p>
                        <p><strong>Data Requirements:</strong> {data['data_requirements']}</p>
                        <p><strong>Training Time:</strong> {data['training_time_hours']} hours</p>
                        <p><strong>Interpretability:</strong> {data['interpretability']}</p>
                    </div>
                </details>
            </div>
            """, unsafe_allow_html=True)
    
    # Implementation Guidance
    st.subheader("🚀 Implementation Guidance")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### ✅ Quick Start Recommendations
        
        **For immediate deployment:**
        1. **XGBoost** - Best balance of accuracy and speed
        2. Set up with 1 year of historical data
        3. Expected accuracy: ~4.2% MAPE
        4. Implementation time: 1 week
        
        **For maximum accuracy:**
        1. **LSTM Neural Network** - Best overall performance
        2. Requires 2+ years of high-quality data
        3. Expected accuracy: ~3.8% MAPE  
        4. Implementation time: 2-3 weeks
        """)
    
    with col2:
        st.markdown("""
        ### 📋 Next Steps
        
        **To get started:**
        1. Assess your data quality and quantity
        2. Define accuracy vs speed requirements
        3. Choose model based on recommendation above
        4. Contact our team for implementation support
        
        **Performance monitoring:**
        - Set up automated accuracy tracking
        - Monitor for model drift monthly
        - Retrain quarterly or when MAPE > 6%
        """)
    
    # Contact Information
    st.markdown("---")
    st.info("💬 **Need help choosing?** Contact our AI team for personalized model selection and implementation guidance.")

if __name__ == "__main__":
    main()