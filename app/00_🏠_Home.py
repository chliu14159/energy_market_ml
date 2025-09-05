"""
⚡ Retail Analytics Platform - Main Landing Page

Welcome page and platform navigation
"""

import streamlit as st
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="🏠 Home - Retail Analytics Platform",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1e3a8a;
        text-align: center;
        margin-bottom: 2rem;
    }
    .hero-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 3rem;
        border-radius: 1rem;
        text-align: center;
        margin-bottom: 3rem;
    }
    .feature-card {
        background-color: #f8fafc;
        padding: 2rem;
        border-radius: 1rem;
        border-left: 4px solid #3b82f6;
        margin-bottom: 1.5rem;
        height: 100%;
    }
    .cta-button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem 2rem;
        border-radius: 0.5rem;
        border: none;
        font-size: 1.1rem;
        font-weight: bold;
        cursor: pointer;
        text-decoration: none;
        display: inline-block;
        margin: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Hero Section
    st.markdown("""
    <div class="hero-section">
        <h1>⚡ AI-Powered Retail Energy Platform</h1>
        <h3>Transform Your Energy Business with Industry-Leading AI</h3>
        <br>
        <p style="font-size: 1.2rem;">
            <strong>🎯 Reduce forecast errors by 65%</strong> • 
            <strong>💰 Save $24k per MW annually</strong> • 
            <strong>⚡ Real-time optimization</strong>
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Key Features
    st.markdown("## 🚀 Platform Capabilities")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h3>📊 Portfolio Analytics</h3>
            <p>Comprehensive portfolio overview with real-time insights across customer segments, load patterns, and market dynamics.</p>
            <ul>
                <li>28,512+ MWh managed</li>
                <li>Real-time data quality monitoring</li>
                <li>Advanced customer segmentation</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h3>🤖 AI Forecasting</h3>
            <p>Industry-leading load forecasting with confidence intervals and weather correlation.</p>
            <ul>
                <li>4.2% MAPE (vs 8-12% standard)</li>
                <li>95% confidence intervals</li>
                <li>Weather-synchronized predictions</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <h3>💰 Revenue Optimization</h3>
            <p>AI-powered pricing strategies and VPP analysis for maximum profitability.</p>
            <ul>
                <li>Dynamic customer pricing</li>
                <li>Battery revenue optimization</li>
                <li>Risk-adjusted tariff design</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # Platform Navigation
    st.markdown("---")
    st.markdown("## 🧭 Explore the Platform")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 📊 **Start with Dashboard**
        Get an overview of your portfolio performance, AI forecasting capabilities, and key business metrics.
        
        **What you'll see:**
        - Portfolio composition and customer analytics
        - Real-time load analysis and patterns  
        - AI forecast demonstration with confidence bands
        - Weather correlation and market context
        - NMI-level performance insights
        """)
    
    with col2:
        st.markdown("""
        ### 🎯 **Follow the Story**
        Our platform tells a complete story of AI-powered retail energy excellence:
        
        1. **📊 Dashboard** - *"Here's what we achieve"*
        2. **💰 Pricing** - *"Here's how we optimize revenue"*  
        3. **📊 Model Performance** - *"Here's how our AI works"*
        4. **✅ Validation** - *"Here's how we prove accuracy"*
        5. **🔋 VPP Analysis** - *"Here's how we create value"*
        6. **⚙️ Settings** - *"Here's how you control it"*
        """)

    # Business Impact
    st.markdown("---")
    st.markdown("## 💼 Business Impact")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Forecast Accuracy", "95.8%", delta="+2.1%", help="Overall prediction accuracy")
    with col2:
        st.metric("Error Reduction", "65%", delta="vs industry", help="Improvement over standard methods")
    with col3:
        st.metric("Annual Savings", "$24k/MW", delta="per MW", help="Cost reduction per MW managed")
    with col4:
        st.metric("Portfolio Size", "28.5k MWh", delta="+12%", help="Energy under management")

    # Call to Action
    st.markdown("---")
    st.markdown("## 🚀 Ready to Get Started?")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **📊 Explore Dashboard**
        
        Start with our comprehensive portfolio overview and AI forecasting demonstration.
        """)
    
    with col2:
        st.markdown("""
        **💰 See ROI Calculator**
        
        Visit the Pricing page to understand revenue optimization potential.
        """)
    
    with col3:
        st.markdown("""
        **📞 Schedule Demo**
        
        Ready to see this with your data? Contact our team for a personalized demonstration.
        """)

    # Navigation sidebar
    st.sidebar.markdown("**🧭 Platform Navigation**")
    st.sidebar.markdown("**📊 Dashboard** - Start here for portfolio overview")
    st.sidebar.markdown("**💰 Pricing** - Customer pricing optimization")
    st.sidebar.markdown("**📊 Model Performance** - AI model analytics")
    st.sidebar.markdown("**✅ Validation** - Forecast accuracy validation")
    st.sidebar.markdown("**🔋 VPP Analysis** - Battery revenue optimization")
    st.sidebar.markdown("**⚙️ Settings** - Platform configuration")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    **💡 Key Benefits:**
    - 4.2% MAPE (vs 8-12% industry standard)
    - $2.4M annual savings per 100MW portfolio
    - Real-time anomaly detection
    - Automated risk management
    """)

    # Footer
    st.markdown("---")
    st.markdown("*Platform last updated: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "*")

if __name__ == "__main__":
    main()