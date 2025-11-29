"""
Agent Sentinel - Command Center Dashboard
MumbaiHacks 2025 - Complete Streamlit Interface
"""
import streamlit as st
import requests
import pandas as pd
import time
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px

# Configuration
API_BASE = "http://localhost:8000"

# Page config
st.set_page_config(
    page_title="Agent Sentinel - Command Center",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #3b82f6, #06b6d4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #3b82f6;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .alert-critical {
        background: linear-gradient(135deg, #7f1d1d 0%, #450a0a 100%);
        border-left: 4px solid #dc2626;
        animation: pulse 2s infinite;
    }
    .alert-high {
        background: linear-gradient(135deg, #7c2d12 0%, #431407 100%);
        border-left: 4px solid #ea580c;
    }
    .alert-medium {
        background: linear-gradient(135deg, #713f12 0%, #3f2408 100%);
        border-left: 4px solid #eab308;
    }
    .alert-low {
        background: linear-gradient(135deg, #14532d 0%, #052e16 100%);
        border-left: 4px solid #22c55e;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }
    .agent-log {
        background: #000;
        color: #0f0;
        font-family: 'Courier New', monospace;
        padding: 1rem;
        border-radius: 5px;
        max-height: 400px;
        overflow-y: auto;
        font-size: 0.85rem;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #3b82f6, #06b6d4);
        color: white;
        font-weight: bold;
        border: none;
        padding: 0.75rem;
        border-radius: 5px;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 0 20px rgba(59, 130, 246, 0.5);
    }
    .threat-meter {
        height: 30px;
        background: #1e293b;
        border-radius: 15px;
        overflow: hidden;
        position: relative;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
    }
</style>
""", unsafe_allow_html=True)

# Utility Functions
def fetch_stats():
    try:
        response = requests.get(f"{API_BASE}/stats", timeout=5)
        return response.json()
    except:
        return None

def fetch_active_alerts():
    try:
        response = requests.get(f"{API_BASE}/active-alerts", timeout=5)
        return response.json()
    except:
        return []

def fetch_analysis_history(limit=20):
    try:
        response = requests.get(f"{API_BASE}/analysis-history?limit={limit}", timeout=5)
        return response.json()
    except:
        return []

def fetch_time_comparison():
    try:
        response = requests.get(f"{API_BASE}/time-comparison", timeout=5)
        return response.json()
    except:
        return None

def simulate_crisis(scenario):
    try:
        response = requests.post(
            f"{API_BASE}/simulate-crisis",
            json={"scenario": scenario},
            timeout=30
        )
        return response.json()
    except Exception as e:
        st.error(f"Simulation failed: {e}")
        return None

def approve_alert(news_id, approved_by):
    try:
        response = requests.post(
            f"{API_BASE}/approve-alert/{news_id}?approved_by={approved_by}",
            timeout=5
        )
        return response.json()
    except Exception as e:
        st.error(f"Approval failed: {e}")
        return None

def analyze_custom_news(headline, content):
    try:
        response = requests.post(
            f"{API_BASE}/analyze",
            json={
                "headline": headline,
                "content": content,
                "enable_counter_narrative": True
            },
            timeout=60
        )
        return response.json()
    except Exception as e:
        st.error(f"Analysis failed: {e}")
        return None

def get_alert_color(level):
    colors = {
        "LOW": "#22c55e",
        "MEDIUM": "#eab308",
        "HIGH": "#ea580c",
        "CRITICAL": "#dc2626"
    }
    return colors.get(level, "#64748b")

def get_alert_emoji(level):
    emojis = {
        "LOW": "✅",
        "MEDIUM": "⚠️",
        "HIGH": "🔥",
        "CRITICAL": "🚨"
    }
    return emojis.get(level, "ℹ️")

# Header
st.markdown('<div class="main-header">🛡️ AGENT SENTINEL</div>', unsafe_allow_html=True)
st.markdown('<p style="text-align:center; color:#64748b; font-size:1.1rem;">Autonomous Defense System Against Misinformation • MumbaiHacks 2025</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Control Panel")
    
    # Auto-refresh toggle
    auto_refresh = st.checkbox("🔄 Auto-Refresh (5s)", value=True)
    
    # Category filter
    st.markdown("### 📊 Filters")
    category_filter = st.selectbox(
        "Category",
        ["All", "Politics", "Health", "Business", "Technology", "Social", "Crime"]
    )
    
    alert_level_filter = st.multiselect(
        "Alert Level",
        ["LOW", "MEDIUM", "HIGH", "CRITICAL"],
        default=["HIGH", "CRITICAL"]
    )
    
    st.markdown("---")
    
    # Crisis Simulation
    st.markdown("### 🔥 Crisis Simulation")
    st.info("Demonstrate system under extreme threat scenarios")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔥 Cyberattack", use_container_width=True):
            with st.spinner("Simulating..."):
                result = simulate_crisis("cyberattack")
                if result:
                    st.session_state.selected_alert = result
                    st.success("Crisis simulated!")
                    st.rerun()
    
    with col2:
        if st.button("⚠️ Riot", use_container_width=True):
            with st.spinner("Simulating..."):
                result = simulate_crisis("riot")
                if result:
                    st.session_state.selected_alert = result
                    st.success("Crisis simulated!")
                    st.rerun()
    
    if st.button("🌊 Earthquake", use_container_width=True):
        with st.spinner("Simulating..."):
            result = simulate_crisis("earthquake")
            if result:
                st.session_state.selected_alert = result
                st.success("Crisis simulated!")
                st.rerun()
    
    st.markdown("---")
    
    # Custom Analysis
    st.markdown("### 🔍 Custom Analysis")
    with st.expander("Analyze News"):
        custom_headline = st.text_input("Headline")
        custom_content = st.text_area("Content", height=100)
        
        if st.button("🚀 Analyze Now", use_container_width=True):
            if custom_headline and custom_content:
                with st.spinner("Analyzing..."):
                    result = analyze_custom_news(custom_headline, custom_content)
                    if result:
                        st.session_state.selected_alert = result
                        st.success("Analysis complete!")
                        st.rerun()
            else:
                st.warning("Please enter both headline and content")
    
    st.markdown("---")
    st.markdown("#### 📡 System Status")
    st.success("🟢 OPERATIONAL")

# Main Content
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Dashboard",
    "🚨 Active Alerts", 
    "📈 Analytics",
    "🤖 Agent Monitor",
    "📚 History"
])

# TAB 1: Dashboard
with tab1:
    # Time Comparison Banner
    time_comp = fetch_time_comparison()
    if time_comp:
        st.markdown("### ⚡ THE SPEED ADVANTAGE")
        
        col1, col2, col3 = st.columns([2, 1, 2])
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h4 style="color:#ef4444;">Traditional Method</h4>
                <h1 style="color:#ef4444; margin:0;">{time_comp['traditional_method']['time_human']}</h1>
                <p style="color:#94a3b8; margin:0;">Manual verification by team</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div style="text-align:center; padding:2rem 0;">
                <h1 style="color:#fbbf24; font-size:3rem;">⚡</h1>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card" style="border-left-color:#22c55e;">
                <h4 style="color:#22c55e;">Agent Sentinel</h4>
                <h1 style="color:#22c55e; margin:0;">{time_comp['sentinel_method']['time_human']}</h1>
                <p style="color:#94a3b8; margin:0;">{time_comp['speed_multiplier']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
    
    # Stats Cards
    stats = fetch_stats()
    if stats:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h4 style="color:#94a3b8;">📊 Total Analyzed</h4>
                <h1 style="color:#3b82f6; margin:0;">{stats.get('total_analyzed', 0)}</h1>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card" style="border-left-color:#dc2626;">
                <h4 style="color:#94a3b8;">🚨 Active Alerts</h4>
                <h1 style="color:#dc2626; margin:0;">{stats.get('active_alerts', 0)}</h1>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card" style="border-left-color:#22c55e;">
                <h4 style="color:#94a3b8;">🛡️ Threats Prevented</h4>
                <h1 style="color:#22c55e; margin:0;">{stats.get('threats_prevented', 0)}</h1>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card" style="border-left-color:#a855f7;">
                <h4 style="color:#94a3b8;">⏱️ Time Saved</h4>
                <h1 style="color:#a855f7; margin:0;">{stats.get('time_saved_hours', 0)}h</h1>
            </div>
            """, unsafe_allow_html=True)
        
        # Alert Distribution Chart
        st.markdown("### 📊 Alert Distribution")
        
        if stats.get('alert_distribution'):
            fig = go.Figure(data=[go.Pie(
                labels=list(stats['alert_distribution'].keys()),
                values=list(stats['alert_distribution'].values()),
                marker=dict(colors=['#22c55e', '#eab308', '#ea580c', '#dc2626']),
                hole=0.4
            )])
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

# TAB 2: Active Alerts
with tab2:
    st.markdown("### 🚨 ACTIVE ALERTS REQUIRING ATTENTION")
    
    alerts = fetch_active_alerts()
    
    # Filter alerts
    if alert_level_filter:
        alerts = [a for a in alerts if a['alert_level'] in alert_level_filter]
    
    if not alerts:
        st.info("✅ No active alerts. System is secure.")
    else:
        for alert in alerts:
            level = alert['alert_level']
            emoji = get_alert_emoji(level)
            
            with st.container():
                col1, col2 = st.columns([4, 1])
                
                with col1:
                    st.markdown(f"""
                    <div class="metric-card alert-{level.lower()}">
                        <div style="display:flex; justify-content:space-between; align-items:center;">
                            <h3>{emoji} {alert['headline']}</h3>
                            <span style="background:{get_alert_color(level)}; padding:0.25rem 0.75rem; border-radius:5px; font-weight:bold;">{level}</span>
                        </div>
                        <p style="color:#cbd5e1; margin:0.5rem 0;">{alert['content'][:200]}...</p>
                        <div style="display:flex; gap:1rem; margin-top:0.5rem; font-size:0.85rem; color:#94a3b8;">
                            <span>🎯 Score: {alert['falsehood_score']:.2f}</span>
                            <span>⏱️ Processed: {alert['processing_time']:.2f}s</span>
                            <span>🔍 Verified: {'✅' if alert['verification']['is_verified'] else '❌'}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    if st.button("👁️ View Details", key=f"view_{alert['news_id']}", use_container_width=True):
                        st.session_state.selected_alert = alert
                    
                    if st.button("✅ Approve", key=f"approve_{alert['news_id']}", use_container_width=True):
                        result = approve_alert(alert['news_id'], "Judge_Panel")
                        if result:
                            st.success("Alert approved and deployed!")
                            time.sleep(1)
                            st.rerun()
                
                st.markdown("<br>", unsafe_allow_html=True)

# TAB 3: Analytics
with tab3:
    st.markdown("### 📈 THREAT ANALYTICS & VISUALIZATION")
    
    history = fetch_analysis_history(50)
    
    if history:
        # Threat Timeline
        st.markdown("#### 📅 Threat Timeline")
        
        df = pd.DataFrame([{
            'timestamp': datetime.fromisoformat(h['analyzed_at'].replace('Z', '+00:00')),
            'falsehood_score': h['falsehood_score'],
            'alert_level': h['alert_level'],
            'headline': h['headline'][:50] + '...'
        } for h in history])
        
        fig = px.scatter(
            df,
            x='timestamp',
            y='falsehood_score',
            color='alert_level',
            hover_data=['headline'],
            color_discrete_map={
                'LOW': '#22c55e',
                'MEDIUM': '#eab308',
                'HIGH': '#ea580c',
                'CRITICAL': '#dc2626'
            }
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Processing Time Analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### ⚡ Processing Speed")
            avg_time = sum(h['processing_time'] for h in history) / len(history)
            
            st.metric("Average Processing Time", f"{avg_time:.2f}s")
            
            fig = go.Figure(data=[go.Histogram(
                x=[h['processing_time'] for h in history],
                marker_color='#3b82f6'
            )])
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white',
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### 🎯 Verification Success Rate")
            verified_count = sum(1 for h in history if h['verification']['is_verified'])
            success_rate = (verified_count / len(history)) * 100
            
            st.metric("Verification Rate", f"{success_rate:.1f}%")
            
            fig = go.Figure(data=[go.Bar(
                x=['Verified', 'Unverified'],
                y=[verified_count, len(history) - verified_count],
                marker_color=['#22c55e', '#dc2626']
            )])
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white',
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)

# TAB 4: Agent Monitor
with tab4:
    st.markdown("### 🤖 AGENT ACTIVITY MONITOR")
    
    if 'selected_alert' in st.session_state:
        alert = st.session_state.selected_alert
        
        # Agent Actions Log
        st.markdown("#### 🔍 Real-Time Agent Actions")
        
        log_text = ""
        for action in alert['actions_taken']:
            timestamp = action['timestamp'].split('T')[1].split('.')[0]
            log_text += f"[{timestamp}] {action['action_type']}: {action['details']} [{action['status']}]\n"
        
        st.markdown(f'<div class="agent-log">{log_text}</div>', unsafe_allow_html=True)
        
        # Processing Breakdown
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### ⏱️ Processing Breakdown")
            
            verification_time = alert['verification']['verification_time']
            total_time = alert['processing_time']
            
            fig = go.Figure(data=[go.Pie(
                labels=['Verification', 'Analysis', 'Other'],
                values=[verification_time, total_time - verification_time, 0.5],
                marker=dict(colors=['#3b82f6', '#8b5cf6', '#06b6d4'])
            )])
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white',
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### 🎯 Threat Assessment")
            
            st.metric("Falsehood Score", f"{alert['falsehood_score']:.2%}")
            st.metric("Alert Level", alert['alert_level'])
            st.metric("Viral Probability", f"{alert['viral_prediction']['probability']:.2%}")
            
            if alert['viral_prediction']['will_go_viral']:
                st.error(f"⚠️ VIRAL RISK: Est. Reach {alert['viral_prediction']['estimated_reach']:,}")
    else:
        st.info("👈 Select an alert or run a crisis simulation to see agent activity")

# TAB 5: History
with tab5:
    st.markdown("### 📚 ANALYSIS HISTORY")
    
    history = fetch_analysis_history(50)
    
    if history:
        df = pd.DataFrame([{
            'Time': datetime.fromisoformat(h['analyzed_at'].replace('Z', '+00:00')).strftime('%Y-%m-%d %H:%M:%S'),
            'Headline': h['headline'],
            'Alert': h['alert_level'],
            'Score': f"{h['falsehood_score']:.2f}",
            'Verified': '✅' if h['verification']['is_verified'] else '❌',
            'Processing': f"{h['processing_time']:.2f}s"
        } for h in history])
        
        st.dataframe(df, use_container_width=True, height=600)
    else:
        st.info("No analysis history available")

# Selected Alert Detail Modal
if 'selected_alert' in st.session_state and st.session_state.selected_alert:
    alert = st.session_state.selected_alert
    
    with st.expander("📋 DETAILED ANALYSIS REPORT", expanded=True):
        st.markdown(f"## {get_alert_emoji(alert['alert_level'])} {alert['headline']}")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Falsehood Score", f"{alert['falsehood_score']:.2%}")
        with col2:
            st.metric("Alert Level", alert['alert_level'])
        with col3:
            st.metric("Processing Time", f"{alert['processing_time']:.2f}s")
        
        # Threat Meter
        st.markdown("#### 🎯 Threat Meter")
        st.progress(alert['falsehood_score'])
        
        # Verification
        st.markdown("#### ✅ Verification Results")
        st.info(alert['verification']['summary'])
        st.caption(f"Sources found: {len(alert['verification']['sources'])} | Verification time: {alert['verification']['verification_time']:.2f}s")
        
        # Counter Narrative
        if alert.get('counter_narrative'):
            st.markdown("#### 🛡️ Counter-Narrative")
            st.success(alert['counter_narrative']['narrative'])
            
            st.markdown("**Citations:**")
            for cite in alert['counter_narrative']['citations']:
                st.caption(f"• {cite}")
            
            st.markdown("**Deployment Platforms:**")
            platforms = ', '.join(alert['counter_narrative']['target_platforms'])
            st.info(f"📡 {platforms}")
        
        if st.button("❌ Close Detail View", use_container_width=True):
            del st.session_state.selected_alert
            st.rerun()

# Auto-refresh
if auto_refresh:
    time.sleep(5)
    st.rerun()