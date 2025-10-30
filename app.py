"""
Streamlit frontend for AgentCert MVP
"""
import streamlit as st
import requests
import json
import pandas as pd
from datetime import datetime
from typing import Optional

# Page configuration
st.set_page_config(
    page_title="AgentCert MVP",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Backend API URL
API_URL = "http://localhost:8000"

# Initialize session state
if 'certification' not in st.session_state:
    st.session_state.certification = None

if 'attempts' not in st.session_state:
    st.session_state.attempts = 0

if 'current_response' not in st.session_state:
    st.session_state.current_response = None

if 'current_evaluation' not in st.session_state:
    st.session_state.current_evaluation = None


def check_backend():
    """Check if backend is running"""
    try:
        response = requests.get(f"{API_URL}/", timeout=2)
        return response.status_code == 200
    except:
        return False


# Main UI
def main():
    """Main Streamlit app"""
    
    # Header
    st.title("🤖 AgentCert MVP")
    st.markdown("### Trust and Certification System for AI Agents")
    st.markdown("---")
    
    # Check backend status
    backend_status = check_backend()
    if not backend_status:
        st.error("⚠️ Backend API is not running. Please start it with: `uvicorn backend:app --reload`")
        st.stop()
    else:
        st.success("✅ Backend API connected")
    
    # Sidebar
    with st.sidebar:
        st.header("📋 Navigation")
        st.info("""
        **How to use:**
        1. Enter a financial scenario
        2. Click "Run Task"
        3. Review evaluation scores
        4. Generate certification when ready
        """)
        
        # Clear history button
        if st.button("🗑️ Clear History", use_container_width=True):
            try:
                response = requests.delete(f"{API_URL}/history")
                if response.status_code == 200:
                    st.session_state.certification = None
                    st.session_state.current_response = None
                    st.session_state.current_evaluation = None
                    st.success("History cleared!")
                    st.rerun()
            except Exception as e:
                st.error(f"Error clearing history: {e}")
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📝 Financial Task Input")
        
        # Predefined questions
        st.subheader("Quick Examples")
        example_questions = [
            "How should I diversify a $10K portfolio?",
            "What are the risks of investing in a single tech stock?",
            "Explain compound interest to a beginner.",
            "What is dollar-cost averaging?",
            "How does an emergency fund help with financial security?"
        ]
        
        selected_example = st.selectbox(
            "Choose an example question:",
            [""] + example_questions
        )
        
        # Task input
        task_prompt = st.text_area(
            "Or enter your own financial scenario:",
            value=selected_example if selected_example else "",
            height=100,
            placeholder="Example: How should I invest $5,000 as a beginner?"
        )
        
        # Run task button
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("▶️ Run Task", type="primary", use_container_width=True):
                if not task_prompt:
                    st.warning("Please enter a task prompt first.")
                else:
                    with st.spinner("🤖 Agent is thinking..."):
                        try:
                            response = requests.post(
                                f"{API_URL}/run_task",
                                json={"prompt": task_prompt}
                            )
                            if response.status_code == 200:
                                data = response.json()
                                st.session_state.current_response = data
                                st.session_state.attempts += 1
                                st.success("✅ Agent response generated!")
                            else:
                                st.error(f"Error: {response.status_code}")
                        except Exception as e:
                            st.error(f"Error calling backend: {e}")
        
        with col_btn2:
            if st.button("📊 Evaluate Response", use_container_width=True):
                if st.session_state.current_response:
                    with st.spinner("🔍 Evaluating response..."):
                        try:
                            response = requests.post(
                                f"{API_URL}/evaluate",
                                json=st.session_state.current_response
                            )
                            if response.status_code == 200:
                                data = response.json()
                                st.session_state.current_evaluation = data
                                st.success("✅ Evaluation complete!")
                            else:
                                st.error(f"Error: {response.status_code}")
                        except Exception as e:
                            st.error(f"Error calling backend: {e}")
                else:
                    st.warning("Please run a task first.")
    
    with col2:
        st.header("🎯 Current Status")
        
        if st.session_state.current_response:
            st.markdown("### 📤 Agent Response")
            st.info(st.session_state.current_response['response'])
            
            if st.session_state.current_evaluation:
                st.markdown("### 📈 Evaluation Scores")
                
                eval_data = st.session_state.current_evaluation
                scores = {
                    "Accuracy": eval_data['accuracy'],
                    "Clarity": eval_data['clarity'],
                    "Compliance": eval_data['compliance'],
                    "Average": eval_data.get('average_score', 0)
                }
                
                # Display scores with progress bars
                for metric, score in scores.items():
                    if metric != "Average":
                        st.metric(metric, f"{score}/100")
                        st.progress(score / 100)
                    else:
                        st.metric(metric, f"{score:.2f}/100")
                        st.progress(score / 100)
                
                # Feedback
                st.markdown("### 💬 Feedback")
                st.text_area(
                    "Evaluator feedback:",
                    value=eval_data['feedback'],
                    height=100,
                    disabled=True
                )
    
    # Certification section
    st.markdown("---")
    st.header("🏆 Certification")
    
    col_cert1, col_cert2 = st.columns([3, 1])
    
    with col_cert1:
        if st.button("✨ Generate Certification", type="primary", use_container_width=True):
            with st.spinner("🎓 Generating certification..."):
                try:
                    response = requests.post(f"{API_URL}/certify")
                    if response.status_code == 200:
                        st.session_state.certification = response.json()
                        st.success("✅ Certification generated!")
                except Exception as e:
                    st.error(f"Error: {e}")
        
        if st.session_state.certification:
            cert = st.session_state.certification
            st.markdown("### 📜 Certification Details")
            
            # Status badge
            if cert['certified']:
                st.success("✅ CERTIFIED AGENT")
            else:
                st.warning(f"⚠️ Not certified (Need {cert['passed_threshold']}% in all categories)")
            
            # Scores
            st.markdown("**Performance Scores:**")
            scores_df = pd.DataFrame(
                [cert['scores']],
                index=["Overall Performance"]
            )
            st.dataframe(scores_df, use_container_width=True)
            
            # Certification JSON
            st.markdown("**Certification JSON:**")
            cert_copy = cert.copy()
            if cert_copy.get('issued_at'):
                cert_copy['issued_at'] = cert_copy['issued_at'].strftime("%Y-%m-%d %H:%M:%S UTC")
            
            st.code(json.dumps(cert_copy, indent=2), language="json")
    
    with col_cert2:
        # Badge display
        if st.session_state.certification and st.session_state.certification['certified']:
            st.markdown("### 🏅 Badge")
            st.markdown("""
            <div style="
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 40px;
                border-radius: 10px;
                text-align: center;
                font-size: 24px;
                font-weight: bold;
            ">
                ✓ Certified Agent
            </div>
            """, unsafe_allow_html=True)
    
    # History section
    st.markdown("---")
    st.header("📊 Evaluation History")
    
    try:
        response = requests.get(f"{API_URL}/history")
        if response.status_code == 200:
            history = response.json()
            
            if history['evaluations']:
                # Summary metrics
                col_h1, col_h2, col_h3, col_h4 = st.columns(4)
                
                with col_h1:
                    st.metric("Total Attempts", len(history['evaluations']))
                with col_h2:
                    st.metric("Avg Accuracy", f"{history['average_accuracy']:.1f}%")
                with col_h3:
                    st.metric("Avg Clarity", f"{history['average_clarity']:.1f}%")
                with col_h4:
                    st.metric("Avg Compliance", f"{history['average_compliance']:.1f}%")
                
                # History table
                history_df = pd.DataFrame([
                    {
                        "Attempt": i + 1,
                        "Accuracy": e['accuracy'],
                        "Clarity": e['clarity'],
                        "Compliance": e['compliance'],
                        "Average": round(e['average_score'], 2),
                        "Feedback": e['feedback'][:50] + "..." if len(e['feedback']) > 50 else e['feedback']
                    }
                    for i, e in enumerate(history['evaluations'])
                ])
                
                st.dataframe(history_df, use_container_width=True, hide_index=True)
                
                # Line chart
                if len(history['evaluations']) > 1:
                    st.markdown("### 📈 Score Trend")
                    scores_over_time = {
                        "Attempt": range(1, len(history['evaluations']) + 1),
                        "Accuracy": [e['accuracy'] for e in history['evaluations']],
                        "Clarity": [e['clarity'] for e in history['evaluations']],
                        "Compliance": [e['compliance'] for e in history['evaluations']]
                    }
                    scores_df = pd.DataFrame(scores_over_time)
                    st.line_chart(scores_df.set_index("Attempt"), use_container_width=True)
            else:
                st.info("No evaluation history yet. Run some tasks and evaluate them!")
    except Exception as e:
        st.warning("Could not fetch history. Backend may be down.")


if __name__ == "__main__":
    main()

