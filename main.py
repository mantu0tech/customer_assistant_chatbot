import os
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from groq import Groq
import kagglehub
from dotenv import load_dotenv
import streamlit as st
from datetime import datetime
import re
import html

# ==========================
# 🎨 PAGE CONFIGURATION
# ==========================
st.set_page_config(
    page_title="AI Customer Support",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================
# 🎨 CUSTOM CSS
# ==========================
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 5px solid #2196f3;
        color: #1565c0;
    }
    .bot-message {
        background-color: #f3e5f5;
        border-left: 5px solid #9c27b0;
        color: #4a148c;
    }
    .matched-instruction {
        background-color: #fff3e0;
        border-left: 5px solid #ff9800;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        font-size: 0.9rem;
        color: #e65100;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem;
        font-size: 1rem;
        border-radius: 0.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        color: white;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ==========================
# 🔐 CONFIGURATION & INITIALIZATION
# ==========================
@st.cache_resource
def initialize_system():
    """Initialize all components once"""
    
    # Load environment variables
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    
    if not api_key:
        st.error("❌ GROQ_API_KEY not found in .env file")
        st.stop()
    
    client = Groq(api_key=api_key)
    
    # Download dataset
    with st.spinner("📥 Downloading dataset from KaggleHub..."):
        dataset_path = kagglehub.dataset_download(
            "bitext/bitext-gen-ai-chatbot-customer-support-dataset"
        )
        csv_path = os.path.join(
            dataset_path,
            "Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv"
        )
    
    # Load data
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["instruction", "response"]).reset_index(drop=True)
    
    # Load embedding model
    with st.spinner("🤖 Loading AI models..."):
        embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        instruction_embeddings = embedding_model.encode(
            df["instruction"].tolist(),
            convert_to_tensor=False,
            show_progress_bar=False
        )
    
    # Build FAISS index
    dimension = instruction_embeddings[0].shape[0]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(instruction_embeddings).astype('float32'))
    
    return client, df, embedding_model, index

# Initialize
client, df, embedding_model, index = initialize_system()

# ==========================
# 🔎 HELPER FUNCTIONS
# ==========================
def get_best_match(user_input, top_k=1):
    user_embedding = embedding_model.encode([user_input])
    distances, indices = index.search(
        np.array(user_embedding).astype('float32'),
        top_k
    )
    return df.iloc[indices[0]], distances[0]

def rephrase_with_groq(user_input, original_response, model="llama-3.3-70b-versatile"):
    prompt = f"""You are a helpful and empathetic customer support agent.

Customer message: "{user_input}"

Previous system response: "{original_response}"

Rewrite the response to sound more human, friendly, and professional. 
IMPORTANT: Respond with plain text only, no HTML tags or special formatting."""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a professional customer support assistant. Always respond in plain text without any HTML tags or markup."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )
        # Strip any HTML tags if they appear
        content = response.choices[0].message.content.strip()
        # Remove HTML tags using simple string replacement
        import re
        content = re.sub('<[^<]+?>', '', content)
        return content
    except Exception as e:
        st.error(f"⚠️ Groq API error: {e}")
        return original_response

# ==========================
# 💾 SESSION STATE
# ==========================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "conversation_count" not in st.session_state:
    st.session_state.conversation_count = 0

# ==========================
# 🎯 MAIN APP
# ==========================
# Header
st.markdown('<p class="main-header">🤖 AI Customer Support Assistant</p>', unsafe_allow_html=True)
st.markdown("### Powered by FAISS, Sentence Transformers & Groq LLM")

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/chatbot.png", width=100)
    st.title("📊 Dashboard")
    
    # Metrics
    st.markdown(f"""
    <div class="metric-card">
        <h3>{len(df):,}</h3>
        <p>Training Records</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="metric-card">
        <h3>{st.session_state.conversation_count}</h3>
        <p>Conversations</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Settings
    st.subheader("⚙️ Settings")
    show_matched = st.checkbox("Show matched instruction", value=True)
    show_confidence = st.checkbox("Show confidence score", value=True)
    
    st.markdown("---")
    
    # Clear chat button
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.session_state.conversation_count = 0
        st.rerun()
    
    st.markdown("---")
    st.markdown("### 💡 Tips")
    st.info("""
    - Ask about account issues
    - Inquire about shipping
    - Request product info
    - Get billing help
    """)

# Main chat area
st.markdown("---")

# Display chat history
for message in st.session_state.messages:
    if message["role"] == "user":
        with st.container():
            st.markdown(f"""
            <div class="chat-message user-message">
                <b style="color: #1565c0;">👤 You</b><br>
                <span style="color: #0d47a1;">{html.escape(message["content"])}</span>
            </div>
            """, unsafe_allow_html=True)
    else:
        with st.container():
            # Main bot message
            st.markdown(f"""
            <div class="chat-message bot-message">
                <b style="color: #4a148c;">🤖 Support Bot</b><br>
                <span style="color: #6a1b9a;">{html.escape(message["content"])}</span>
            </div>
            """, unsafe_allow_html=True)
            
            # Matched instruction (separate container)
            if show_matched and "matched_instruction" in message:
                st.markdown(f"""
                <div class="matched-instruction">
                    <b style="color: #e65100;">🔎 Matched Query:</b> <span style="color: #bf360c;">{html.escape(message["matched_instruction"])}</span>
                </div>
                """, unsafe_allow_html=True)
            
            # Confidence score (separate element)
            if show_confidence and "confidence" in message:
                confidence_pct = (1 / (1 + message["confidence"])) * 100
                st.markdown(f"""
                <small style="color: #7b1fa2;">📊 Confidence: {confidence_pct:.1f}%</small>
                """, unsafe_allow_html=True)

# Chat input
user_input = st.chat_input("Type your message here...")

if user_input:
    # Add user message
    st.session_state.messages.append({
        "role": "user",
        "content": user_input,
        "timestamp": datetime.now()
    })
    
    # Get bot response
    with st.spinner("🤔 Thinking..."):
        try:
            matched, distances = get_best_match(user_input)
            matched_instruction = matched.iloc[0]["instruction"]
            matched_response = matched.iloc[0]["response"]
            
            final_reply = rephrase_with_groq(
                user_input=user_input,
                original_response=matched_response
            )
            
            # Add bot message
            st.session_state.messages.append({
                "role": "assistant",
                "content": final_reply,
                "matched_instruction": matched_instruction,
                "confidence": float(distances[0]),
                "timestamp": datetime.now()
            })
            
            st.session_state.conversation_count += 1
            
        except Exception as e:
            st.error(f"❌ Error: {e}")
    
    st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p>Built with ❤️ using Streamlit, FAISS, Sentence Transformers & Groq</p>
</div>
""", unsafe_allow_html=True)