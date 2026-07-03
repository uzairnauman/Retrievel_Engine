import streamlit as st
# Import your exact, untouched search function
from retrieve import search  

# 1. Page Configuration
st.set_page_config(page_title="Quran RAG Testing Using FAISS", page_icon="📖", layout="centered")

# 2. Inject Google Fonts & custom CSS for a centered layout
st.html("""
    <style>
        /* Import Google Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Product+Sans:wght@400;700&family=Roboto:wght@400;500&display=swap');
        
        /* Apply fonts globally */
        html, body, [class*="css"], .stMarkdown {
            font-family: 'Product Sans', 'Roboto', sans-serif !important;
        }
        
        /* Center the landing container */
        .google-layout {
            text-align: center;
            margin-top: 15vh;
            margin-bottom: 2rem;
        }
        
        .google-title {
            font-size: 3.5rem;
            font-weight: bold;
            background: linear-gradient(45deg, #4285F4, #EA4335, #FBBC05, #34A853);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }
        
        .google-caption {
            color: #70757a;
            font-size: 1.1rem;
            margin-bottom: 2rem;
        }

        /* Style the result blocks like search snippets */
        .search-result {
            background-color: #ffffff;
            border: 1px solid #dadce0;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 1px 6px rgba(32,33,36,0.1);
        }
    </style>
""")

# 3. Initialize Search / History State
if "has_searched" not in st.session_state:
    st.session_state.has_searched = False
if "messages" not in st.session_state:
    st.session_state.messages = []

# 4. Render Google-style Home Page (Only if no search has happened yet)
if not st.session_state.has_searched:
    st.html("""
        <div class="google-layout">
            <div class="google-title">Quran RAG Using FAISS</div>
            <div class="google-caption">by Uzair Nauman</div>
        </div>
    """)

# 5. Persistent Search Bar (Always active)
# In Streamlit, st.chat_input stays fixed at the bottom, mimicking modern chat/search boxes
if user_query := st.chat_input("Search the Quran or ask a question..."):
    st.session_state.has_searched = True
    # Store query in session history
    st.session_state.messages.append({"role": "user", "content": user_query})
    
    # Trigger the search immediately
    with st.spinner("Searching verses..."):
        search_results = search(user_query, k=7)
        
        if not search_results:
            response_html = "<p style='color:#70757a;'>No highly relevant verses found.</p>"
        else:
            response_html = f"<div style='color: #202124; font-size: 0.9rem; margin-bottom: 15px;'>Showing the top results for: <b>{user_query}</b></div>"
            for res in search_results:
                surah = res.get("surah", "")
                surah_name = res.get("surah_name", "")
                ayah = res.get("ayah", "")
                arabic = res.get("arabic", "")
                english = res.get("english", "")
                score = res.get("score", 0)
                
                response_html += f"""
                <div class="search-result">
                    <div style="color: #1a0dab; font-size: 1.2rem; cursor: pointer; font-weight: 500;">
                        Surah {surah} ({surah_name}) — Ayah {ayah}
                    </div>
                    <div style="color: #006621; font-size: 0.85rem; margin-bottom: 8px;">
                        Semantic Match Score: {score:.4f}
                    </div>
                    <div style="font-size: 1.3rem; text-align: right; direction: rtl; margin-bottom: 10px; font-family: 'Amiri', serif; color: #3c4043;">
                        {arabic}
                    </div>
                    <div style="color: #4d5156; line-height: 1.5;">
                        {english}
                    </div>
                </div>
                """
        st.session_state.messages.append({"role": "assistant", "content": response_html})

# 6. Display search results layout if a query was submitted
if st.session_state.has_searched:
    # Small top brand header when looking at results
    st.html("<h3 style='margin-top:0; color:#4285F4;'>📖 Quranic Search Results</h3><hr>")
    
    # Render latest assistant results directly as HTML snippet cards
    for msg in st.session_state.messages:
        if msg["role"] == "assistant":
            st.html(msg["content"])