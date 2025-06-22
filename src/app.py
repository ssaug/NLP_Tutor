import os
import signal
import pickle
import threading
import time
import hashlib
import streamlit as st
import yt_dlp
from PyPDF2 import PdfReader
from functions import get_pdf_content, get_chunks, get_vectorstore, conversation_chain
from html_template import css, ai_template, human_template, hide_st_style
from langchain.schema import HumanMessage, AIMessage

# YouTube Search
def get_youtube_videos(query, max_results=3):
    search_query = f"ytsearch{max_results}:{query}"
    ydl_opts = {
        "quiet": True,
        "skip_download": True,
        "extract_flat": True,
        "force_generic_extractor": False
    }
    results = []
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(search_query, download=False)
            for entry in info.get("entries", []):
                video_id = entry.get("id")
                title = entry.get("title")
                if video_id and title:
                    results.append({
                        "title": title,
                        "url": f"https://www.youtube.com/watch?v={video_id}"
                    })
    except Exception as e:
        print(f"YouTube fetch error: {e}")
    return results

def hash_files(uploaded_files):
    hash_sha256 = hashlib.sha256()
    names = []
    for file in uploaded_files:
        file.seek(0)
        hash_sha256.update(file.read())
        file.seek(0)
        names.append(file.name)
    return hash_sha256.hexdigest(), ", ".join(names)

def save_vectorstore_to_disk(vectorstore, file_key):
    os.makedirs("cache", exist_ok=True)
    with open(f"cache/{file_key}.pkl", "wb") as f:
        pickle.dump(vectorstore, f)

def load_cached_files():
    os.makedirs("cache", exist_ok=True)
    return [f[:-4] for f in os.listdir("cache") if f.endswith(".pkl")]

def clear_all_cache():
    if os.path.exists("cache"):
        for f in os.listdir("cache"):
            os.remove(os.path.join("cache", f))

def async_process_files(uploaded_files, file_key):
    try:
        start_time = time.time()
        raw_text = ""
        total_pages = 0
        processed_pages = 0

        for document in uploaded_files:
            reader = PdfReader(document)
            total_pages += len(reader.pages)

        for document in uploaded_files:
            reader = PdfReader(document)
            for page in reader.pages:
                raw_text += page.extract_text()
                processed_pages += 1
                progress = int((processed_pages / total_pages) * 100)
                elapsed = time.time() - start_time
                eta = int(((elapsed / processed_pages) * total_pages) - elapsed) if processed_pages > 0 else 0
                st.session_state[f"load_progress_{file_key}"] = progress
                st.session_state[f"load_eta_{file_key}"] = eta

        text_chunks = get_chunks(raw_text)
        if not text_chunks:
            st.session_state[f"load_done_{file_key}"] = "error"
            return

        new_vectorstore = get_vectorstore(text_chunks)
        file_path = f"cache/{file_key}.pkl"
        if os.path.exists(file_path):
            with open(file_path, "rb") as f:
                existing_vectorstore = pickle.load(f)
            existing_vectorstore.merge_from(new_vectorstore)
            new_vectorstore = existing_vectorstore

        save_vectorstore_to_disk(new_vectorstore, file_key)
        st.session_state[f"load_done_{file_key}"] = "done"

    except Exception as e:
        st.session_state[f"load_done_{file_key}"] = "error"
        st.session_state[f"load_error_{file_key}"] = str(e)

def generate_page():
    st.set_page_config(page_title="ChatPDF", page_icon="📄", layout="wide")
    st.write(css, unsafe_allow_html=True)
    st.markdown(hide_st_style, unsafe_allow_html=True)

    if "file_name_map" not in st.session_state:
        st.session_state["file_name_map"] = {}

    with st.sidebar:
        st.image("./src/assets/vertical_logo.png", use_column_width=True)
        st.subheader("", divider="red")

        cached_files = load_cached_files()
        if cached_files:
            st.markdown("### 📂 Available Sessions:")
            options = [f"{st.session_state['file_name_map'].get(k, k)} ({k[:8]})" for k in cached_files]
            file_lookup = dict(zip(options, cached_files))
            selected_label = st.selectbox("Choose a file to search:", options)
            selected_key = file_lookup[selected_label]

            if st.button("LOAD SELECTED", use_container_width=True):
                with open(f"cache/{selected_key}.pkl", "rb") as f:
                    st.session_state["vectorstore"] = pickle.load(f)
                st.session_state["files_names"] = selected_label
                st.success(f"Loaded session: {selected_label}")

        if st.button("CLEAR ALL CACHE", use_container_width=True):
            clear_all_cache()
            st.session_state["file_name_map"] = {}
            st.success("All cached sessions cleared.")
            st.rerun()

        if uploaded_files := st.file_uploader("Upload PDFs and click PROCESS", type="pdf", accept_multiple_files=True):
            if st.button("PROCESS", help="Click to process uploaded PDFs.", use_container_width=True):
                file_key, file_label = hash_files(uploaded_files)
                st.session_state["files_names"] = file_key
                st.session_state["file_name_map"][file_key] = file_label
                st.session_state[f"load_done_{file_key}"] = "loading"
                st.session_state[f"load_progress_{file_key}"] = 0
                st.session_state[f"load_eta_{file_key}"] = 0
                threading.Thread(target=async_process_files, args=(uploaded_files, file_key), daemon=True).start()
                st.success(f"Started loading in background: {file_label}")

        if st.button("REFRESH STATUS", use_container_width=True):
            st.experimental_rerun()

        for key in list(st.session_state.keys()):
            if key.startswith("load_done_"):
                file_key = key.replace("load_done_", "")
                file_label = st.session_state["file_name_map"].get(file_key, file_key)
                status = st.session_state[key]
                if status == "done":
                    st.success(f"✅ Loaded: {file_label}")
                elif status == "error":
                    msg = st.session_state.get(f"load_error_{file_key}", "Unknown error")
                    st.error(f"❌ Failed to load: {file_label}\n{msg}")
                else:
                    progress = st.session_state.get(f"load_progress_{file_key}", 0)
                    eta = st.session_state.get(f"load_eta_{file_key}", 0)
                    st.info(f"⏳ Loading {file_label} — ETA: {eta} sec")
                    st.progress(progress, text=f"Progress: {progress}%")

        if st.button("NEW CHAT", use_container_width=True):
            st.session_state.pop("chat_history", None)
            st.rerun()

        if st.button("RESTART APP", use_container_width=True):
            st.session_state.clear()
            st.warning("Restarting app...")
            st.rerun()

        if st.button("KILL APP", use_container_width=True):
            os.kill(os.getpid(), signal.SIGTERM)

    st.header(":speech_balloon: Chat with your NLP Tutor", divider="red")

    if "vectorstore" in st.session_state:
        st.success(f"You are talking with session: {st.session_state['files_names']}")
        if "chat_history" not in st.session_state:
            st.session_state["chat_history"] = [
                AIMessage(content="Hi I'm your NLP AI assistant to help you find specific information, summarize content, or simply explore your document.")
            ]

        if user_message := st.chat_input("What do you want to know about your documents?"):
            st.session_state["chat_history"].append(HumanMessage(content=user_message))
            response = ""
            for chunk in conversation_chain(user_message, st.session_state['vectorstore'], st.session_state["chat_history"]):
                response += chunk
            st.session_state["chat_history"].append(AIMessage(content=response))

        for msg in st.session_state["chat_history"]:
            template = ai_template if isinstance(msg, AIMessage) else human_template
            st.write(template.replace("{{MSG}}", "\n\n" + msg.content), unsafe_allow_html=True)

        if user_message:
            st.markdown("\n**🎬 Related Videos:**")
            for video in get_youtube_videos(user_message):
                st.markdown(f"▶️ **[{video['title']}]({video['url']})**", unsafe_allow_html=True)
    else:
        st.write(ai_template.replace("{{MSG}}", "\n\nHi I'm your NLP AI assistant. Upload a PDF to get started! :rocket:"), unsafe_allow_html=True)

if __name__ == "__main__":
    generate_page()
