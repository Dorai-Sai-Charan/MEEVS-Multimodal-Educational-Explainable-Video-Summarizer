# app.py - Main Streamlit Interface

import streamlit as st
import os
import tempfile
import cv2
import re
from transformers import BartTokenizer
from fpdf import FPDF
from gtts import gTTS

from modules.download_youtube import download_youtube_video
from modules.extract_audio import extract_audio
from modules.transcribe_audio import transcribe_with_whisper
from modules.summarize_multimodal import summarize_from_texts
from modules.ocr_text import extract_text_from_frames

# App Configuration
st.set_page_config(page_title="🎓 Educational Video Summarizer")
st.title("🎥 Educational Video Summarizer")
st.markdown("This tool summarizes educational videos using speech transcription and NLP.")

# Step 1: YouTube Link Input
youtube_url = st.text_input("🔗 Enter YouTube Video URL:")

# Simple YouTube URL validation pattern
youtube_pattern = r'^(https?\:\/\/)?(www\.youtube\.com|youtu\.be)\/.+$'

if youtube_url:
    if not re.match(youtube_pattern, youtube_url):
        st.error("❌ Invalid YouTube URL. Please enter a valid link.")
        st.stop()

    # Step 2: Download YouTube Video
    if "video_path" not in st.session_state:
        with st.spinner("📥 Downloading video..."):
            st.session_state.video_path = download_youtube_video(youtube_url, output_folder="videos")
            st.success("✅ Video downloaded!")

    st.video(st.session_state.video_path)

    # Step 3: Extract Audio
    if "audio_path" not in st.session_state:
        with st.spinner("🎧 Extracting audio..."):
            st.session_state.audio_path = extract_audio(st.session_state.video_path)
            st.success("✅ Audio extracted!")

    # Step 4: Transcribe Audio
    if "transcript" not in st.session_state:
        with st.spinner("🧠 Transcribing with Whisper..."):
            st.session_state.transcript = transcribe_with_whisper(st.session_state.audio_path)
            st.success("✅ Transcription complete!")

    st.text_area("📝 Transcript", st.session_state.transcript, height=300)

    # Step 5: OCR from Frames
    if "ocr_text" not in st.session_state:
        with st.spinner("🔍 Extracting OCR text from frames..."):
            temp_dir = tempfile.mkdtemp()
            cap = cv2.VideoCapture(st.session_state.video_path)
            frame_paths = []
            frame_rate = cap.get(cv2.CAP_PROP_FPS)
            frame_num = 0
            FRAME_INTERVAL_SECONDS = 5

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                if int(frame_num % (frame_rate * FRAME_INTERVAL_SECONDS)) == 0:
                    frame_path = os.path.join(temp_dir, f"frame_{frame_num}.jpg")
                    cv2.imwrite(frame_path, frame)
                    frame_paths.append(frame_path)
                frame_num += 1
            cap.release()

            st.session_state.ocr_text = extract_text_from_frames(frame_paths[:30])
            st.success("✅ OCR text extracted!")

    st.text_area("🔍 OCR Text", st.session_state.ocr_text, height=200)

    # Step 6: Display token count (for debugging if needed)
    tokenizer = BartTokenizer.from_pretrained("bart_large_arxiv_model")
    combined_text = f"TRANSCRIPT:\n{st.session_state.transcript}\n\nSLIDES TEXT:\n{st.session_state.ocr_text}"
    tokens = tokenizer(combined_text, return_tensors="pt")
    st.write(f"🔢 Token count: {tokens['input_ids'].shape[1]}")

    # Step 7: Generate Summary
    if "summary" not in st.session_state:
        with st.spinner("📝 Generating summary..."):
            st.session_state.summary = summarize_from_texts(st.session_state.transcript, st.session_state.ocr_text)
            st.success("✅ Summary generated!")

    st.text_area("📌 Final Summary", st.session_state.summary, height=300)

    # Step 8: Download Summary as PDF
    def create_pdf(summary_text, filename="summary.pdf"):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.set_font("Arial", size=12)
        for line in summary_text.split("\n"):
            pdf.multi_cell(0, 10, line)
        pdf_path = os.path.join(tempfile.gettempdir(), filename)
        pdf.output(pdf_path)
        return pdf_path

    pdf_path = create_pdf(st.session_state.summary)
    with open(pdf_path, "rb") as pdf_file:
        st.download_button(
            label="📄 Download Summary as PDF",
            data=pdf_file,
            file_name="summary.pdf",
            mime="application/pdf"
        )

    # Step 9: Text-to-Speech Audio of Summary
    tts = gTTS(text=st.session_state.summary, lang='en')
    audio_path = os.path.join(tempfile.gettempdir(), "summary_audio.mp3")
    tts.save(audio_path)

    with open(audio_path, "rb") as audio_file:
        st.audio(audio_file.read(), format="audio/mp3")
