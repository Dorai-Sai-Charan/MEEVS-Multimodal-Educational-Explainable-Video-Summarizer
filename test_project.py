import pytest
import subprocess
from paddleocr import PaddleOCR
from gtts import gTTS
import os

# Test Case 1: YouTube Downloader Test
def test_youtube_downloader():
    url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    result = subprocess.run(["yt-dlp", url], capture_output=True, text=True)
    assert result.returncode == 0
    assert "video" in result.stdout

# Test Case 2: Audio Extractor Test
def test_audio_extractor():
    video_path = "test_video.mp4"
    audio_output_path = "test_audio.wav"
    
    result = subprocess.run(["ffmpeg", "-i", video_path, audio_output_path], capture_output=True, text=True)
    
    assert result.returncode == 0
    assert os.path.exists(audio_output_path)
    os.remove(audio_output_path)

# Test Case 3: OCR Module Test
def test_ocr_module():
    ocr = PaddleOCR(use_angle_cls=True, lang='en')
    image_path = "test_frame.jpg"
    result = ocr.ocr(image_path, cls=True)
    
    assert len(result) > 0
    assert isinstance(result[0], list)

# Test Case 4: Text Merger Test
def test_text_merger():
    transcript = "This is a transcript"
    ocr_text = "This is extracted from an image"
    merged_text = merge_texts(transcript, ocr_text)
    assert merged_text == "This is a transcript. This is extracted from an image"

# Test Case 5: BART Summarizer Test
def test_bart_summarizer():
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    text = "This is a long text that needs to be summarized."
    
    summary = summarizer(text, max_length=50, min_length=25, do_sample=False)
    assert len(summary[0]['summary_text']) > 0
    assert isinstance(summary[0]['summary_text'], str)

# Test Case 6: PDF Generator Test
def test_pdf_generator():
    summary = "This is a summary"
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, summary)
    
    pdf_output_path = "test_summary.pdf"
    pdf.output(pdf_output_path)
    
    assert os.path.exists(pdf_output_path)
    os.remove(pdf_output_path)

# Test Case 7: Text-to-Speech (gTTS) Test
def test_tts():
    text = "This is a test audio"
    tts = gTTS(text=text, lang='en')
    audio_output_path = "test_audio.mp3"
    tts.save(audio_output_path)
    
    assert os.path.exists(audio_output_path)
    os.remove(audio_output_path)

# Test Case 8: Streamlit UI Test
def test_streamlit_ui():
    st.text_input("Enter YouTube URL")
    st.button("Download Video")
    
    assert "Enter YouTube URL" in st.text_input
    assert "Download Video" in st.button

# Test Case 9: Summary Download Test
def test_summary_download():
    url = "http://localhost:8501/download-summary"
    response = requests.get(url)
    
    assert response.status_code == 200
    assert "application/pdf" in response.headers['Content-Type']

# Test Case 10: Audio Playback Test
def test_audio_playback():
    url = "http://localhost:8501/play-audio"
    response = requests.get(url)
    
    assert response.status_code == 200
    assert "audio/mp3" in response.headers['Content-Type']

# Test Case 11: Token Length Handling Test
def test_token_length():
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
    text = "This is a sample input for testing the token length."
    tokens = tokenizer.encode(text)
    
    assert len(tokens) <= tokenizer.model_max_length

# Running the tests
if __name__ == "__main__":
    pytest.main()
