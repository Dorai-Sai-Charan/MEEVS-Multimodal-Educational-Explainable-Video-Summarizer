# test_download.py
import sys
sys.stdout.reconfigure(encoding='utf-8')

from modules.download_youtube import download_youtube_video

# Provide a valid educational YouTube video URL
test_url = "https://www.youtube.com/watch?v=pqgUfv7UP4A&ab_channel=AiWorld"

print("Downloading video...")
video_path = download_youtube_video(test_url)

print(f"✅ Download complete! File saved at: {video_path}")
