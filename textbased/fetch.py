# import os
# import yt_dlp

# def Download(link):
#     # Define the output path
#     download_dir = os.path.join(os.getcwd(), "videos")

#     # Create the 'videos' folder if it doesn't exist
#     if not os.path.exists(download_dir):
#         os.makedirs(download_dir)

#     ydl_opts = {
#         'format': 'best',  # Download the best quality
#         'outtmpl': os.path.join(download_dir, '%(title)s.%(ext)s'),  # Save in 'videos' folder
#     }

#     try:
#         with yt_dlp.YoutubeDL(ydl_opts) as ydl:
#             ydl.download([link])
#         print(f"✅ Download is completed successfully. Saved to: {download_dir}")
#     except Exception as e:
#         print(f"❌ An error has occurred: {str(e)}")

# # Test with your YouTube URL
# # Download("https://www.youtube.com/watch?v=I-8cUbClYzI")

# links=["https://www.youtube.com/watch?v=yOOCBqZtkYE","https://www.youtube.com/watch?v=Zu9rfb7xNJY","https://www.youtube.com/watch?v=J4f-c1f1kxg"]
# for i in range(len(links)):
#     Download(links[i])

import os
import yt_dlp
import random

def Download(link):
    # Define the output path
    download_dir = os.path.join(os.getcwd(), "videos")

    # Create the 'videos' folder if it doesn't exist
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)

    # Generate a random 6-digit number for the filename
    random_id = str(random.randint(100000, 999999))

    ydl_opts = {
        'format': 'best',  # Download the best available quality
        'outtmpl': os.path.join(download_dir, f'{random_id}.%(ext)s'),  # Save as random number
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([link])
        print(f"✅ Download completed successfully. Saved as {random_id} in: {download_dir}")
    except Exception as e:
        print(f"❌ An error occurred: {str(e)}")

# Test with multiple YouTube URLs
links = [
    "https://www.youtube.com/watch?v=yOOCBqZtkYE",
    "https://www.youtube.com/watch?v=Zu9rfb7xNJY",
    "https://www.youtube.com/watch?v=J4f-c1f1kxg"
]

for link in links:
    Download(link)
