import cv2
import os

def extract_key_frames_every_n(video_path, output_dir="key_frames", n=60):
    """
    Extract one frame every N frames from a video.

    Args:
        video_path (str): Path to the input video file.
        output_dir (str): Directory where key frames will be saved.
        n (int): Extract one frame every N frames.
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video loaded: {total_frames} frames at {fps:.2f} FPS.")
    print(f"Extracting 1 frame every {n} frames...")

    frame_count = 0
    key_frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % n == 0:
            key_frame_count += 1
            filename = os.path.join(output_dir, f"keyframe_{key_frame_count:04d}.jpg")
            cv2.imwrite(filename, frame)

        frame_count += 1

    cap.release()
    print(f"\n✅ Done! Extracted {key_frame_count} key frames to '{output_dir}'.")

# Example usage
if __name__ == "__main__":
    extract_key_frames_every_n("review.mp4", output_dir="key_frames", n=60)
