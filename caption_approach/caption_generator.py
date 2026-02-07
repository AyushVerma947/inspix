
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
from PIL import Image
import os


model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)



max_length = 32
num_beams = 4
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}
def predict_step(image_paths):
  images = []
  for image_path in image_paths:
    i_image = Image.open(image_path)
    if i_image.mode != "RGB":
      i_image = i_image.convert(mode="RGB")

    images.append(i_image)

  pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
  pixel_values = pixel_values.to(device)

  output_ids = model.generate(pixel_values, **gen_kwargs)

  preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
  preds = [pred.strip() for pred in preds]
  return preds


# ans = predict_step(['C:/Users/Ayush/Desktop/apps/inspix/key_frames/keyframe_0009.jpg']) 
# print(ans)

import os

def caption_folder(folder_path, batch_size=8):
    # Collect image file paths
    valid_ext = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    image_files = [ 
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith(valid_ext)
    ]

    print(f"Found {len(image_files)} images")

    all_captions = {}

    # Process in batches (faster than one-by-one)
    for i in range(0, len(image_files), batch_size):
        batch_paths = image_files[i:i+batch_size]
        captions = predict_step(batch_paths)
        

        for path, caption in zip(batch_paths, captions):
            all_captions[path] = caption
            print(f"{os.path.basename(path)} → {caption}")

    return all_captions


folder = r"C:/Users/Ayush/Desktop/apps/inspix/key_frames"
results = caption_folder(folder)