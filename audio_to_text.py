# import torch
# from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
# # from datasets import load_dataset


# device = "cuda:0" if torch.cuda.is_available() else "cpu"
# torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# model_id = "openai/whisper-large-v3"

# model = AutoModelForSpeechSeq2Seq.from_pretrained(
#     model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True
# )
# model.to(device)

# processor = AutoProcessor.from_pretrained(model_id)

# pipe = pipeline(
#     "automatic-speech-recognition",
#     model=model,
#     tokenizer=processor.tokenizer,
#     feature_extractor=processor.feature_extractor,
#     chunk_length_s=30,
#     batch_size=16,  # batch size for inference - set based on your device
#     torch_dtype=torch_dtype,
#     device=device,
# )

# # dataset = load_dataset("distil-whisper/librispeech_long", "clean", split="validation")
# sample = "C:/Users/Ayush/Desktop/apps/inspix/audio.mp3"

# result = pipe(sample)
# print(result["text"])

import whisper

# Load the pre-trained Whisper model
model = whisper.load_model("base")

# Transcribe audio
result = model.transcribe("audio.mp3")

print(result)

with open("transcription.txt", "w") as file:
    file.write(result["text"])

