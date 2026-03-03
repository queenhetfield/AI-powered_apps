from PIL import Image
from transformers import AutoProcessor, BlipForConditionalGeneration
import glob

processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

image_dir = '/Users/laura/Pictures/Desktop/*'

with open("captions_local.txt", "w", encoding="utf-8") as caption_file:
    for idx, image in enumerate(glob.glob(image_dir), start=1):
            
        inputs = processor(images=image, return_tensors="pt")
        out = model.generate(**inputs, max_new_tokens=50)
        caption = processor.decode(out[0], skip_special_tokens=True)

        caption_file.write(f"{image_dir}{image}: {caption}\n")
        print(f"[{idx}] Caption saved")