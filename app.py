import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from model import Encoder, Generator, Discriminator
import io

# Initialize the models
netE = Encoder()
netG = Generator()
# Load pre-trained model weights
netE.load_state_dict(torch.load("netE8.model", map_location=torch.device('cpu')))
netG.load_state_dict(torch.load("netG8.model", map_location=torch.device('cpu')))
netE.eval()
netG.eval()

# Streamlit interface
st.title("Image Compression and Decompression")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
compression_ratio = st.slider("Compression Ratio", 0, 100, 50)

def compress_and_decompress(image, compression_ratio):
    # Preprocess the image
    transform = transforms.Compose([
        transforms.Resize((218, 178)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    image_tensor = transform(image).unsqueeze(0)

    # Compress and decompress the image
    encoded_img = netE(image_tensor)
    reconstructed_img = netG(encoded_img)

    # Postprocess the image
    reconstructed_img = reconstructed_img.squeeze().detach().cpu().numpy().transpose(1, 2, 0)
    reconstructed_img = (reconstructed_img * 0.5) + 0.5
    reconstructed_img = np.clip(reconstructed_img, 0, 1)
    return reconstructed_img

def get_image_download_link(img, filename, text):
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    img_str = buffered.getvalue()
    href = f'<a href="data:image/jpeg;base64,{base64.b64encode(img_str).decode()}" download="{filename}">{text}</a>'
    return href

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Original Image", use_column_width=True)

    if st.button("Compress and Decompress"):
        compressed_image = compress_and_decompress(image, compression_ratio)
        st.image(compressed_image, caption="Compressed and Decompressed Image", use_column_width=True)
        # Convert the compressed image array to an Image object
        compressed_image_pil = Image.fromarray((compressed_image * 255).astype(np.uint8))
        # Create a download link for the compressed image
        download_link = get_image_download_link(compressed_image_pil, "compressed_image.jpg", "Download Compressed Image")
        st.markdown(download_link, unsafe_allow_html=True)
