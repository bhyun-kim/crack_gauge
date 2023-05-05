import streamlit as st
import mmcv 
import numpy as np

from PIL import Image
from io import BytesIO
from mmseg.apis import init_model, inference_model, show_result_pyplot
from streamlit_image_comparison import image_comparison

segmentor_cfg = '../../configs/uos_crack/cgnet_uos_crack.py'
segmentor_ckpt = '../../checkpoints/cgnet_uos_crack.pth'

@st.cache_resource
def load_model(config, checkpoint, device='cuda:0'):
    model = init_model(config, checkpoint, device=device)
    return model

def preprocess_image(image):
    image = Image.open(image)
    image = np.array(image)
    return image

def convert_image(img):
    img = Image.fromarray(img)
    buf = BytesIO()
    img.save(buf, format="PNG")
    byte_im = buf.getvalue()
    return byte_im

def segment_crack(model, image):
    image = image[:, :, ::-1]
    result = inference_model(model, image)
    vis_result = show_result_pyplot(model, image, result, show=False)
    return mmcv.bgr2rgb(vis_result)

def preprocess_image(image):
    image = Image.open(image)
    image = np.array(image)
    return image

def main(upload):

    image = preprocess_image(upload)

    segment_result = segment_crack(segmentor, image)
    # segment_result = convert_image(segment_result)

    # image = convert_image(image)

    image_comparison(
        img1=image,
        img2=segment_result,
        label1="Original",
        label2="Segmentation Result",
    )

    # st.download_button(
    #     "Download Segmentation Result",
    #     segment_result,
    #     "segment_result.png",
    #     "image/png",
    # )

st.set_page_config(layout="centered", page_title="Concrete Crack Detection")

st.image("crack_gauge.png")
st.write("Crack gauges have long been civil engineers' reliable partners. \
          Our project aims to introduce AI as our new ally, revolutionizing structural health monitoring. \
          This demo is one of the projects of \
          [Smart Structures & Systems Lab at the University of Seoul](%s)."% ("https://shm.uos.ac.kr/"))

my_upload = st.file_uploader("Upload an image :gear:", type=["png", "jpg", "jpeg"])

segmentor = load_model(segmentor_cfg, segmentor_ckpt)
# col1, col2 = st.columns(2)

if my_upload is not None:
    main(my_upload)
else:
    main("crack.jpg")





    
    