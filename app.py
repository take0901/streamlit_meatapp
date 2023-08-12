import av
from streamlit_webrtc import webrtc_streamer
from PIL import Image
from ultralytics import YOLO

model = YOLO('static/meat.pt')

def video_frame_callback(frame):
    frame = frame.to_ndarray(format='bgr24')
    img = Image.fromarray(frame)
    img = model.predict(img)
    img_array = img[0].plot()

    return av.VideoFrame.from_ndarray(img_array, format="bgr24")


ctx = webrtc_streamer(key="example", video_frame_callback=video_frame_callback,
                      rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
