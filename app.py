import av
from streamlit_webrtc import webrtc_streamer
from PIL import Image
import cv2
from ultralytics import YOLO

model = YOLO('static/meat.pt')

def video_frame_callback(frame):
    frame = frame.to_ndarray(format='bgr24')
    img = Image.fromarray(frame)
    img = model.predict(img)
    img_array = img[0].plot()
    #predict後の画像は色がおかしいので直す
    img_array_color = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)

    return av.VideoFrame.from_ndarray(img_array_color, format="bgr24")


ctx = webrtc_streamer(key="example", video_frame_callback=video_frame_callback,
                      rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
