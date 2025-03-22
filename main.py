import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
import google.generativeai as genai
from PIL import Image
import streamlit as st

# Configuration de la page Streamlit
st.set_page_config(layout="wide")

# En-tête simplifié avec HTML et CSS
st.markdown("""
    <style>
        .ai-box {
            background: linear-gradient(135deg, #6a11cb, #2575fc);
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.2);
            text-align: center;
            color: #ffffff;
            font-family: 'Arial', sans-serif;
        }
        .ai-box h2 {
            font-size: 24px;
            font-weight: bold;
            margin-bottom: 10px;
        }
        .ai-box p {
            font-size: 16px;
        }
    </style>

    <div class="ai-box">
        <h2>🤖 AI's Response</h2>
        <p>Draw your math expression and see the AI's solution here!</p>
    </div>
""", unsafe_allow_html=True)

# Colonnes pour l'interface
col1, col2 = st.columns([3, 1.5])
with col1:
    Run = st.checkbox("Run", value=True)
    frame_window = st.image([])
with col2:
    output_text = st.subheader("AI's Answer will appear here.")

# Configuration de l'API Google Generative AI
genai.configure(api_key="l'API Google")
model = genai.GenerativeModel(model_name='gemini-1.5-flash')

# Configuration de la capture vidéo
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# Instancier la classe HandDetector
detector = HandDetector(staticMode=False, maxHands=2, modelComplexity=1, detectionCon=0.8, minTrackCon=0.5)

# Fonction de détection de la main
def getInfoHand(img):
    hands, img = detector.findHands(img, draw=True, flipType=True)
    if hands:
        hand1 = hands[0]
        lmList = hand1["lmList"]
        fingers = detector.fingersUp(hand1)
        return fingers, lmList
    else:
        return None

# Fonction de dessin
def Draw(info, canvas, prev_pos):
    fingers, lmList = info
    current_pos = None
    if fingers == [0, 1, 0, 0, 0]:
        current_pos = lmList[8][0:2]
        if prev_pos is None:
            prev_pos = current_pos
        cv2.line(canvas, current_pos, prev_pos, color=(0, 255, 255), thickness=12)
    elif fingers == [1, 0, 0, 0, 0]:
        canvas = np.zeros_like(img)
        output_text.empty()
    return current_pos, canvas

# Fonction pour envoyer au modèle IA
def sendToAI(model, canvas, info):
    fingers, lmList = info
    if fingers == [1, 1, 1, 1, 0]:
        pil_img = Image.fromarray(canvas)
        response = model.generate_content(['found the result of mathematics with max 5 line details ', pil_img])
        return response.text

canvas = None
prev_pos = None
img_fusion = None
output_model = ""
while True:
    success, img = cap.read()
    img = cv2.flip(img, flipCode=1)
    if canvas is None:
        canvas = np.zeros_like(img)
    info = getInfoHand(img)
    if info:
        fingers, lmList = info
        prev_pos, canvas = Draw(info, canvas, prev_pos)
        output_model = sendToAI(model, canvas, info)

    img_fusion = cv2.addWeighted(img, 0.8, canvas, 0.6, 0)
    frame_window.image(img_fusion, channels="BGR")
    if output_model:
        output_text.text(output_model)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()