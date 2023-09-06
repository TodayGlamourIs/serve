# 初始化資料庫連線
from pymongo.mongo_client import MongoClient
url = "mongodb+srv://*******@cluster0.aagcvdi.mongodb.net/?retryWrites=true&w=majority"
client = MongoClient(url)
# try:
#     client.admin.command('ping')
#     print("Pinged your deployment. You successfully connected to MongoDB!")
# except Exception as e:
#     print(e)
# print("連線成功")
db=client.test

from flask import Flask, request, redirect, render_template, session, jsonify
import pydicom
from keras.models import load_model
import numpy as np
import cv2
from PIL import Image
import base64
from io import BytesIO
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage, ImageMessage,QuickReply, QuickReplyButton, MessageAction, ImageSendMessage
import time
import logging
import configparser

app = Flask(__name__, static_folder="static", static_url_path="/" )
app.secret_key="LoginTest"
config = configparser.ConfigParser()
config.read('config.ini')

# 设置日志记录
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# 處理路由
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/member")
def member():
    if "nickname" in session:
        return render_template("member.html")
    else:
        return redirect("/") 

# /error?msg="各式錯誤訊息"
@app.route("/error")
def error():
    message=request.args.get("msg","發生錯誤,請聯繫客服")
    return render_template("error.html", message=message)

@app.route("/signup", methods=["POST"])
def signup():
    # 從前端接收資料
    nickname=request.form["nickname"]
    email=request.form["email"]
    password=request.form["password"]
    # 根據接收到的資料與資料庫互動
    collection=db.users
    result=collection.find_one({
        "email":email
    })
    # 檢查是否有重覆的email
    if result != None:
        return redirect("/error?msg=信箱已經被註冊")
    # 把資料放進資料庫.完成註冊
    collection.insert_one({
        "nickname":nickname,
        "email":email,
        "password":password
    })
    return redirect("/")

@app.route("/signin", methods=["POST"])
def signin():
    # 從前端取得使用者輸入
    email=request.form["email"]
    password=request.form["password"]
    #與資料庫作互動
    collection=db.users
    result=collection.find_one({
        "$and":[
            {"email":email},
            {"password":password}
            ]
    })
    # 找不到對應資料，登入失敗
    if result==None:
        return redirect("/error?msg=帳號或密碼輸入錯誤")
    # 登入成功，在 session 紀錄會員資訊，導向會員頁面
    session["nickname"]=result["nickname"]
    return redirect("/model_input")


# 加載訓練好的模型
model_path = 'densenetD14.h5'
model = load_model(model_path)

def preprocess_image(dicom_path):
    dicom_data = pydicom.dcmread(dicom_path)
    original_image_array = dicom_data.pixel_array.astype(np.float32)
    
    # 調整窗口級/窗口寬
    window_center = dicom_data.WindowCenter if 'WindowCenter' in dicom_data else None
    window_width = dicom_data.WindowWidth if 'WindowWidth' in dicom_data else None
    if window_center and window_width:
        vmin = window_center - window_width // 2
        vmax = window_center + window_width // 2
        original_image_array = np.clip(original_image_array, vmin, vmax)
        original_image_array = (original_image_array - vmin) / (vmax - vmin) * 255.0

    original_image_array = original_image_array.astype(np.uint8)

    # 處理用於預測的圖像
    processed_image_array = cv2.resize(original_image_array, (224, 224))
    processed_image_array = cv2.cvtColor(processed_image_array, cv2.COLOR_GRAY2BGR)
    
    return original_image_array, processed_image_array

@app.route("/model_input")
def model_input():
    return render_template("model_input.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if file:
        dicom_path = "temp_image.dcm"
        file.save(dicom_path)
        original_image_array, processed_image_array = preprocess_image(dicom_path)

        # 记录总体开始时间
        start_time = time.time()

        # 记录图像上传开始时间
        start_upload_time = time.time()
        
        
        # 將原始圖像保存為PNG並編碼為Base64
        buffer = BytesIO()
        original_image = Image.fromarray(original_image_array.astype('uint8'))
        original_image.save(buffer, format="PNG")
        base64_image = base64.b64encode(buffer.getvalue()).decode()

        # 將處理過的圖像餵入模型進行預測
        processed_image_array = np.expand_dims(processed_image_array, axis=0)
        processed_image_array = processed_image_array / 255.0
        prediction = model.predict(processed_image_array).tolist()

        # 记录图像上传结束时间
        end_upload_time = time.time()

        # 计算图像上传时间
        upload_time = end_upload_time - start_upload_time

        # 记录总体结束时间
        end_time = time.time()

        # 计算总体运行时间
        total_time = end_time - start_time

        result = {
            "image": f"data:image/png;base64,{base64_image}",
            "other": prediction[0][0],
            "pneumonia": prediction[0][1],
            "pulmonary_edema": prediction[0][2],
            "atelectasis": prediction[0][3]
        }

        # 记录耗时信息到日志
        logger.info("Upload time: %.4f seconds", upload_time)
        logger.info("Total time: %.4f seconds", total_time)

        return jsonify(result)
    
#針對PNG做圖片前處理
def preprocess_png_image(png_path):
    # 使用 PIL 讀取 PNG 圖片
    original_image = Image.open(png_path)
    original_image_array = np.array(original_image)
    
    # 如果圖片是灰度的，將其轉換為 BGR
    if len(original_image_array.shape) == 2:
        processed_image_array = cv2.cvtColor(original_image_array, cv2.COLOR_GRAY2BGR)
    else:
        processed_image_array = original_image_array

    # 處理用於預測的圖像
    processed_image_array = cv2.resize(processed_image_array, (224, 224))
    
    return original_image_array, processed_image_array



# 初始化 LINE Bot API 和 WebhookHandler
line_bot_api = LineBotApi(config.get('line-bot', 'channel_access_token'))
handler = WebhookHandler(config.get('line-bot', 'channel_secret'))

@app.route("/line_webhook", methods=['POST'])
def line_webhook():
    # 從 HTTP 請求中獲取 X-Line-Signature 頭的值
    signature = request.headers['X-Line-Signature']

    # 獲取請求主體
    body = request.get_data(as_text=True)

    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        return 'Invalid signature. Please check your channel access token/channel secret.', 400

    return 'OK'

@handler.add(MessageEvent, message=TextMessage)
def handle_image_message(event):
    if event.message.text == "使用說明":
        reply_msg = "請上傳X光片的圖片檔，系統會依據您提供的圖片進行判斷，並回覆您肺部病症預測結果"
    elif event.message.text == "我們的網站":
        reply_msg = "Http..........."
    elif event.message.text == "聯絡我們":
        # 建立快速按鈕
        quick_reply = QuickReply(items=[
            QuickReplyButton(action=MessageAction(label="組長：楊致", text="Email：lovedad1018@gmail.com")),
            QuickReplyButton(action=MessageAction(label="組員：陳玥年", text="Email：a829516@gmail.com")),
            QuickReplyButton(action=MessageAction(label="組員：陳昭宇", text="Email：chaoyu0314@gmail.com")),
            QuickReplyButton(action=MessageAction(label="組員：周伯儒", text="Email：chou8102@gmail.com")),
            QuickReplyButton(action=MessageAction(label="組員：粘馨云", text="Email：hsingyun0813@gmail.com")),
            QuickReplyButton(action=MessageAction(label="組員：黃柏章", text="Email：iutvoo0936@gmail.com"))
        ])
        reply_msg = "請選擇要聯絡的組員"
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply_msg, quick_reply=quick_reply))
    elif"@gmail.com" in event.message.text:
        if event.message.text == "Email：lovedad1018@gmail.com":
            reply_msg="履歷參考：https://drive.google.com/drive/shared-with-me"                                                     
        elif event.message.text == "Email：a829516@gmail.com":
            reply_msg="履歷參考：https://pda.104.com.tw/profile/share/dhX2z6rGCOBjyeGBX2aEPMFBMsT7A7wM"                                                     
        elif event.message.text == "Email：chaoyu0314@gmail.com":
            reply_msg="履歷參考："                                                     
       
        elif event.message.text == "Email：hsingyun0813@gmail.com":
            reply_msg="履歷參考：https://drive.google.com/drive/u/0/folders/1gVXNnEwXMNueSlJAs7DS_s4ybJFXLd66"
        elif event.message.text == "Email：iutvoo0936@gmail.com":
            reply_msg="履歷參考："
    else:
        reply_msg = "請上傳圖片檔"

    line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply_msg))

@handler.add(MessageEvent, message=ImageMessage)
def handle_image(event):
    # 從 LINE 伺服器下載圖片
    message_content = line_bot_api.get_message_content(event.message.id)
    with open('temp_image.png', 'wb') as fd:   # 注意這裡的檔名已經從 .dcm 改成 .png
        for chunk in message_content.iter_content():
            fd.write(chunk)

    # 對圖片進行預測
    original_image_array, processed_image_array = preprocess_png_image('temp_image.png')  # 使用新的 PNG 處理函數
    processed_image_array = np.expand_dims(processed_image_array, axis=0)
    processed_image_array = processed_image_array / 255.0
    prediction = model.predict(processed_image_array).tolist()

    # 根據預測結果生成回覆
    reply_msg = f"""
    預測結果:
    其他病症可能性: {(prediction[0][0] * 100):.2f}%
    肺炎可能性: {(prediction[0][1] * 100):.2f}%
    肺水腫可能性: {(prediction[0][2] * 100):.2f}%
    肺不張可能性: {(prediction[0][3] * 100):.2f}%
    結果僅基於X光片的分析，並不代表最終診斷。建議您儘早與專業醫生進行進一步診斷和討論，以確定您的肺部健康狀況
    """
    line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply_msg))


if __name__ == "__main__":  
    app.run(host='0.0.0.0', port=8001)