# Dockerfile for Flask App
FROM python:3.9.13

#在映像檔創建一個工作目錄
WORKDIR /app

# 将应用代码复制到容器中
COPY . /app
COPY densenetD14.h5 /app/

# 复制两个文件夹及其内容到工作目录
COPY static /app/static
COPY templates /app/templates

# 安装依赖项
RUN python -m pip install --upgrade pip
RUN apt-get update && apt-get install -y libgl1-mesa-glx
RUN pip install pymongo flask pydicom keras opencv-python numpy Pillow line-bot-sdk tensorflow


CMD ["python", "app.py"]