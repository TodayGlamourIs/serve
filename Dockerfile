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
COPY requirements.txt .
COPY config.ini .

# 安装依赖项
RUN pip install --no-cache-dir -r requirements.txt


CMD ["python", "app.py"]