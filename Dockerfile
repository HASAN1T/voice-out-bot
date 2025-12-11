FROM python:3.11-slim

# تثبيت ffmpeg (مهم لقراءة التنسيقات الصوتية)
RUN apt-get update && \
    apt-get install -y ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# إعداد مجلد العمل
WORKDIR /app

# نسخ المتطلبات أولاً (للاستفادة من الـ cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# نسخ باقي الملفات
COPY . .

# إنشاء مجلدات مؤقتة
RUN mkdir -p /tmp /app/models

# تعيين TORCH_HOME لحفظ النموذج داخل контейнер
ENV TORCH_HOME=/app/models

# تشغيل البوت
CMD ["python", "bot.py"]