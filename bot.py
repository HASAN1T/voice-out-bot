import os
import tempfile
import torch
from flask import Flask, request
from demucs.pretrained import get_model
from demucs.apply import apply_model
from demucs.audio import AudioFile, save_audio
from telebot import TeleBot
from telebot.types import InlineKeyboardMarkup, InlineKeyboardButton

# =============================
# الإعدادات
# =============================
BOT_TOKEN = os.getenv('BOT_TOKEN')
YOUR_CHAT_ID = os.getenv('YOUR_CHAT_ID')
WEBHOOK_URL = os.getenv('WEBHOOK_URL')  # مثال: https://my-bot.onrender.com

if not BOT_TOKEN or BOT_TOKEN == 'YOUR_LOCAL_BOT_TOKEN':
    raise ValueError("❌ يجب تعيين BOT_TOKEN")

bot = TeleBot(BOT_TOKEN)
app = Flask(__name__)

# مجلد لحفظ نموذج demucs
MODEL_DIR = '/app/models' if WEBHOOK_URL else './models'
os.makedirs(MODEL_DIR, exist_ok=True)
os.environ['TORCH_HOME'] = MODEL_DIR

# تخزين مؤقت للملفات حسب chat_id
user_files = {}

# =============================
# دالة إشعار التشغيل
# =============================
def notify_startup(mode):
    print(f"✅ البوت يعمل في الوضع: {mode}")

# =============================
# معالجة الملفات الصوتية
# =============================
@bot.message_handler(content_types=['audio', 'document'])
def handle_audio(message):
    if message.audio:
        file_id = message.audio.file_id
        mime_type = 'audio/mpeg'
    elif message.document and message.document.mime_type and message.document.mime_type.startswith('audio/'):
        file_id = message.document.file_id
        mime_type = message.document.mime_type
    else:
        bot.reply_to(message, "⚠️ يرجى إرسال ملف صوتي فقط (MP3, WAV, OGG).")
        return

    bot.reply_to(message, "جارٍ تحميل الملف الصوتي...")

    try:
        file_info = bot.get_file(file_id)
        downloaded_file = bot.download_file(file_info.file_path)

        # تحديد الامتداد
        suffix = '.mp3'
        if 'wav' in mime_type:
            suffix = '.wav'
        elif 'ogg' in mime_type:
            suffix = '.ogg'

        # مجلد مؤقت مناسب (Linux على Render / محلي)
        tmp_dir = '/tmp' if WEBHOOK_URL else '.'

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=tmp_dir) as tmp:
            tmp.write(downloaded_file)
            input_path = tmp.name

        user_files[message.chat.id] = input_path

        # أزرار الاختيار
        markup = InlineKeyboardMarkup()
        markup.row(
            InlineKeyboardButton("🎤 صوت المغني", callback_data="vocals"),
            InlineKeyboardButton("🎶 الموسيقى فقط", callback_data="accompaniment")
        )
        bot.send_message(message.chat.id, "اختر ما تريد استخراجه:", reply_markup=markup)

    except Exception as e:
        bot.reply_to(message, "❌ حدث خطأ أثناء تحميل الملف.")
        print(f"خطأ التحميل: {e}")

# =============================
# معالجة الضغط على الأزرار
# =============================
@bot.callback_query_handler(func=lambda call: True)
def handle_choice(call):
    chat_id = call.message.chat.id
    choice = call.data

    if chat_id not in user_files:
        bot.answer_callback_query(call.id, "❌ لم يتم العثور على ملف صوتي.", show_alert=True)
        return

    input_path = user_files[chat_id]
    bot.edit_message_text("يتم المعالجة... قد يستغرق 30–90 ثانية.", chat_id, call.message.id)

    try:
        # تحميل النموذج (htdemucs لا يحتاج diffq)
        model = get_model('htdemucs')
        model.cpu()

        # تحميل الصوت
        wav = AudioFile(input_path).read(streams=0, samplerate=model.samplerate)
        ref = wav.mean(0)
        wav = (wav - ref.mean()) / ref.std()
        sources = apply_model(model, wav[None], device='cpu', shifts=1, split=True)[0]
        sources = sources * ref.std() + ref.mean()

        # اختيار الناتج
        if choice == 'vocals':
            output_audio = sources[model.sources.index('vocals')]
            caption = "🎤 تم استخراج صوت المغني!"
        else:
            other_indices = [i for i, src in enumerate(model.sources) if src != 'vocals']
            accompaniment = torch.stack([sources[i] for i in other_indices]).sum(0)
            output_audio = accompaniment
            caption = "🎶 تم استخراج الموسيقى بدون صوت!"

        # حفظ الملف الناتج
        output_dir = '/tmp' if WEBHOOK_URL else '.'
        output_path = tempfile.mktemp(suffix='.mp3', dir=output_dir)
        save_audio(output_audio, output_path, samplerate=model.samplerate, bitrate=192)

        # إرسال النتيجة
        with open(output_path, 'rb') as f:
            bot.send_audio(chat_id, f, caption=caption)

        # تنظيف
        os.remove(input_path)
        os.remove(output_path)
        user_files.pop(chat_id, None)

    except Exception as e:
        error_msg = f"❌ خطأ أثناء المعالجة: {str(e)[:150]}"
        bot.send_message(chat_id, error_msg)
        print(f"خطأ المعالجة: {e}")
        if os.path.exists(input_path):
            os.remove(input_path)
        user_files.pop(chat_id, None)

# =============================
# نقطة نهاية Webhook
# =============================
@app.route(f'/{BOT_TOKEN}', methods=['POST'])
def webhook():
    if request.headers.get('content-type') == 'application/json':
        json_str = request.get_data().decode('utf-8')
        update = TeleBot.de_json(json_str)
        bot.process_new_updates([update])
        return 'OK', 200
    return 'Unsupported Media Type', 415

# =============================
# بدء التشغيل
# =============================
if __name__ == '__main__':
    if WEBHOOK_URL:
        # ============== وضع Render (Webhook) ==============
        # إزالة أي webhook سابق
        bot.remove_webhook()
        # تعيين webhook جديد
        bot.set_webhook(url=f"{WEBHOOK_URL.rstrip('/')}/{BOT_TOKEN}")
        notify_startup("سحابي (Webhook عبر Flask)")
        # تشغيل خادم Flask على المنفذ المطلوب
        port = int(os.environ.get('PORT', 10000))
        app.run(host='0.0.0.0', port=port)
    