import os
import tempfile
import torch
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
WEBHOOK_URL = os.getenv('WEBHOOK_URL')

if not BOT_TOKEN or BOT_TOKEN == 'YOUR_LOCAL_BOT_TOKEN':
    raise ValueError("❌ يجب تعيين BOT_TOKEN")

bot = TeleBot(BOT_TOKEN)

# مجلد النماذج
MODEL_DIR = '/app/models' if WEBHOOK_URL else './models'
os.makedirs(MODEL_DIR, exist_ok=True)
os.environ['TORCH_HOME'] = MODEL_DIR

user_files = {}

# =============================
# إشعار التشغيل
# =============================
def notify_startup(mode="محلي"):
    print(f"✅ البوت يعمل في الوضع {mode}!")

# =============================
# معالجة الملفات
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
        suffix = '.mp3'
        if 'wav' in mime_type:
            suffix = '.wav'
        elif 'ogg' in mime_type:
            suffix = '.ogg'
        tmp_dir = '/tmp' if WEBHOOK_URL else '.'
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=tmp_dir) as tmp:
            tmp.write(downloaded_file)
            input_path = tmp.name
        user_files[message.chat.id] = input_path
        markup = InlineKeyboardMarkup()
        markup.row(
            InlineKeyboardButton("🎤 صوت المغني", callback_data="vocals"),
            InlineKeyboardButton("🎶 الموسيقى فقط", callback_data="accompaniment")
        )
        bot.send_message(message.chat.id, "اختر ما تريد استخراجه:", reply_markup=markup)
    except Exception as e:
        bot.reply_to(message, "❌ خطأ في التحميل.")
        print(f"خطأ التحميل: {e}")

# =============================
# معالجة الأزرار
# =============================
@bot.callback_query_handler(func=lambda call: True)
def handle_choice(call):
    chat_id = call.message.chat.id
    choice = call.data
    if chat_id not in user_files:
        bot.answer_callback_query(call.id, "❌ لم يتم العثور على ملف.", show_alert=True)
        return
    input_path = user_files[chat_id]
    bot.edit_message_text("يتم المعالجة... قد يستغرق 30–90 ثانية.", chat_id, call.message.id)
    try:
        # ✅ استخدام htdemucs (لا يحتاج diffq)
        model = get_model('htdemucs')
        model.cpu()
        wav = AudioFile(input_path).read(streams=0, samplerate=model.samplerate)
        ref = wav.mean(0)
        wav = (wav - ref.mean()) / ref.std()
        sources = apply_model(model, wav[None], device='cpu', shifts=1, split=True)[0]
        sources = sources * ref.std() + ref.mean()
        if choice == 'vocals':
            output_audio = sources[model.sources.index('vocals')]
            caption = "🎤 تم استخراج صوت المغني!"
        else:
            other_indices = [i for i, src in enumerate(model.sources) if src != 'vocals']
            accompaniment = torch.stack([sources[i] for i in other_indices]).sum(0)
            output_audio = accompaniment
            caption = "🎶 تم استخراج الموسيقى بدون صوت!"
        output_dir = '/tmp' if WEBHOOK_URL else '.'
        output_path = tempfile.mktemp(suffix='.mp3', dir=output_dir)
        save_audio(output_audio, output_path, samplerate=model.samplerate, bitrate=192)
        with open(output_path, 'rb') as f:
            bot.send_audio(chat_id, f, caption=caption)
        os.remove(input_path)
        os.remove(output_path)
        user_files.pop(chat_id, None)
    except Exception as e:
        error_msg = f"❌ خطأ: {str(e)[:150]}"
        bot.send_message(chat_id, error_msg)
        print(f"خطأ المعالجة: {e}")
        if os.path.exists(input_path):
            os.remove(input_path)
        user_files.pop(chat_id, None)

# =============================
# التشغيل (محلي أو Webhook)
# =============================
if __name__ == '__main__':
    if WEBHOOK_URL:
        from flask import Flask, request
        app = Flask(__name__)
        @app.route(f'/{BOT_TOKEN}', methods=['POST'])
        def webhook():
            if request.headers.get('content-type') == 'application/json':
                json_string = request.get_data().decode('utf-8')
                update = telebot.types.Update.de_json(json_string)
                bot.process_new_updates([update])
                return 'OK', 200
            return 'Unsupported Media Type', 415
        bot.remove_webhook()
        bot.set_webhook(url=WEBHOOK_URL + '/' + BOT_TOKEN)
        notify_startup("سحابي (Webhook)")
        app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 10000)))
    else:
        notify_startup("محلي (Polling)")
        bot.polling(none_stop=True, timeout=60)