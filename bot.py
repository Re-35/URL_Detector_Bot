import os
import logging
import asyncio
import pandas as pd
import numpy as np
import tldextract as tld
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO, StringIO
from PIL import Image
import joblib
import xgboost as xgb
from sklearn.base import BaseEstimator, TransformerMixin
from urllib.parse import urlparse
from collections import Counter
import re
import math
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, BotCommand
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, ContextTypes, filters
from telegram.constants import ParseMode

# Configure logging with more detail
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

print("🚀 Starting URL Detector Bot...")
print(f"📁 Current working directory: {os.getcwd()}")
print(f"📋 Files in directory: {os.listdir('.')}")

# Your BuyMeACoffee URL
BUYMEACOFFEE_URL = "https://buymeacoffee.com/re35"

class FeatureExtract(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.extentions = ['com', 'org', 'net', 'gov', 'us', 'sa', 'uk', 'ca', 'au', 'de', 'jp', 'cn', 'ae', 'ch', 'nl', 'edu', 'int', 'ai']
        self.suspicios_words = ['login','account', 'free', 'win', 'verify', 'secure', 'bank', 'update', 'pypal', 'invoice', 'payment', 'signin', 'renew', 'reward', 'promo', 'prize', 'winner', 'gift', 'safe', 'crack', 'serial', 'torrent', 'patch', 'exe', 'download', 'google', 'microsoft', 'amazon', 'gov', 'apple', 'support', 'customer', 'service', 'absher', 'alrajhi', 'alahli', 'samba', 'riyad', 'alinma', 'gosi', 'moh', 'moi', 'hrsd', 'zakat', 'nis', 'stc', 'noon', 'extra', 'jarir', 'mobily', 'zain']

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        self.urls = x.iloc[:,0].astype(str) if not isinstance(x, np.ndarray) else x[:,0].astype(str)
        self.urls = self.urls.str.lower() if not isinstance(x, np.ndarray) else np.char.lower(self.urls)
        
        features = []
        for u in self.urls:
            length = len(u)
            url_parsed = urlparse(u)
            ext = tld.extract(u).suffix
            ext = ext.split('.')[1] if ('.' in ext) else ext
            subdomain = tld.extract(u).subdomain
            s = subdomain.count('.')
            path = url_parsed.path.strip('/')
            query = url_parsed.query
            suspicios_count = sum(1 for word in re.split(r'[/-]', u) if word in self.suspicios_words)
            count = Counter(u)
            entropy = -np.sum((i/length) * math.log2(i/length) for i in count.values())
            
            f = [
                length,
                len(path) if path is not None else 0,
                len(query) if query is not None else 0,
                len(tld.extract(u).domain),
                s+1,
                u.count('-') if u.count('-') > 2 else 0,
                u.count('.'),
                u.count('/'),
                u.count('%'),
                suspicios_count,
                entropy,
                0 if any(c.isdigit() for c in path) else 1,
                1 if re.search(r'//\d+\.\d+\.\d+\.\d+', u) else 0,
                1 if 'https' in u else 0,
                1 if ext in self.extentions else 0
            ]
            features.append(f)
        return np.array(features)

# Load the model with better error handling
print("📦 Loading machine learning model...")
try:
    if os.path.exists('url_model.joblib'):
        model = joblib.load('url_model.joblib')
        print("✅ Model loaded successfully")
        logger.info("Model loaded successfully")
    else:
        print("❌ Model file 'url_model.joblib' not found!")
        print(f"Available files: {os.listdir('.')}")
        raise FileNotFoundError("Model file not found")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    logger.error(f"Failed to load model: {e}")
    exit(1)

# Country dictionaries
dic_en = {'sa': 'Saudi Arabia', 'ae': 'United Arab Emirates', 'kw': 'Kuwait', 'om': 'Oman', 'qa': 'Qatar', 'bh': 'Bahrain', 'us': 'United States', 'ca': 'Canada', 'uk': 'United Kingdom', 'de': 'Germany', 'fr': 'France', 'es': 'Spain', 'it': 'Italy', 'ru': 'Russia', 'cn': 'China', 'jp': 'Japan', 'kr': 'South Korea', 'in': 'India', 'au': 'Australia', 'br': 'Brazil', 'za': 'South Africa', 'mx': 'Mexico', 'ar': 'Argentina', 'eg': 'Egypt', 'tr': 'Turkey', 'nl': 'Netherlands', 'se': 'Sweden', 'no': 'Norway', 'fi': 'Finland', 'dk': 'Denmark', 'pl': 'Poland', 'ch': 'Switzerland', 'be': 'Belgium', 'nz': 'New Zealand', 'sg': 'Singapore', 'my': 'Malaysia', 'id': 'Indonesia', 'ph': 'Philippines', 'ng': 'Nigeria', 'pk': 'Pakistan', 'Not defined': 'Not defined'}

dic_ar = {'sa': 'المملكة العربية السعودية', 'ae': 'الإمارات العربية المتحدة', 'kw': 'الكويت', 'om': 'عُمان', 'qa': 'قطر', 'bh': 'البحرين', 'us': 'الولايات المتحدة الأمريكية', 'ca': 'كندا', 'uk': 'المملكة المتحدة', 'de': 'ألمانيا', 'fr': 'فرنسا', 'es': 'إسبانيا', 'it': 'إيطاليا', 'ru': 'روسيا', 'cn': 'الصين', 'jp': 'اليابان', 'kr': 'كوريا الجنوبية', 'in': 'الهند', 'au': 'أستراليا', 'br': 'البرازيل', 'za': 'جنوب أفريقيا', 'mx': 'المكسيك', 'ar': 'الأرجنتين', 'eg': 'مصر', 'tr': 'تركيا', 'nl': 'هولندا', 'se': 'السويد', 'no': 'النرويج', 'fi': 'فنلندا', 'dk': 'الدانمارك', 'pl': 'بولندا', 'ch': 'سويسرا', 'be': 'بلجيكا', 'nz': 'نيوزيلندا', 'sg': 'سنغافورة', 'my': 'ماليزيا', 'id': 'إندونيسيا', 'ph': 'الفلبين', 'ng': 'نيجيريا', 'pk': 'باكستان', 'Not defined': 'غير معروف'}

# User language storage
user_languages = {}

def get_language_keyboard():
    """Create language selection keyboard"""
    keyboard = [
        [InlineKeyboardButton("🇬🇧 English", callback_data='lang_en')],
        [InlineKeyboardButton("🇸🇦 العربية", callback_data='lang_ar')]
    ]
    return InlineKeyboardMarkup(keyboard)

def get_main_keyboard(lang):
    """Create main menu keyboard based on language"""
    if lang == 'ar':
        keyboard = [
            [InlineKeyboardButton("🔗 فحص رابط واحد", callback_data='check_single')],
            [InlineKeyboardButton("📄 فحص ملف CSV", callback_data='check_file')],
            [InlineKeyboardButton("☕ ادعمني", callback_data='donate')],
            [InlineKeyboardButton("🌐 تغيير اللغة", callback_data='change_lang')]
        ]
    else:
        keyboard = [
            [InlineKeyboardButton("🔗 Check Single URL", callback_data='check_single')],
            [InlineKeyboardButton("📄 Check CSV File", callback_data='check_file')],
            [InlineKeyboardButton("☕ Support Me", callback_data='donate')],
            [InlineKeyboardButton("🌐 Change Language", callback_data='change_lang')]
        ]
    return InlineKeyboardMarkup(keyboard)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Start command handler"""
    user_id = update.effective_user.id
    
    if user_id not in user_languages:
        welcome_text = """🌟 Welcome to URL Detector Bot! / مرحباً بك في بوت فاحص الروابط! 🌟

🔍 This bot helps protect your data by detecting suspicious URLs
🔍 هذا البوت يساعد في حماية بياناتك من خلال اكتشاف الروابط المشبوهة

⚠️ **Important Notice / تنبيه مهم:**
• The AI model provides helpful guidance but may not be 100% accurate
• Always use your judgment when clicking links
• نموذج الذكاء الاصطناعي يقدم إرشادات مفيدة لكنه قد لا يكون دقيقاً بنسبة 100%
• استخدم دائماً حكمك الشخصي عند التعامل مع الروابط

🌐 Please choose your language / يرجى اختيار لغتك:"""
        
        await update.message.reply_text(
            welcome_text,
            reply_markup=get_language_keyboard()
        )
    else:
        lang = user_languages[user_id]
        if lang == 'ar':
            text = """🔍 *فاحص الروابط*

🛡️ احم بياناتك من الروابط المشبوهة

⚠️ *تنبيه مهم:*
النموذج يقدم إرشادات مفيدة لكن قد لا يكون دقيقاً بنسبة 100%. استخدم حكمك الشخصي دائماً عند التعامل مع الروابط.

اختر الخدمة المطلوبة:"""
        else:
            text = """🔍 *URL Detector*

🛡️ Protect your data from malicious URLs

⚠️ *Important Notice:*
The model provides helpful guidance but may not be 100% accurate. Always use your personal judgment when dealing with URLs.

Choose the service you need:"""
        
        await update.message.reply_text(
            text,
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=get_main_keyboard(lang)
        )

async def language_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle language selection"""
    query = update.callback_query
    await query.answer()
    
    user_id = query.from_user.id
    lang = query.data.split('_')[1]
    user_languages[user_id] = lang
    
    if lang == 'ar':
        text = "🔍 *فاحص الروابط*\n\nاحم بياناتك من الروابط المشبوهة\n\nاختر الخدمة المطلوبة:"
    else:
        text = "🔍 *URL Detector*\n\nProtect your data from malicious URLs\n\nChoose the service you need:"
    
    await query.edit_message_text(
        text,
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=get_main_keyboard(lang)
    )

async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle button callbacks"""
    query = update.callback_query
    await query.answer()
    
    user_id = query.from_user.id
    lang = user_languages.get(user_id, 'en')
    
    if query.data == 'check_single':
        if lang == 'ar':
            text = "🔗 *فحص رابط واحد*\n\nأرسل الرابط الذي تريد فحصه:"
        else:
            text = "🔗 *Check Single URL*\n\nSend the URL you want to check:"
        
        context.user_data['waiting_for'] = 'single_url'
        await query.edit_message_text(text, parse_mode=ParseMode.MARKDOWN)
        
    elif query.data == 'check_file':
        if lang == 'ar':
            text = "📄 *فحص ملف CSV*\n\nأرسل ملف CSV يحتوي على عمود باسم 'url'"
        else:
            text = "📄 *Check CSV File*\n\nSend a CSV file containing a column named 'url'"
        
        context.user_data['waiting_for'] = 'csv_file'
        await query.edit_message_text(text, parse_mode=ParseMode.MARKDOWN)
        
    elif query.data == 'donate':
        if lang == 'ar':
            text = f"☕ *ادعمني*\n\nإذا أعجبك البوت، يمكنك دعمي:\n{BUYMEACOFFEE_URL}\n\nشكراً لك! ❤️"
        else:
            text = f"☕ *Support Me*\n\nIf you like this bot, you can support me:\n{BUYMEACOFFEE_URL}\n\nThank you! ❤️"
        
        await query.edit_message_text(text, parse_mode=ParseMode.MARKDOWN)
        
    elif query.data == 'change_lang':
        await query.edit_message_text(
            "Please choose your language:\nيرجى اختيار لغتك:",
            reply_markup=get_language_keyboard()
        )

def predict_single_url(url, lang='en'):
    """Predict single URL safety"""
    if not url.strip():
        return 'Please enter a valid URL' if lang == 'en' else 'يرجى إدخال رابط صحيح'
    
    try:
        # Get country
        country = tld.extract(url).suffix
        c_key = country if len(country) == 2 else 'Not defined'
        c_name = dic_en.get(c_key, 'Not defined') if lang == 'en' else dic_ar.get(c_key, 'غير معروف')
        
        # Make prediction
        prediction = model.predict(np.array([[url]]))[0]
        if lang == 'ar':
            result = 'الرابط آمن ✅' if prediction == 0 else 'كن حذراً الرابط مشبوه ⚠️'
            return f'النتيجة: {result}\nالدولة: {c_name}'
        else:
            result = 'The URL is safe ✅' if prediction == 0 else 'Be careful, the URL is suspicious ⚠️'
            return f'Result: {result}\nCountry: {c_name}'
    except Exception as e:
        logger.error(f"Error predicting URL: {e}")
        return f'Error: {str(e)}'

async def process_csv_file(file_content, lang='en'):
    """Process CSV file and return results"""
    try:
        # Read CSV
        df = pd.read_csv(StringIO(file_content))
        
        if 'url' not in df.columns:
            if lang == 'ar':
                return None, None, None, 'يجب أن يحتوي الملف على عمود باسم "url"'
            else:
                return None, None, None, 'The file must contain a column named "url"'
        
        # Limit to 1000 URLs to avoid timeout
        if len(df) > 1000:
            df = df.head(1000)
        
        # Extract countries
        df['country_code'] = df['url'].apply(lambda u: tld.extract(u).suffix if len(tld.extract(u).suffix) == 2 else 'Not defined')
        df['country'] = df['country_code'].map(dic_ar if lang == 'ar' else dic_en)
        
        # Make predictions
        predictions = model.predict(df[['url']])
        df['pred'] = predictions
        
        if lang == 'ar':
            df['result'] = df['pred'].map({0: 'آمن ✅', 1: 'مشبوه ⚠️'})
        else:
            df['result'] = df['pred'].map({0: 'Safe ✅', 1: 'Suspicious ⚠️'})
        
        # Create plots
        plt.style.use('default')
        
        # Plot 1: Overall distribution
        plt.figure(figsize=(8, 6))
        counts = df['result'].value_counts()
        colors = ['#4CAF50', '#F44336']
        plt.pie(counts.values, labels=counts.index, autopct='%1.1f%%', colors=colors, startangle=90)
        
        if lang == 'ar':
            plt.title('نسبة الروابط المشبوهة والآمنة', fontsize=16, pad=20)
        else:
            plt.title('Distribution of Safe and Suspicious URLs', fontsize=16, pad=20)
        
        # Save plot 1
        buf1 = BytesIO()
        plt.tight_layout()
        plt.savefig(buf1, format='png', dpi=300, bbox_inches='tight')
        plt.close()
        buf1.seek(0)
        
        # Plot 2: Country distribution
        plt.figure(figsize=(12, 8))
        top_countries = df['country'].value_counts().head(15)
        country_result = df[df['country'].isin(top_countries.index)].groupby(['country', 'result']).size().unstack(fill_value=0)
        
        country_result.plot(kind='bar', color=['#4CAF50', '#F44336'], figsize=(12, 8))
        
        if lang == 'ar':
            plt.title('توزيع الروابط حسب الدول', fontsize=16, pad=20)
            plt.xlabel('الدول', fontsize=12)
            plt.ylabel('عدد الروابط', fontsize=12)
        else:
            plt.title('URL Distribution by Countries', fontsize=16, pad=20)
            plt.xlabel('Countries', fontsize=12)
            plt.ylabel('Number of URLs', fontsize=12)
        
        plt.xticks(rotation=45, ha='right')
        plt.legend()
        
        # Save plot 2
        buf2 = BytesIO()
        plt.tight_layout()
        plt.savefig(buf2, format='png', dpi=300, bbox_inches='tight')
        plt.close()
        buf2.seek(0)
        
        # Create results text
        total = len(df)
        safe = sum(df['pred'] == 0)
        suspicious = sum(df['pred'] == 1)
        
        if lang == 'ar':
            results_text = f"""📊 *نتائج تحليل الملف*

📝 إجمالي الروابط: {total}
✅ روابط آمنة: {safe} ({safe/total*100:.1f}%)
⚠️ روابط مشبوهة: {suspicious} ({suspicious/total*100:.1f}%)

🌍 أهم الدول:
"""
            for country, count in top_countries.head(5).items():
                results_text += f"• {country}: {count}\n"
        else:
            results_text = f"""📊 *File Analysis Results*

📝 Total URLs: {total}
✅ Safe URLs: {safe} ({safe/total*100:.1f}%)
⚠️ Suspicious URLs: {suspicious} ({suspicious/total*100:.1f}%)

🌍 Top Countries:
"""
            for country, count in top_countries.head(5).items():
                results_text += f"• {country}: {count}\n"
        
        return results_text, buf1, buf2, None
        
    except Exception as e:
        logger.error(f"Error processing CSV: {e}")
        error_msg = f'خطأ في معالجة الملف: {str(e)}' if lang == 'ar' else f'Error processing file: {str(e)}'
        return None, None, None, error_msg

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle text messages"""
    user_id = update.effective_user.id
    lang = user_languages.get(user_id, 'en')
    waiting_for = context.user_data.get('waiting_for')
    
    if waiting_for == 'single_url':
        url = update.message.text.strip()
        
        if lang == 'ar':
            await update.message.reply_text("🔍 جاري فحص الرابط...")
        else:
            await update.message.reply_text("🔍 Checking URL...")
        
        result = predict_single_url(url, lang)
        await update.message.reply_text(f"*{result}*", parse_mode=ParseMode.MARKDOWN)
        
        context.user_data['waiting_for'] = None
        
        if lang == 'ar':
            text = "هل تريد فحص رابط آخر؟"
        else:
            text = "Would you like to check another URL?"
        
        await update.message.reply_text(text, reply_markup=get_main_keyboard(lang))
    
    else:
        if lang == 'ar':
            text = "🔍 *فاحص الروابط*\n\nاحم بياناتك من الروابط المشبوهة\n\nاختر الخدمة المطلوبة:"
        else:
            text = "🔍 *URL Detector*\n\nProtect your data from malicious URLs\n\nChoose the service you need:"
        
        await update.message.reply_text(
            text,
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=get_main_keyboard(lang)
        )

async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle document uploads"""
    user_id = update.effective_user.id
    lang = user_languages.get(user_id, 'en')
    waiting_for = context.user_data.get('waiting_for')
    
    if waiting_for == 'csv_file':
        document = update.message.document
        
        if not document.file_name.endswith('.csv'):
            if lang == 'ar':
                await update.message.reply_text("⚠️ يرجى إرسال ملف CSV فقط")
            else:
                await update.message.reply_text("⚠️ Please send a CSV file only")
            return
        
        if lang == 'ar':
            await update.message.reply_text("📊 جاري معالجة الملف... قد يستغرق هذا بعض الوقت")
        else:
            await update.message.reply_text("📊 Processing file... This may take a moment")
        
        try:
            file = await context.bot.get_file(document.file_id)
            file_content = await file.download_as_bytearray()
            file_content = file_content.decode('utf-8')
            
            results_text, plot1, plot2, error = await process_csv_file(file_content, lang)
            
            if error:
                await update.message.reply_text(f"⌘ {error}")
                return
            
            await update.message.reply_text(results_text, parse_mode=ParseMode.MARKDOWN)
            
            if plot1:
                caption = "📊 نسبة الروابط الآمنة والمشبوهة" if lang == 'ar' else "📊 Distribution of Safe and Suspicious URLs"
                await update.message.reply_photo(plot1, caption=caption)
            
            if plot2:
                caption = "🌐 توزيع الروابط حسب الدول" if lang == 'ar' else "🌐 URL Distribution by Countries"
                await update.message.reply_photo(plot2, caption=caption)
            
            context.user_data['waiting_for'] = None
            
            text = "تم الانتهاء من تحليل الملف! هل تريد فحص ملف آخر؟" if lang == 'ar' else "File analysis completed! Would you like to check another file?"
            await update.message.reply_text(text, reply_markup=get_main_keyboard(lang))
            
        except Exception as e:
            logger.error(f"Error handling document: {e}")
            error_msg = f'خطأ: {str(e)}' if lang == 'ar' else f'Error: {str(e)}'
            await update.message.reply_text(f"⌘ {error_msg}")
    
    else:
        if lang == 'ar':
            await update.message.reply_text("يرجى اختيار 'فحص ملف CSV' من القائمة أولاً")
        else:
            await update.message.reply_text("Please select 'Check CSV File' from the menu first")

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Help command handler"""
    user_id = update.effective_user.id
    lang = user_languages.get(user_id, 'en')
    
    if lang == 'ar':
        help_text = """🔍 *فاحص الروابط - المساعدة*

*الخدمات المتاحة:*
🔗 فحص رابط واحد - تحقق من أمان رابط معين
📄 فحص ملف CSV - فحص عدة روابط مرة واحدة

*كيفية الاستخدام:*
1️⃣ اختر اللغة المفضلة
2️⃣ اختر الخدمة من القائمة
3️⃣ أرسل الرابط أو الملف
4️⃣ احصل على النتائج

*ملاحظات:*
• يجب أن يحتوي ملف CSV على عمود باسم 'url'
• الحد الأقصى 1000 رابط لكل ملف
• يدعم البوت اللغتين العربية والإنجليزية

☕ يمكنك دعم المطور من خلال الضغط على زر الدعم"""
    else:
        help_text = """🔍 *URL Detector - Help*

*Available Services:*
🔗 Check Single URL - Verify the safety of a specific URL
📄 Check CSV File - Check multiple URLs at once

*How to use:*
1️⃣ Choose your preferred language
2️⃣ Select service from the menu
3️⃣ Send the URL or file
4️⃣ Get the results

*Notes:*
• CSV file must contain a column named 'url'
• Maximum 1000 URLs per file
• Bot supports Arabic and English languages

☕ You can support the developer by clicking the support button"""
    
    await update.message.reply_text(help_text, parse_mode=ParseMode.MARKDOWN)

def main():
    """Main function to run the bot"""
    print("🔑 Checking for BOT_TOKEN...")
    
    # Get token from environment variable
    TOKEN = os.getenv('BOT_TOKEN')
    
    if not TOKEN:
        print("❌ BOT_TOKEN environment variable not found!")
        print("Available environment variables:")
        for key in os.environ.keys():
            if 'TOKEN' in key.upper() or 'BOT' in key.upper():
                print(f"  {key}")
        exit(1)
    
    print("✅ BOT_TOKEN found!")
    print("🤖 Creating Telegram application...")
    
    try:
        # Create application
        application = Application.builder().token(TOKEN).build()
        
        print("📱 Adding handlers...")
        
        # Add handlers
        application.add_handler(CommandHandler("start", start))
        application.add_handler(CommandHandler("help", help_command))
        application.add_handler(CallbackQueryHandler(language_callback, pattern='^lang_'))
        application.add_handler(CallbackQueryHandler(button_callback))
        application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
        application.add_handler(MessageHandler(filters.Document.ALL, handle_document))
        
        # Set bot commands
        commands = [
            BotCommand("start", "Start the bot"),
            BotCommand("help", "Show help message")
        ]
        
        print("🚀 Starting bot with polling...")
        
        # Run the bot
        application.run_polling(allowed_updates=Update.ALL_TYPES)
        
    except Exception as e:
        print(f"❌ Failed to start bot: {e}")
        logger.error(f"Failed to start bot: {e}")
        exit(1)

if __name__ == '__main__':
    main()
