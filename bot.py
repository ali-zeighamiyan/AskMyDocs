import os
import faiss
import logging
from telegram import Update
from telegram.ext import CommandHandler, MessageHandler, filters, CallbackContext, CallbackQueryHandler, ApplicationBuilder, ContextTypes
from telegram import ReplyKeyboardMarkup, InlineKeyboardButton, InlineKeyboardMarkup
from llm import LLM


TOKEN = "6929830229:AAEXbYO97fey0HwecRuIPFTLXYT-WxzgigI"
directory_path = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(directory_path, "temp")


os.makedirs(BASE_DIR, exist_ok=True)

user_sessions = {}  # Store user-specific LLM instances

def get_user_llm(user_id):
    """Retrieve or create an LLM instance for each user."""
    if user_id not in user_sessions:
        user_sessions[user_id] = LLM(user_id)
    return user_sessions[user_id]

async def save_file(file, file_name, user_id, cluster_name):
    file_dir = os.path.join(BASE_DIR, user_id, cluster_name)
    os.makedirs(file_dir, exist_ok=True)  # Create cluster directory if not exists
    file_path = os.path.join(file_dir, file_name)
    await file.download_to_drive(file_path)
        
# Handle "/start" command
async def start(update: Update, context: CallbackContext):
    await update.message.reply_text("Welcome! Upload a document and specify its cluster.")

async def handle_document(update: Update, context: CallbackContext):
    """Handle document uploads & ask for cluster name."""
    user_id = str(update.message.chat_id)
    file = update.message.document
    file_id = file.file_id
    file_name = file.file_name

    # Ask user for cluster
    await update.message.reply_text(f"Specify a cluster name for {file_name}:")
    
    # Store file temporarily in user_data
    context.user_data["pending_file"] = file_id
    context.user_data["pending_file_name"] = file_name
    context.user_data["state"] = "waiting_for_cluster_name"
async def handle_cluster(update: Update, context: CallbackContext):
    user_id = str(update.message.chat_id)
    cluster = update.message.text
    file_id = context.user_data.get("pending_file")
    file_name = context.user_data.get("pending_file_name")

    if not file_id:
        await update.message.reply_text("No document found. Please upload a file first.")
        return

    # Download file
    file = await context.bot.get_file(file_id)
    await save_file(file, file_name, user_id, cluster)
    await update.message.reply_text(f"Document **{file_name}** saved under cluster **{cluster}**!")
    # Clear temp storage
    context.user_data.pop("pending_file", None)
    context.user_data.pop("pending_file_name", None)
    context.user_data["state"] = None

async def list_clusters(update: Update, context: CallbackContext):
    """Show available clusters for selection."""
    user_id = str(update.message.chat_id)
    user_dir = os.path.join(BASE_DIR, user_id)

    if not os.path.exists(user_dir):
        await update.message.reply_text("You have no clusters yet. Upload a document first.")
        return

    clusters = os.listdir(user_dir)
    if not clusters:
        await update.message.reply_text("No clusters found.")
        return
    context.user_data["state"] = "waiting_for_cluster_selection"

    # Show clusters as options
    keyboard = [[cluster] for cluster in clusters]
    reply_markup = ReplyKeyboardMarkup(keyboard, one_time_keyboard=True)

    await update.message.reply_text("Select a cluster:", reply_markup=reply_markup)

async def list_files(update: Update, context: CallbackContext):
    """Show available files in the selected cluster."""
    user_id = str(update.message.chat_id)
    cluster = update.message.text
    user_cluster_dir = os.path.join(BASE_DIR, user_id, cluster)

    if not os.path.exists(user_cluster_dir):
        await update.message.reply_text("Cluster not found. Please try again.")
        return

    files = os.listdir(user_cluster_dir)
    if not files:
        await update.message.reply_text(f"No files found in cluster **{cluster}**.")
        return

    # Save selected cluster & show files
    context.user_data["selected_cluster"] = cluster
    keyboard = [[InlineKeyboardButton(file, callback_data=f"select_file:{file}")] for file in files]
    keyboard.append([InlineKeyboardButton("✅ Done Selecting", callback_data="done_selecting")])
    reply_markup = InlineKeyboardMarkup(keyboard)


    await update.message.reply_text(f"Select files from **{cluster}**:", reply_markup=reply_markup)

async def process_question(update: Update, context: CallbackContext):
    """Process user question using FAISS retrieval."""
    user_id = str(update.message.chat_id)
    selected_files = context.user_data.get("selected_files", [])
    question = update.message.text

    if not selected_files:
        await update.message.reply_text("Please select files first.")
        return

    # Process embeddings & store in FAISS
    print("get user llm instance ...")
    llm: LLM = get_user_llm(user_id)
    print("process the doc ...")
    llm.load_or_save_doc(cluster_name=context.user_data["selected_cluster"], selected_docs_name=selected_files)
    print("run the chain ...")
    ai_answer = llm.run_chain(selected_docs_name=selected_files, question=question)
    await update.message.reply_text(f"Answer: {ai_answer}")

async def select_file_callback(update: Update, context: CallbackContext):
    """Handle multi-file selection."""
    query = update.callback_query
    file_name = query.data.split(":")[1]

    selected_files = context.user_data.get("selected_files", [])
    if file_name not in selected_files:
        selected_files.append(file_name)
    else:
        selected_files.remove(file_name)  # Toggle selection

    context.user_data["selected_files"] = selected_files
    await query.answer(f"Selected: {', '.join(selected_files)}")
    
async def handle_text_response(update: Update, context: ContextTypes.DEFAULT_TYPE):
    state = context.user_data.get("state")

    if state == "waiting_for_cluster_name":
        await handle_cluster(update, context)
    elif state == "waiting_for_cluster_selection":
        await list_files(update, context)
    elif state == "waiting_for_question":
        await process_question(update, context)
    else:
        await update.message.reply_text("Sorry, I didn’t understand that. Use /start or /list_clusters to begin.")

async def done_selecting_callback(update: Update, context: ContextTypes):
    query = update.callback_query
    context.user_data["state"] = "waiting_for_question"
    await query.answer("OK, now ask your question")

def main():
    application = ApplicationBuilder().token(TOKEN).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("list_clusters", list_clusters))
    application.add_handler(MessageHandler(filters.ATTACHMENT, handle_document))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text_response))

    application.add_handler(CallbackQueryHandler(select_file_callback, pattern=r"^select_file:"))
    application.add_handler(CallbackQueryHandler(done_selecting_callback, pattern="^done_selecting$"))


    application.run_polling()

if __name__ == "__main__":
    main()
