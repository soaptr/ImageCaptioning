import logging
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler, filters
from neuro import ImageCaption


model, tokenizer = ImageCaption.load()

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(chat_id=update.effective_chat.id,
                                   text="I'm a image caption bot, you can send me a " \
                                        "picture and I'll tell you what I see on it.")


async def caption(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # await context.bot.send_message(chat_id=update.effective_chat.id, text=update.message.text)
    file_id = update.message.photo[-1].file_id  # message.document.file_id
    new_file = await context.bot.get_file(file_id)
    #file_path = await new_file.download_to_drive()
    image = ImageCaption.transform_image(new_file.file_path)
    caption_text = ImageCaption.caption_image_transformer(model, image, tokenizer)
    # await context.bot.send_document(chat_id=chat_id, document='tests/test.png')
    await context.bot.send_message(chat_id=update.effective_chat.id, text=caption_text)
    # await context.bot.send_photo(chat_id=update.effective_chat.id, photo=file_id)


if __name__ == '__main__':
    with open("token.txt") as tg_token_file:
        tg_token = tg_token_file.read()
    application = ApplicationBuilder().token(tg_token).build()

    start_handler = CommandHandler('start', start)
    caption_handler = MessageHandler(filters.PHOTO & (~filters.COMMAND), caption)
    application.add_handler(caption_handler)
    application.add_handler(start_handler)
    application.run_polling()

