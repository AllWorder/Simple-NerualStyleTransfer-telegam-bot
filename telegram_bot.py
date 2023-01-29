import torch
import torchvision.models as models

from cmath import nan
from urllib import response
import telebot
import re
from logging import exception
import time

from utils import image_loader, save_image
from StyleTransferNN import run_style_transfer

'''
This file decribes bot's behavior

To run bot print: python3 telegram_bot.py
Don't forget to fill your bot-token in TELEGRAM_TOKEN
'''

# Global constants
IMAGE_SIZE = 512
ITERATION_LIMIT = 300
STYLE_WEIGHT = 10000000
CONTENT_WEIGHT = 1

TELEGRAM_TOKEN = < place your token here >



# Trying to use gpu, if it's possible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using {device}')

# Downloading model
cnn = models.vgg19(pretrained=True).features.to(device).eval()
cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

style_img = []
content_img = []



# Creating bot and it's interface
bot = telebot.TeleBot(TELEGRAM_TOKEN)
bot.enable_save_next_step_handlers(delay=2)

@bot.message_handler(commands=['start'])
def send_welcome(message):
    msg = bot.reply_to(message, 'send content photo')
    bot.register_next_step_handler(msg, get_content_image)

def get_content_image(message):
    try:
        fileID = message.photo[-1].file_id
        file_info = bot.get_file(fileID)
        downloaded_file = bot.download_file(file_info.file_path)

        with open("content.png", 'wb') as new_file:
            new_file.write(downloaded_file)

        msg = bot.reply_to(message, 'Good, now send style photo')
        bot.register_next_step_handler(msg, get_style_image_and_run_transfer)
    except Exception as e:
        bot.reply_to(message, 'something went wrong, try again')

def get_style_image_and_run_transfer(message):
    try:
        fileID = message.photo[-1].file_id
        file_info = bot.get_file(fileID)
        downloaded_file = bot.download_file(file_info.file_path)

        with open("style.png", 'wb') as new_file:
            new_file.write(downloaded_file)

        msg = bot.reply_to(message, 'Good, wait for transfering. It can take several minutes')

        style_image = image_loader("style.png", image_size=IMAGE_SIZE, device=device)
        content_image = image_loader("content.png", image_size=IMAGE_SIZE, device=device)
        input_image = content_image.clone()
        output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                               content_image, style_image, input_image, num_steps=ITERATION_LIMIT, device=device, style_weight=STYLE_WEIGHT, content_weight=CONTENT_WEIGHT,)
        save_image(output, 'result')
        photo = open('result.png', 'rb')
        bot.send_photo(message.chat.id, photo, caption = 'Your photo with new style')

    except Exception as e:
        bot.reply_to(message, 'something went wrong, try again')
        

   
bot.infinity_polling()
