#!/usr/bin/env python
BOT_TOKEN = "" #bot token
import string
from std_msgs.msg import String
# import torch
# import transformers
# from transformers import BitsAndBytesConfig
import requests
# from IPython.display import Audio
# from gazelle.gazelle import (
#     GazelleConfig,
#     GazelleForConditionalGeneration,
#     GazelleProcessor,
# )
import telegram
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters

import rospy
# from llm.srv import ChatLLM, ChatLLMResponse
from planning.srv import New, NewResponse
from vqa.srv import GetDescription
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# def inference_collator(audio_input, prompt="Transcribe the following \n<|audio|>", audio_dtype=torch.float16):
#     # audio_values = audio_processor(
#     #     audio=audio_input, return_tensors="pt", sampling_rate=16000
#     # ).input_values
#     msgs = [
#         {"role": "user", "content": prompt},
#     ]
#     labels = tokenizer.apply_chat_template(
#         msgs, return_tensors="pt", add_generation_prompt=True
#     )
#     return {
#         # "audio_values": audio_values.squeeze(0).to("cuda").to(audio_dtype),
#         "input_ids": labels.to("cuda"),
#     }

def extract_text(text):
  split_text = text.split("[/INST]")
#   split_text = text.split("</s>")
  if len(split_text) > 1:
    return split_text[1].strip()  # Remove leading/trailing whitespace
  else:
    return None

def get_url():
    contents = requests.get('https://unsplash.com/photos/woman-with-dslr-camera-e616t35Vbeg')    
    url = contents['url']
    return url
def send_message(update, context):
    chat_id = update.effective_chat.id
    text = update.message.text

    # Handle commands starting with '/send'
    # if text.startswith('/send'):
        # Extract the message content from the command
    # service = rospy.Service('llm_server', New, chat)

    message_content = text[len('/send '):]
    rospy.loginfo('send message {}'.format(message_content))

    pub = rospy.Publisher('/llm_topic', String, queue_size=10)
    rate = rospy.Rate(1)
    pub.publish(message_content)
    rate.sleep()

    # inputs = inference_collator(0, message_content)
    # output = tokenizer.decode(model.generate(**inputs, max_new_tokens=300)[0])
    # output = extract_text(output)
    # context.bot.send_message(chat_id=chat_id, text=output[:-4])
    # else:
    #     # Send the received message back to the user
    #     context.bot.send_message(chat_id=chat_id, text=text)

    # return message_content


def describe_service(question):
    rospy.loginfo('waiting for ViLT service...')
    rospy.wait_for_service('description_service')

    send = rospy.ServiceProxy('description_service', GetDescription)

    response = send(question)

    return response

def desc_message(update, context):
    chat_id = update.effective_chat.id
    text = update.message.text

    # Handle commands starting with '/send'
    # if text.startswith('/send'):
        # Extract the message content from the command
    # service = rospy.Service('llm_server', New, chat)
    message_content = text[len('/desc '):]
    rospy.loginfo('describe message {}'.format(message_content))
    message_response = describe_service(message_content)
    rospy.loginfo('describe response {}'.format(message_response))
    # pub = rospy.Publisher('/vilt_topic', String, queue_size=10)
    # rate = rospy.Rate(1)
    # pub.publish(message_content)
    # rate.sleep()

    # # inputs = inference_collator(0, message_content)
    # # output = tokenizer.decode(model.generate(**inputs, max_new_tokens=300)[0])
    # # output = extract_text(output)
    # # context.bot.send_message(chat_id=chat_id, text=output[:-4])
    # # else:
    # #     # Send the received message back to the user
    # #     context.bot.send_message(chat_id=chat_id, text=text)

    # # return message_content
    # return MessageHandler.END


def echo(update, context):
    update.message.reply_text(update.message.text)

def main():

    # model_dir = "/home/mustar/test_ws/src/llm/src/"
    # config = GazelleConfig.from_pretrained("/home/mustar/test_ws/src/llm/src/")
    # tokenizer = transformers.AutoTokenizer.from_pretrained("/home/mustar/test_ws/src/llm/src/")
    # quantization_config_4bit_dq = BitsAndBytesConfig(
    # load_in_4bit=True,
    # bnb_4bit_compute_dtype=torch.bfloat16,
    # bnb_4bit_use_double_quant=True,
    # )
    # model = GazelleForConditionalGeneration.from_pretrained(
    # model_dir,
    # device_map="cuda:0",
    # quantization_config=quantization_config_4bit_dq,
    # )

    updater = Updater(token=BOT_TOKEN, use_context=True)
    dispatcher = updater.dispatcher

    # Handle messages with '/send' command or any other text
    dispatcher.add_handler(CommandHandler('send', send_message))

    dispatcher.add_handler(CommandHandler('desc', desc_message))

    dispatcher.add_handler(MessageHandler(Filters.text & ~Filters.command, echo))

    # Start the Updater to receive updates
    updater.start_polling()

    # Idle the Updater to prevent it from stopping
    updater.idle()
    # print(send_message())
# -------------------------------------------------------------------------------------------------------

if __name__=="__main__":

    rospy.init_node('llm_server')
    rospy.loginfo('ChatLLM Server Initiated')

    try:
        main()

    except rospy.ROSInterruptException:
        pass