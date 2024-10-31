#!/usr/bin/env python

# from urllib import response
from pydoc import describe
import rospy
import subprocess
import sys
import os
from std_msgs.msg import String
from navigation.srv import NavigationLocation
from grounding.srv import CommandTasks
from detection.srv import ObjectExist
from detection.srv import ManipObject
from detection.srv import GetImage
from vqa.srv import GetDescription
# from llm.srv import ChatLLM
from planning.srv import New

# BOT_TOKEN = "6472938704:AAEb8YcStTHq3fUmeR6VvJMb8-XJtfnQCKc"
import string
import requests
# import telegram
# from telegram.ext import Updater, CommandHandler, MessageHandler, Filters



init = [
        ("Robot", 'robot'),
        ('Location', 'kitchen'),
        ('Location', 'bathroom'),
        ('Location', 'bedroom'),
        ('Obj', 'apple'),
        ('Obj', 'banana'),
        ('Obj', 'water'),
        ('Obj', 'can'),
        ("HandEmpty", 'robot'),
        ("CanMove", 'robot'),
        ("At", 'robot', 'bathroom'),
        ("At", 'can', 'kitchen'),
        # ('At', 'water', 'kitchen'),
        # ('At', 'apple', 'bathroom'),
        # ('At', 'banana', 'bedroom'),
    ]

goal = [
    # ('At', 'robot', 'bathroom'),
    # ('At', 'apple', 'kitchen'), 
    # ('At', 'banana', 'bathroom'), 
    # ('At', 'water', 'bedroom'), 
]


def save_file(path, txt):
    with open(path, 'w') as file:
        file.write(txt)


def set_goal(list_sentence):
    goal = []
    print(list_sentence)
    for sentence in list_sentence:
        # sentence = sentence.split()
        if sentence[0] == 'go':
            print(sentence[2])
            goal.append(('At', 'robot', sentence[2]))
        elif sentence[0] == 'get':
            goal.append(('At', sentence[1], sentence[2]))
    print(goal)
    return goal


def send_init(path, list_sentence):
    goal = set_goal(list_sentence)
    init_string = str(init) + '\n'
    goal_string = str(goal)
    print(init_string, goal_string)
    save_file(path, init_string + goal_string)


def run_script(script_path, txt_path):
    
    conda_environment_name = 'project_env'
    pddlstream_home = '/home/mustar/pddlstream'

    try:
        subprocess.check_call(['conda', 'run', '-n', conda_environment_name, 'python', '-m', script_path], cwd=pddlstream_home)
    except subprocess.CalledProcessError as e:
        print('error while calling planning node')


    with open(txt_path, 'r') as file:
        lines = file.readlines()
        txt = ' '.join(lines).replace('\n', '')

    # print(txt)

    actions = eval(txt)

    return actions

def nav_service(location):
    rospy.loginfo('waiting for navigation service...')
    rospy.wait_for_service('send_location')

    send = rospy.ServiceProxy('send_location', NavigationLocation)


    response = send(location)
    rospy.loginfo('response returned')
    return response

def print_solve(plan):
    totalString = ''
    cnt = 1
    action_list = []
    for action in plan:
        st = ''
        # name, first, second = action.name, action.args[1], action.args[2]
        name, first, second = action[0], action[1], action[2]
        action_list.append([name, first, second])
        if name == 'pick':
            st = name + ' ' + first + ' from ' + second
        elif name == 'navigate':
            st = name + ' from ' + first + ' to ' + second
        elif name == 'place':
            st = name + ' ' + first + ' in ' + second
        totalString += str(cnt) + ' ' + st + '\n'
        cnt += 1
    print(totalString)
    # return totalString, action_list


def split_descritpion(string, word):
    index = string.find(word)      
    part1 = string[:index].strip()  
    part2 = string[index + len(word):].strip() 
    return part1, part2
    

def grounding_service(command):
    rospy.loginfo('waiting for grounding service...')
    rospy.wait_for_service('command_tasks')

    send = rospy.ServiceProxy('command_tasks', CommandTasks)

    tasks = send(command)
    rospy.loginfo('tasks received.')

    tasks, description = split_descritpion(tasks.tasks, 'Description')
    print(tasks)

    #split the received string of tasks
    sentences = tasks.split('.')
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
    list_tasks = [string.split() for string in sentences]

    print(list_tasks, description)

    return list_tasks, description

def check_service(object_description):
    rospy.loginfo('waiting for detection service...')
    rospy.wait_for_service('detect_object')

    send = rospy.ServiceProxy('detect_object', ObjectExist)

    exists = send(object_description)

    return exists

def manip_service(action):
    rospy.loginfo('waiting for manipulation service...')
    rospy.wait_for_service('manipulate_object')

    send = rospy.ServiceProxy('manipulate_object', ManipObject)

    success = send(action)

    return success

def describe_service(question):
    rospy.loginfo('waiting for ViLT service...')
    rospy.wait_for_service('get_describtion')

    send = rospy.ServiceProxy('get_describtion', GetDescription)

    response = send(question)

    return response

# def llm_service(question):
#     rospy.loginfo('waiting for LLM service...')
#     rospy.wait_for_service('chat_llm')

#     send = rospy.ServiceProxy('chat_llm', ChatLLM)

#     response = send(question)

#     return response

def filter_questions(list_sentence, sentence):
    describe = 0
    for idx, action in enumerate(list_sentence) :
        # if asking a question redirect to LLM
        if action[0] == 'question':
            list_sentence.pop(idx)
            rospy.loginfo("LLM is responding...")   # LLM               ************
            # rospy.loginfo(llm_service(sentence))
        elif action[0] == 'describe':
            list_sentence.pop(idx)
            describe = 1
    return list_sentence, describe

def map_actions(actions, sentence, description):
    prev_pick = 0
    for action in actions:

        act = action[0]
        object = action[1]
        location = action[2]

        if act == 'navigate':                       # Navigation        Tested + Live
            nav_service(location)
        elif act == 'pick':
            if check_service(description):          # DINO              Tested ******
                prev_pick = 1
                # print('object picked')
                prev_pick = manip_service(act)      # Pick + DOPE       Tested ******
                if not prev_pick:
                    rospy.loginfo('picking failed....')
                else:
                    rospy.loginfo('{} not detected'.format(object))
            else: 
                rospy.loginfo('no object detected')
        elif act == 'place' and prev_pick:
            rospy.loginfo('place object')
            place = manip_service(act)  
            prev_pick = 0                           # Place + DOPE      ************
            if not place:
                rospy.loginfo('placing failed....')
        elif act == 'describe':
            describe_service(sentence)              # ViLT              Tested ******


class MySubscriber:
    def __init__(self):
        self.received_message = None
        # rospy.init_node('string_subscriber', anonymous=True)
        rospy.Subscriber('/llm_topic', String, self.callback)

    def callback(self, data):
        # rospy.loginfo("Received message: %s", data.data)
        self.received_message = data.data

    def get_last_message(self):
        return self.received_message


def get_sentence():
    subscriber = MySubscriber()

    # Wait for the first message to be received
    while subscriber.get_last_message() is None and not rospy.is_shutdown():
        rospy.sleep(0.1)

    # Once a message is received, you can access it using the get_last_message method
    sentence = subscriber.get_last_message()
    # print("Last message received:", last_message)
    return sentence


if __name__ == '__main__':

    rospy.init_node('main_client')

    init_txt_path = '/home/mustar/test_ws/src/planning/src/first_attempt/init.txt'
    planning_script_path = 'examples.first_attempt.planning_node'
    actions_txt_path = '/home/mustar/test_ws/src/planning/src/first_attempt/actions.txt'


    # sentence = get_sentence()             
                   
    # rospy.loginfo('Recieved Sentence: {}'.format(sentence))
    

    # sentence = ' '.join(sys.argv[1:])
    sentence = 'get red tomato can to bedroom'
    list_sentence = [['get', 'can', 'bedroom']]
    description = 'red tomato can'
    # actions = [['navigate', 'can', 'bathroom']]

    rospy.loginfo("waiting for grounding service")
    # list_sentence, description = grounding_service(sentence)
    rospy.loginfo("Sentence Received: {}".format(sentence))
    rospy.loginfo('Language Grounding Output: {}'.format(list_sentence))

    # # list_sentence[0][0] = 'pick'

    # print(list_sentence)

    # # list_sentence, desc = filter_questions(list_sentence, sentence)
    # # print(list_sentence)

    # # print("error in get instead of pick")

    send_init(init_txt_path, list_sentence)
    actions = run_script(planning_script_path, actions_txt_path)
    
    rospy.loginfo('Planning Output:')
    print_solve(actions)

    # # # if desc:
    # # #     actions.append(['describe', '', ''])


    rospy.loginfo('Start executing actions...')
    map_actions(actions, sentence, description)

    rospy.loginfo('All actions completed.')


    rospy.spin()







