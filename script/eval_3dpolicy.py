import sys
import os
import subprocess

sys.path.append("./")
sys.path.append(f"./policy")
sys.path.append("./description/utils")
from envs import CONFIGS_PATH
from envs.utils.create_actor import UnStableError

import numpy as np
from pathlib import Path
from collections import deque
import traceback

import yaml
from datetime import datetime
import importlib
import argparse
import pdb

from generate_episode_instructions import *

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)

class Env:
    def __init__(self):
        pass
    @staticmethod
    def class_decorator(task_name):
        envs_module = importlib.import_module(f"envs.{task_name}")
        try:
            env_class = getattr(envs_module, task_name)
            env_instance = env_class()
        except:
            raise SystemExit("No Task")
        return env_instance
    @staticmethod
    def get_camera_config(camera_type):
        camera_config_path = os.path.join(parent_directory, "../task_config/_camera_config.yml")

        assert os.path.isfile(camera_config_path), "task config file is missing"

        with open(camera_config_path, "r", encoding="utf-8") as f:
            args = yaml.load(f.read(), Loader=yaml.FullLoader)

        assert camera_type in args, f"camera {camera_type} is not defined"
        return args[camera_type]
    @staticmethod
    def get_embodiment_config(robot_file):
        robot_config_file = os.path.join(robot_file, "config.yml")
        with open(robot_config_file, "r", encoding="utf-8") as f:
            embodiment_args = yaml.load(f.read(), Loader=yaml.FullLoader)
        return embodiment_args
    @staticmethod
    def get_embodiment_file(embodiment_types,embodiment_type):
        robot_file = embodiment_types[embodiment_type]["file_path"]
        if robot_file is None:
            raise "No embodiment files"
        return robot_file
    def dual_arm(self):
        return self.task.get_dual_arm()
    def Create_env(self,task_name,head_camera_type,seed,task_num,instruction_type,task_config):
        with open(f'./task_config/{task_config}.yml', 'r', encoding='utf-8') as f:
            self.args = yaml.load(f.read(), Loader=yaml.FullLoader)
        self.args['task_name'] = task_name
        self.args["task_config"] = task_config
        self.args["ckpt_setting"] = None
        self.embodiment_type = self.args.get("embodiment")
        embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")
        with open(embodiment_config_path, "r", encoding="utf-8") as f:
            self._embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)
        with open(CONFIGS_PATH + "_camera_config.yml", "r", encoding="utf-8") as f:
            self._camera_config = yaml.load(f.read(), Loader=yaml.FullLoader)
        head_camera_type = self.args["camera"]["head_camera_type"]
        self.args["head_camera_h"] = self._camera_config[head_camera_type]["h"]
        self.args["head_camera_w"] = self._camera_config[head_camera_type]["w"]
        if len(self.embodiment_type) == 1:
            self.args["left_robot_file"] = self.get_embodiment_file(self._embodiment_types, self.embodiment_type[0])
            self.args["right_robot_file"] = self.get_embodiment_file(self._embodiment_types, self.embodiment_type[0])
            self.args["dual_arm_embodied"] = True
        elif len(self.embodiment_type) == 3:
            self.args["left_robot_file"] = self.get_embodiment_file(self._embodiment_types, self.embodiment_type[0])
            self.args["right_robot_file"] = self.get_embodiment_file(self._embodiment_types, self.embodiment_type[1])
            self.args["embodiment_dis"] = self.embodiment_type[2]
            self.args["dual_arm_embodied"] = False
        else:
            raise ValueError("embodiment items should be 1 or 3")
        if len(self.embodiment_type) == 1:
            self.embodiment_name = str(self.embodiment_type[0])
        else:
            self.embodiment_name = str(self.embodiment_type[0]) + "+" + str(self.embodiment_type[1])
        self.args["left_embodiment_config"] = Env.get_embodiment_config(self.args["left_robot_file"])
        self.args["right_embodiment_config"] = Env.get_embodiment_config(self.args["right_robot_file"])
        print("============= Config =============\n")
        print("\033[95mMessy Table:\033[0m " + str(self.args["domain_randomization"]["cluttered_table"]))
        print("\033[95mRandom Background:\033[0m " + str(self.args["domain_randomization"]["random_background"]))
        if self.args["domain_randomization"]["random_background"]:
            print(" - Clean Background Rate: " + str(self.args["domain_randomization"]["clean_background_rate"]))
        print("\033[95mRandom Light:\033[0m " + str(self.args["domain_randomization"]["random_light"]))
        if self.args["domain_randomization"]["random_light"]:
            print(" - Crazy Random Light Rate: " + str(self.args["domain_randomization"]["crazy_random_light_rate"]))
        print("\033[95mRandom Table Height:\033[0m " + str(self.args["domain_randomization"]["random_table_height"]))
        print("\033[95mRandom Head Camera Distance:\033[0m " + str(self.args["domain_randomization"]["random_head_camera_dis"]))

        print("\033[94mHead Camera Config:\033[0m " + str(self.args["camera"]["head_camera_type"]) + f", " +
            str(self.args["camera"]["collect_head_camera"]))
        print("\033[94mWrist Camera Config:\033[0m " + str(self.args["camera"]["wrist_camera_type"]) + f", " +
            str(self.args["camera"]["collect_wrist_camera"]))
        print("\033[94mEmbodiment Config:\033[0m " + self.embodiment_name)
        print("\n==================================")
        self.task=self.class_decorator(self.args["task_name"])
        self.st_seed = 100000 * (1 + seed)
        self.task_num = task_num
        self.clear_cache_freq=self.args['clear_cache_freq']
        self.args["eval_mode"] = True
        self.instruction_type = instruction_type
        return self.Check_seed(self.task_num,self.clear_cache_freq)




        
    def Init_task_env(self,seed,id,episode_info_list,test_num):
        self.env_state=0 #0:running 1:success 2:fail
        self.step=0
        self.succ_seed=seed
        self.task.setup_demo(now_ep_num=id, seed = seed, is_test = True, ** self.args)
        results = generate_episode_descriptions(self.args["task_name"], episode_info_list, test_num)
        print(results)
        instruction = np.random.choice(results[0][self.instruction_type])
        self.task.set_instruction(instruction=instruction)
        return instruction
    def Check_seed(self,test_num, clear_cache_freq):
        expert_check=True
        print("Task name: ", self.args["task_name"])
        suc_seed_list=[]
        now_id_list = []
        succ_tnt=0
        now_seed=self.st_seed
        now_id = 0
        self.task.cus=0
        self.task.test_num = 0
        episode_info_list_total=[]
        while succ_tnt<test_num:
            render_freq = self.args['render_freq']
            self.args['render_freq'] = 0
            if expert_check:
                try:
                    self.task.setup_demo(now_ep_num=now_id, seed = now_seed, is_test = True, ** self.args)
                    episode_info=self.task.play_once()
                    self.task.close()
                    suc_seed_list.append(now_seed)
                    now_id_list.append(now_id)
                    now_id += 1
                    succ_tnt += 1
                    now_seed += 1
                    episode_info_list = [episode_info["info"]]
                    episode_info_list_total.append(episode_info_list)
                except Exception as e:
                    stack_trace = traceback.format_exc()
                    print(' -------------')
                    print('Error: ', stack_trace)
                    print(' -------------')
                    self.task.close()
                    now_seed += 1
                    self.args['render_freq'] = render_freq
                    print('error occurs !')
                    continue
            self.args['render_freq'] = render_freq
        return suc_seed_list, now_id_list, episode_info_list_total
    def Detect_env_state(self):
        if self.step>self.task.step_lim:
            self.env_state=2
        if self.task.eval_success:
            self.env_state=1
    def Take_action(self,actions):
        # actions=[]
        # actions.append(action)
        # actions=np.array(actions)
        # self.task.apply_action(actions)
        # actions=action
        for action in actions:
            self.task.take_action(action)
        self.step+=actions.shape[0]
        self.Detect_env_state()
        if self.env_state==1:
            print('Task Success!')
            self.Close_env()
            return "success"
        elif self.env_state==2:
            print('Task Failed!')
            self.Close_env()
            return "fail"
        else:
            return "run"
    def Close_env(self):
        self.task.close_env(clear_cache=((self.succ_seed + 1) % self.clear_cache_freq == 0))
        if self.task.render_freq:
            self.task.viewer.close()
        print ('Env Closed!')
        self.task._take_picture()
    def get_observation(self):
        return self.task.get_obs()