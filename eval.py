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

class EnvGrader:
    def __init__(self, usr_args_path, policy_name):
        # 从usr_args获取基本参数
        with open(usr_args_path, "r", encoding="utf-8") as f:
            usr_args = yaml.load(f.read(), Loader=yaml.FullLoader)
        self.task_name = usr_args["task_name"]
        self.task_config = usr_args["task_config"]

        # 加载任务配置
        with open(f"./task_config/{self.task_config}.yml", "r", encoding="utf-8") as f:
            self.args = yaml.load(f.read(), Loader=yaml.FullLoader)
        
        # 添加基本参数到args
        self.args['task_name'] = self.task_name
        self.args["task_config"] = self.task_config
        
        # 设置机器人实体配置
        embodiment_type = self.args.get("embodiment")
        embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")
        
        with open(embodiment_config_path, "r", encoding="utf-8") as f:
            self._embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)
            
        # 获取相机配置
        with open(CONFIGS_PATH + "_camera_config.yml", "r", encoding="utf-8") as f:
            self._camera_config = yaml.load(f.read(), Loader=yaml.FullLoader)
        
        # 设置相机参数
        head_camera_type = self.args["camera"]["head_camera_type"]
        self.args["head_camera_h"] = self._camera_config[head_camera_type]["h"]
        self.args["head_camera_w"] = self._camera_config[head_camera_type]["w"]
        self.camera_type = head_camera_type
        
        # 设置机器人文件
        if len(embodiment_type) == 1:
            self.args["left_robot_file"] = self._get_embodiment_file(embodiment_type[0])
            self.args["right_robot_file"] = self._get_embodiment_file(embodiment_type[0])
            self.args["dual_arm_embodied"] = True
            self.robot_file = self.args["left_robot_file"]
            self.embodiment_name = str(embodiment_type[0])
        elif len(embodiment_type) == 3:
            self.args["left_robot_file"] = self._get_embodiment_file(embodiment_type[0])
            self.args["right_robot_file"] = self._get_embodiment_file(embodiment_type[1])
            self.args["embodiment_dis"] = embodiment_type[2]
            self.args["dual_arm_embodied"] = False
            self.robot_file = self.args["left_robot_file"]
            self.embodiment_name = str(embodiment_type[0]) + "+" + str(embodiment_type[1])
        else:
            raise ValueError("embodiment items should be 1 or 3")
        
        # 加载左右手臂配置
        self.args["left_embodiment_config"] = self._get_embodiment_config(self.args["left_robot_file"])
        self.args["right_embodiment_config"] = self._get_embodiment_config(self.args["right_robot_file"])
        self.embodiment_args = self.args["left_embodiment_config"]
        
        # 创建环境
        self.env = self._class_decorator(self.task_name)        
        
        # 初始化环境
        seed = usr_args.get("seed", 0)
        self.st_seed = 100000 * (1 + seed)
        
        # 设置结果保存目录
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.save_dir = Path(f"eval_result/{self.task_name}/{policy_name}/{self.task_config}/{current_time}")
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.video_size = None
        
        # 设置视频记录
        if self.args.get("eval_video_log", False):
            self.video_save_dir = self.save_dir
            camera_config = self._get_camera_config(self.args["camera"]["head_camera_type"])
            self.video_size = str(camera_config["w"]) + "x" + str(camera_config["h"])
            self.video_save_dir.mkdir(parents=True, exist_ok=True)
            self.args["eval_video_save_dir"] = self.video_save_dir
        
        # 打印配置信息
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
        # 保存手臂维度
        self.left_arm_dim = len(self.args["left_embodiment_config"]["arm_joints_name"][0])
        self.right_arm_dim = len(self.args["right_embodiment_config"]["arm_joints_name"][1]) 
        # 初始化评估状态
        self.expert_check = True
        self.env.suc = 0
        self.env.test_num = 0
        self.now_id = 0
        self.succ_seed = 0
        self.now_seed = self.st_seed
        self.clear_cache_freq = self.args["clear_cache_freq"]
        self.args['eval_mode'] = True
        self.suc_test_seed_list = []
        
    def _class_decorator(self, task_name):
        """
        动态导入并实例化特定任务的环境类
        
        Args:
            task_name (str): 任务名称
            
        Returns:
            环境实例
        """
        envs_module = importlib.import_module(f"envs.{task_name}")
        try:
            env_class = getattr(envs_module, task_name)
            env_instance = env_class()
        except:
            raise SystemExit("No Task")
        return env_instance
    
    def _get_camera_config(self, camera_type):
        """
        读取相机配置参数
        
        Args:
            camera_type (str): 相机类型名称
            
        Returns:
            dict: 相机配置参数
        """
        camera_config_path = os.path.join(parent_directory, "../task_config/_camera_config.yml")
        
        assert os.path.isfile(camera_config_path), "task config file is missing"
        
        with open(camera_config_path, "r", encoding="utf-8") as f:
            args = yaml.load(f.read(), Loader=yaml.FullLoader)
            
        assert camera_type in args, f"camera {camera_type} is not defined"
        return args[camera_type]
    
    def _get_embodiment_config(self, robot_file):
        """
        读取机器人实体配置
        
        Args:
            robot_file (str): 机器人配置文件路径
            
        Returns:
            dict: 机器人配置参数
        """
        robot_config_file = os.path.join(robot_file, "config.yml")
        with open(robot_config_file, "r", encoding="utf-8") as f:
            embodiment_args = yaml.load(f.read(), Loader=yaml.FullLoader)
        return embodiment_args
    
    def _get_embodiment_file(self, embodiment_type):
        """
        获取机器人配置文件路径
        
        Args:
            embodiment_type (str): 机器人类型
            
        Returns:
            str: 配置文件路径
        """
        robot_file = self._embodiment_types[embodiment_type]["file_path"]
        if robot_file is None:
            raise ValueError("No embodiment files")
        return robot_file
    def find_valid_scenes(self, test_num=100):
        """
        专家验证阶段：寻找有效的场景供策略评估使用
        
        Args:
            test_num (int): 需要收集的有效场景数量
            
        Returns:
            list: 有效场景的种子列表
        """
        print("\n开始寻找有效场景...")
        
        while self.succ_seed < test_num:
            render_freq = self.args["render_freq"]
            self.args["render_freq"] = 0

            try:
                self.env.setup_demo(now_ep_num=self.now_id, seed=self.now_seed, is_test=True, **self.args)
                self.episode_info = self.env.play_once()
                self.env.close_env()
            except UnStableError as e:
                print(" -------------")
                print("Error: ", e)
                print(" -------------")
                self.env.close_env()
                self.now_seed += 1
                self.args["render_freq"] = render_freq
                continue
            except Exception as e:
                stack_trace = traceback.format_exc()
                print(" -------------")
                print("Error: ", stack_trace)
                print(" -------------")
                self.env.close_env()
                self.now_seed += 1
                self.args["render_freq"] = render_freq
                print("error occurs !")
                continue

            if self.env.plan_success and self.env.check_success():
                self.succ_seed += 1
                self.suc_test_seed_list.append(self.now_seed)
                print(f"找到有效场景: {self.succ_seed}/{test_num}, 种子: {self.now_seed}")
                self.now_id += 1
                self.args["render_freq"] = render_freq
                self.env.setup_demo(now_ep_num=self.now_id-1, seed=self.now_seed, is_test=True, **self.args)
                episode_info_list = [self.episode_info["info"]]
                results=generate_episode_descriptions(self.args["task_name"], episode_info_list, test_num)
                self.instruction = np.random.choice(results[0][self.instruction_type])
                self.env.set_instruction(instruction=self.instruction)
                if self.env.eval_video_path is not None:
                    ffmpeg = subprocess.Popen(
                        [
                            "ffmpeg",
                            "-y",
                            "-loglevel",
                            "error",
                            "-f",
                            "rawvideo",
                            "-pixel_format",
                            "rgb24",
                            "-video_size",
                            self.video_size,
                            "-framerate",
                            "10",
                            "-i",
                            "-",
                            "-pix_fmt",
                            "yuv420p",
                            "-vcodec",
                            "libx264",
                            "-crf",
                            "23",
                            f"{self.env.eval_video_path}/episode{self.env.test_num}.mp4",
                        ],
                        stdin=subprocess.PIPE,
                    )
                    self.env._set_eval_video_ffmpeg(ffmpeg)
                self.succ = False
                return
            else:
                print(f"场景验证失败, 种子: {self.now_seed}")
                self.now_seed += 1
                self.args["render_freq"] = render_freq
                continue     
    def eval_policy(self,action):
        Ongoing = True
        if self.env.take_action_cnt<self.env.step_lim:
            self.env.take_action(action)
            self.observation = self.env.get_obs()
            Ongoing= True
        if self.env.eval_success:
            self.succ = True
            Ongoing = False
            self.env.suc += 1
            print(f"成功完成任务: {self.env.suc}/{self.env.test_num}, 当前种子: {self.now_seed}")
            self.env.close_env(clear_cache=((self.succ_seed + 1) % self.clear_cache_freq == 0))
            if self.env.render_freq:
                self.env.viewer.close()
            self.env.test_num += 1
            print(
            f"\033[93m{self.task_name}\033[0m | \033[94m{self.args['policy_name']}\033[0m | \033[92m{self.args['task_config']}\033[0m | \033[91m{self.args['ckpt_setting']}\033[0m\n"
            f"Success rate: \033[96m{self.env.suc}/{self.env.test_num}\033[0m => \033[95m{round(self.env.suc/self.env.test_num*100, 1)}%\033[0m, current seed: \033[90m{self.now_seed-1}\033[0m\n")
            return self.succ,self.observation,Ongoing
        if self.env.take_action_cnt >= self.env.step_lim:
            self.succ = False
            Ongoing = False
            print(f"任务失败: {self.env.suc}/{self.env.test_num}, 当前种子: {self.now_seed}")
            self.env.close_env(clear_cache=((self.succ_seed + 1) % self.clear_cache_freq == 0))
            if self.env.render_freq:
                self.env.viewer.close()
            self.env.test_num += 1
            print(
            f"\033[93m{self.task_name}\033[0m | \033[94m{self.args['policy_name']}\033[0m | \033[92m{self.args['task_config']}\033[0m | \033[91m{self.args['ckpt_setting']}\033[0m\n"
            f"Success rate: \033[96m{self.env.suc}/{self.env.test_num}\033[0m => \033[95m{round(self.env.suc/self.env.test_num*100, 1)}%\033[0m, current seed: \033[90m{self.now_seed-1}\033[0m\n")
            return self.succ,self.observation,Ongoing
        return self.succ,self.observation,Ongoing
    def _save_results(self):
        """保存评估结果到文件"""
        file_path = os.path.join(self.save_dir, f"_result.txt")
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 计算成功率
        success_rates = [self.env.suc / self.env.test_num] if self.env.test_num > 0 else [0.0]
        
        with open(file_path, "w") as file:
            file.write(f"Timestamp: {current_time}\n\n")
            file.write(f"Instruction Type: {self.instruction_type}\n\n")
            # 以相同格式保存成功率
            file.write("\n".join(map(str, success_rates)))
        
        print(f"Data has been saved to {file_path}")
        print(f"Success rate: {self.env.suc}/{self.env.test_num} => {round(self.env.suc/self.env.test_num*100, 1)}%")