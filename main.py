import os
import gradio as gr
import shutil
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from agent.agent import Agent
from models.MLLMModel import Qwen2VLModel_AWQ
from video_processing.video_processor import VideoProcessor

import utils.logging_setup
from utils.project_paths import VIDEO_STORE_DIR

# 初始化 mllm
mllm = Qwen2VLModel_AWQ()
# 初始化视频处理器
video_processor = VideoProcessor(mllm=mllm)
# 初始化代理
agent = Agent(mllm=mllm, video_processor=video_processor)

# 全局变量
chat_video_info = None
chat_history = []  # 保存聊天记录的全局变量
all_videos = []  # 保存所有视频信息的全局变量

# 视频上传功能
def upload_video(video):
    # 如果video是None
    if video is None:
        return None, "请上传一个视频"
    # 视频路径 
    video_path = os.path.join(VIDEO_STORE_DIR, os.path.basename(video.name))
    shutil.copyfile(video.name, video_path)
    # 处理视频
    video_id = video_processor.process_video(video_path)
    # 生成视频摘要
    agent.progressive_summary(video_id)
    update_all_videos()
    return video_id, f"视频{video.name}已上传与处理，ID为{video_id}"

# 获取数据库中的所有视频 ID
def update_all_videos():
    global all_videos
    videos_metadata = video_processor.get_all_video_metadata()
    all_videos = []
    for data in videos_metadata:
        all_videos.append(f"{data['name']}@{data['video_id']}")


def get_all_videos():
    return all_videos   

# 选择视频功能
def select_video(video_info):
    video_id = video_info.split('@')[-1]
    video_metadata = video_processor.search_video_metadata(video_id)
    video_path = video_metadata['path']
    summary = agent.progressive_summary(video_id)
    global chat_video_info
    chat_video_info = video_info
    return video_path, summary

# 删除视频功能
def delete_video(video_info):
    video_id = video_info.split('@')[-1]
    video_processor.delete_by_video_id(video_id)
    update_all_videos()
    updated_all_videos = get_all_videos()
    return gr.update(choices=updated_all_videos, value=None), \
    gr.update(choices=updated_all_videos, value=None) 

# 清空所有视频功能
def clear_videos():
    video_processor.drop_all()
    update_all_videos()
    updated_all_videos = get_all_videos()
    return gr.update(choices=updated_all_videos, value=None), \
    gr.update(choices=updated_all_videos, value=None) 

# 对话聊天功能：使用 Agent 生成对话回复
def chat(message):
    global chat_video_info, chat_history
    if chat_video_info:
        video_id = chat_video_info.split('@')[-1]
        response = agent.video_chat(video_id, message)
        chat_history.append((message, response))  # 将新的对话添加到聊天记录中
        return chat_history
    else:
        # 如果没有选择视频，返回一条系统消息
        return chat_history + [(message, "请先选择一个视频")]

# 清空聊天记录功能
def clear_chat():
    global chat_history
    chat_history = []  # 清空聊天记录
    agent.clear_history()
    return chat_history

update_all_videos()

# 创建Gradio界面
with gr.Blocks() as demo:
    # 左上部分：视频上传、选择、删除
    with gr.Row():
        with gr.Column():
            video_output = gr.Video(label="当前视频")
            with gr.Row():
                video_select = gr.Dropdown(choices=get_all_videos(), label="选择视频")
                select_btn = gr.Button("选择视频", scale=0.5)
            video_summary = gr.Textbox(label="当前视频总结")

            with gr.Row():
                video_upload = gr.File(label="上传视频")
                upload_btn = gr.Button("上传视频", scale=0.5)

        # 右边部分：聊天记录和聊天框
        with gr.Column():
            chat_box = gr.Chatbot(label="聊天记录")
            message_input = gr.Textbox(label="输入消息")
            with gr.Row():
                send_message_btn = gr.Button("发送消息")
                clear_chat_btn = gr.Button("清空聊天记录")
            with gr.Row():
                video_delete = gr.Dropdown(choices=get_all_videos(), label="删除视频")
                delete_btn = gr.Button("删除视频", scale=0.5)
            clear_videos_btn = gr.Button("清空所有视频")
    
    # 绑定功能
    def start_processing(video):
        # 开始上传视频
        upload_result = upload_video(video)
        update_all_videos()
        updated_all_videos = get_all_videos()

        # 启用聊天按钮
        return (
            gr.update(interactive=True),  # 启用发送消息按钮
            "",  # 清空消息输入框
            gr.update(choices=updated_all_videos, value=None),
            gr.update(choices=updated_all_videos, value=None)
        )
    
    # 视频上传时禁用聊天框，处理完成后重新启用
    upload_btn.click(
        start_processing,
        inputs=video_upload,
        outputs=[send_message_btn, message_input, video_select, video_delete],
        show_progress=True
    )

    # 选择视频、删除视频、清空视频
    select_btn.click(select_video, inputs=video_select, outputs=[video_output, video_summary])
    delete_btn.click(delete_video, inputs=video_delete, outputs=[video_select, video_delete])
    clear_videos_btn.click(clear_videos, outputs=[video_select, video_delete])
    
    # LLM Agent对话绑定
    send_message_btn.click(chat, inputs=message_input, outputs=[chat_box])
    clear_chat_btn.click(clear_chat, outputs=[chat_box])

# 启动Gradio界面
demo.launch()
