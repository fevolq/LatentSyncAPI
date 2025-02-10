#!-*- coding:utf-8 -*-
# FileName:
import asyncio
import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

import requests
import uvicorn
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import RedirectResponse
from omegaconf import OmegaConf
from pydantic import BaseModel, Field
from starlette.responses import FileResponse, JSONResponse

from scripts import inference

WORKERS = max(int(os.getenv('WORKERS', 1)), 1)
INPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), './data/input'))
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), './data/output'))
print(
    f'【当前配置】\n'
    f'并发：{WORKERS}\n'
    f'INPUT_DIR: {INPUT_DIR}\n'
    f'OUTPUT_DIR: {OUTPUT_DIR}'
)


@asynccontextmanager
async def lifespan(_: FastAPI):
    print('------------------------Start------------------------')
    os.makedirs(f'{INPUT_DIR}/video', exist_ok=True)
    os.makedirs(f'{INPUT_DIR}/audio', exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    yield
    print('------------------------End------------------------')


app = FastAPI(lifespan=lifespan)
executor = ThreadPoolExecutor(max_workers=WORKERS)


@app.get("/")
async def root():
    return RedirectResponse(url='/docs')


@app.get("/healthy")
async def healthy():
    return {'code': 200}


@app.exception_handler(RuntimeError)
async def handle_cuda_error(_, exc):
    raise exc


@app.exception_handler(AssertionError)
async def assert_handler(_, exc: AssertionError):
    print(f'捕获assert: {str(exc)}')
    return JSONResponse(
        status_code=400,
        content={'code': 400, 'msg': str(exc)}
    )


@app.exception_handler(Exception)
async def exception_handler(_, exc: Exception):
    print(f'捕获异常：{str(exc)}')
    if isinstance(exc, SystemExit):
        raise exc
    return JSONResponse(
        status_code=500,
        content={'code': 500, 'msg': '服务器异常', 'content': str(exc)}
    )


def download_file(url, name):
    assert url.startswith('http') or url.startswith('https'), 'error file, need one url'
    print(f'Start download file[{name}]: {url}')
    response = requests.get(url, stream=True)
    response.raise_for_status()  # 检查请求是否成功

    path = os.path.join(INPUT_DIR, name)
    with open(path, 'wb') as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)
    print(f'File[{name}] downloaded.')
    return path


@app.post("/upload")
def upload(file: UploadFile = File(...)):
    suffix = Path(file.filename).suffix

    file_name = f'{str(uuid.uuid4())}{suffix}'
    file_type = 'video' if suffix.lower() in ['.mp4', '.avi', '.mov'] else 'audio'

    with open(os.path.join(INPUT_DIR, f'{file_type}/{file_name}'), 'wb') as f:
        f.write(file.file.read())
    print(f'{file.filename} 存储成功: {file_type}/{file_name}')

    return {
        'code': 200,
        'data': {
            'file': file_name,
        }
    }


@app.get("/download")
def download(file: str):
    print(f'Download {file}')

    file_path = os.path.join(OUTPUT_DIR, file)
    assert os.path.exists(file_path), f'{file} not exists'

    return FileResponse(
        path=file_path,
        media_type="application/octet-stream",
        filename=file,
    )


class Inference(BaseModel):
    video: str = Field(description='视频')
    audio: str = Field(description='音频')
    steps: int = Field(default=20, description='迭代步数')
    scale: float = Field(default=1.5)
    seed: int = Field(default=1247, description='随机种子')


def prepare(req: Inference) -> tuple:
    video = req.video
    audio = req.audio
    if video.startswith('http'):
        video_name = f'{str(uuid.uuid4())}.mp4'
        download_file(video, f'video/{video_name}')
        video = video_name
    if audio.startswith('http'):
        audio_name = f'{str(uuid.uuid4())}.mp3'
        download_file(audio, f'audio/{audio}')
        audio = audio_name
    assert os.path.exists(os.path.join(INPUT_DIR, f'video/{video}')), 'video not exists'
    assert os.path.exists(os.path.join(INPUT_DIR, f'audio/{audio}')), 'audio not exists'
    output_video = f'{str(uuid.uuid4())}.mp4'

    options = {
        'video_path': os.path.join(INPUT_DIR, f'video/{video}'),
        'audio_path': os.path.join(INPUT_DIR, f'audio/{audio}'),
        'video_out_path': os.path.join(OUTPUT_DIR, output_video),

        'inference_ckpt_path': 'checkpoints/latentsync_unet.pt',
        'seed': req.seed,
        'inference_steps': req.steps,
        'guidance_scale': req.scale,
    }
    return options, output_video


@app.post("/inference", description='推理。阻塞等待模式')
async def inference_(req: Inference):
    print(f'接收请求：{req.video}, {req.audio}')
    options, output_name = prepare(req)
    config = OmegaConf.load(Path("configs/unet/second_stage.yaml"))

    loop = asyncio.get_event_loop()
    await loop.run_in_executor(executor, inference.main, config, SimpleNamespace(**options))

    return {
        'code': 200,
        'data': {
            'video': output_name,
        }
    }


@app.post("/submit", description='推理。队列模式')
async def submit(req: Inference):
    options, output_name = prepare(req)
    # TODO: 队列
    return {
        'code': 200,
        'data': {
            'task_id': '',
            'state': ''
        }
    }


if __name__ == '__main__':
    uvicorn.run(
        app,
        host='0.0.0.0',
        port=7860,
    )
