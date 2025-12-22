import os
import shutil
import traceback
import uuid
from enum import Enum
from typing import Optional, Dict
import asyncio
from pydantic import BaseModel
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, HTTPException
from fastapi.concurrency import run_in_threadpool
from app.core.engine import SmartReframer
from fastapi.staticfiles import StaticFiles

app = FastAPI()

MODEL_PATH = "app/models/yolox_x.pth"
print("正在初始化 AI 引擎，请稍候...")
# 全局单例引擎
engine = SmartReframer(MODEL_PATH, device="cuda")
print("AI 引擎初始化完毕！")

TEMP_DIR = "/app/temp"
os.makedirs(TEMP_DIR, exist_ok=True)

@app.get("/")
def read_root():
    # 检查模型是否存在
    exists = os.path.exists(MODEL_PATH)
    return {
        "status": "running",
        "model_loaded": exists,
        "model_path": MODEL_PATH if exists else "Not Found"
    }

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/reframe")
async def reframe_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    ratio: str = Form("9:16", description="裁剪比例，如 9:16, 4:3, 1:1"),
    mode: str = Form("normal", description="运镜模式: fast(运动), normal(标准), stable(访谈)"),
    target: str = Form("person", description="追踪主体，如: person, cat, dog"),
    multi_subject: bool = Form(False, description="是否开启多主体拆分模式"),
    split_screen: bool = Form(False, description="是否开启分屏"),
    debug: bool = Form(False),
):
    """
    智能剪辑接口
    - file: 视频文件
    - ratio: 目标比例 (默认 9:16)
    - mode: 运镜模式 (默认 normal)
    """
    # 1. 保存上传的视频
    input_filename = f"input_{file.filename}"
    input_path = os.path.join(TEMP_DIR, input_filename)

    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 2. 定义输出路径
    safe_ratio = ratio.replace(":", "x")
    output_filename = f"reframed_{safe_ratio}_{mode}_{file.filename}"
    output_path = os.path.join(TEMP_DIR, output_filename)

    # 3. 运行处理 (这是一个耗时操作，生产环境建议放入 任务 队列)
    # 这里演示直接调用
    try:
        result_path = engine.process_video(
            input_path,
            output_path,
            ratio_str=ratio,
            mode=mode,
            detect_target=target,
            multi_subject=multi_subject,
            split_screen=split_screen,
            debug=debug
        )
        return {
            "status": "success",
            "message": "Video reframed successfully",
            "output_path": result_path
        }
    except Exception as e:
        error_msg = traceback.format_exc()
        print("!!!!!!!!!!! 发生严重错误 !!!!!!!!!!!")
        print(error_msg)
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

        # 返回给前端，方便查看
        return {
            "status": "error",
            "message": f"Server Error: {str(e)}",
            "traceback": error_msg
        }

app.mount("/videos", StaticFiles(directory=TEMP_DIR), name="videos")


# 任务状态枚举
class JobStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


# 任务数据模型
class JobInfo(BaseModel):
    id: str
    status: JobStatus
    progress: int = 0  # 简单的进度指示 (0-100) - 需要深度改造Engine才能实时更新，暂时先保留字段
    result_paths: Optional[list] = None
    error: Optional[str] = None


# 全局任务存储 (生产环境应该用 Redis)
# 结构: { "job_id": JobInfo }
JOBS: Dict[str, JobInfo] = {}

# 并发信号量：限制同时只能有 1 个视频在 GPU 上跑，防止显存溢出
# 如果你的显卡很强 (如 A100)，可以设为 2 或 3
GPU_SEMAPHORE = asyncio.Semaphore(1)


# --- 核心异步包装函数 ---
async def process_video_task(job_id: str, input_path: str, output_path: str, **kwargs):
    """
    后台任务处理函数
    """
    JOBS[job_id].status = JobStatus.PROCESSING

    # 获取信号量 (排队等待 GPU)
    async with GPU_SEMAPHORE:
        try:
            print(f"Task {job_id}: 获得 GPU 锁，开始处理...")

            # 【关键点】 run_in_threadpool
            # 这会将耗时的同步函数 engine.process_video 扔到单独的线程池中运行
            # 从而释放主事件循环，让 FastAPI 可以继续响应其他请求
            result_paths = await run_in_threadpool(
                engine.process_video,
                input_path=input_path,
                output_path=output_path,
                **kwargs
            )

            # 更新状态
            JOBS[job_id].status = JobStatus.COMPLETED
            JOBS[job_id].result_paths = result_paths
            print(f"Task {job_id}: 处理完成")

        except Exception as e:
            error_msg = traceback.format_exc()
            print(f"Task {job_id} Error: {error_msg}")
            JOBS[job_id].status = JobStatus.FAILED
            JOBS[job_id].error = str(e)


# --- API 接口 ---

@app.post("/reframe-job", response_model=JobInfo)
async def create_reframe_job(
        background_tasks: BackgroundTasks,
        file: UploadFile = File(...),
        ratio: str = Form("9:16"),
        mode: str = Form("normal"),
        target: str = Form("person"),
        multi_subject: bool = Form(False),
        split_screen: bool = Form(False),
        max_split: int = Form(3),
        debug: bool = Form(False)
):
    """
    提交任务接口：
    1. 保存文件
    2. 创建任务 ID
    3. 放入后台队列
    4. 立即返回 ID 给前端
    """
    # 1. 生成 ID 和 路径
    job_id = str(uuid.uuid4())
    input_filename = f"input_{job_id}_{file.filename}"
    input_path = os.path.join(TEMP_DIR, input_filename)

    output_filename = f"reframed_{job_id}_{file.filename}"
    if debug: output_filename = f"debug_{job_id}_{file.filename}"
    output_path = os.path.join(TEMP_DIR, output_filename)

    # 2. 保存文件 (异步读写)
    # UploadFile 很大时建议分块写，这里简化处理
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 3. 创建任务记录
    job_info = JobInfo(id=job_id, status=JobStatus.QUEUED)
    JOBS[job_id] = job_info

    # 4. 参数打包
    process_kwargs = {
        "ratio_str": ratio,
        "mode": mode,
        "detect_target": target,
        "multi_subject": multi_subject,
        "split_screen": split_screen,
        "max_split": max_split,
        "debug": debug
    }

    # 5. 添加到后台任务 (FastAPI 会在响应返回后执行它)
    background_tasks.add_task(
        process_video_task,
        job_id,
        input_path,
        output_path,
        **process_kwargs
    )

    # 6. 立即返回
    return job_info


@app.get("/status/{job_id}", response_model=JobInfo)
async def get_job_status(job_id: str):
    """
    轮询接口：前端每隔几秒调一次这个接口查询进度
    """
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="Job not found")

    return JOBS[job_id]