import os
import shutil
from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from app.core.engine import SmartReframer

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
async def reframe_video(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """
    接收视频 -> 保存 -> 自动剪辑 -> 返回输出路径
    """
    # 1. 保存上传的视频
    input_filename = f"input_{file.filename}"
    input_path = os.path.join(TEMP_DIR, input_filename)

    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 2. 定义输出路径
    output_filename = f"reframed_{file.filename}"
    output_path = os.path.join(TEMP_DIR, output_filename)

    # 3. 运行处理 (这是一个耗时操作，生产环境建议放入 Celery 队列)
    # 这里演示直接调用
    try:
        result_path = engine.process_video(input_path, output_path, target_ratio=(9, 16))
        return {
            "status": "success",
            "message": "Video reframed successfully",
            "output_path": result_path
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }