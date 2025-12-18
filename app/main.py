import os
import shutil
import traceback
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, Form
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
    multi_subject: bool = Form(False, description="是否开启多主体拆分模式")
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
            multi_subject=multi_subject
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