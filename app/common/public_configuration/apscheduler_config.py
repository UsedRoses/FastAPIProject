from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from sqlalchemy import create_engine
from apscheduler.jobstores.redis import RedisJobStore

from common.public_configuration.public_settings import settings

# 通过 Mysql 存储任务
url = f"mysql+pymysql://{settings.DB_USER}:{settings.DB_PASSWORD}@{settings.DB_HOST}/{settings.DB_DATABASE}?charset=utf8mb4"
engine = create_engine(url, echo=False, future=True)
# 通过 Redis 存储任务
jobstores = {
    'default': RedisJobStore(db=settings.REDIS_DB, jobs_key='sv_jobs', run_times_key='sv_jobs_run_times', host=settings.REDIS_HOST, port=settings.REDIS_PORT, password=settings.REDIS_PASSWORD, username=settings.REDIS_USERNAME),
    'mysql': SQLAlchemyJobStore(engine=engine),
}
scheduler = AsyncIOScheduler(jobstores=jobstores)



