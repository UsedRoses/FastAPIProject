from apscheduler.events import EVENT_JOB_EXECUTED, EVENT_JOB_ERROR, EVENT_JOB_REMOVED, JobEvent
from apscheduler.triggers.cron import CronTrigger


def my_listener(event: JobEvent):
    print(f'code: {event.code}, jobstore: {event.jobstore}, job_id: {event.job_id}')


def scheduler_start():
    pass
    # scheduler.add_listener(my_listener, EVENT_JOB_EXECUTED | EVENT_JOB_ERROR | EVENT_JOB_REMOVED)
    # 每5秒检查一次订单支付情况
    # scheduler.add_job(func=check_usdt_trc20_order, trigger=CronTrigger(second="*/5"), id="check_usdt_trc20_order",
    #                   replace_existing=True, coalesce=True)
    # scheduler.add_job(func=check_binance_off_chain_usdt_trc20_order, trigger=CronTrigger(second="*/6"), id="check_binace_off_chain_trc20_order",
    #                   replace_existing=True, coalesce=True)
    #
    # # 每15分钟同步一次汇率
    # scheduler.add_job(func=update_rate, trigger=CronTrigger(minute="*/5"), jobstore="mysql", id="update_rate",
    #                   replace_existing=True, coalesce=True)
    #
    # scheduler.start()


def scheduler_shutdown():
    # scheduler.shutdown()
    pass