import asyncio
import random
import time
import urllib.parse
from telegram import (
    Update,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    BotCommand,
)
from telegram.constants import ChatType
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    CallbackQueryHandler,
    MessageHandler,
    filters,
    ContextTypes, Application,
)

# 替换为你自己的 Bot Token
BOT_TOKEN = "7599945531:AAGLJ5WXoLS2g_dVzuQeELRyg4TMriMc1ZE"

# 不同群组对应的“商品配置”（群组 ID 为键）
GROUP_PRODUCTS = {
    -1001111111111: {  # 群组 A 的 chat_id
        "product_1": {"name": "A组基础套餐", "price": 9.9},
        "product_2": {"name": "A组高级套餐", "price": 19.9},
    },
    -1002222222222: {  # 群组 B 的 chat_id
        "product_1": {"name": "B组旗舰套餐", "price": 49.9},
        "product_2": {"name": "B组至尊套餐", "price": 99.9},
    },
    # … 其它群组
}

# 私聊模式下默认商品（可选）
DEFAULT_PRODUCTS = {
    "product_1": {"name": "通用基础套餐", "price": 9.9},
    "product_2": {"name": "通用高级套餐", "price": 19.9},
}

# 存储订单：order_id -> dict
ORDERS: dict = {}


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    处理 /start，无论群聊还是私聊。
    如果带参数 (context.args)，并且参数是 "group_<group_id>"，
    则在私聊中展示对应群组的商品；否则展示默认菜单。
    """
    args = context.args
    print("args", args)
    print("update", update)
    user = update.effective_user
    if update.effective_chat.type == "private" and args:
        payload = args[0]
        print(payload)
        if payload.startswith("group_"):
            group_id = int(payload.split("_", 1)[1])
            products = GROUP_PRODUCTS.get(group_id, DEFAULT_PRODUCTS)
            await send_product_list(
                chat_id=update.effective_chat.id,
                products=products,
                context=context,
                origin_group=group_id
            )
            return

    if update.effective_chat.type == ChatType.PRIVATE:
        # 能到这里表示用户在单独私聊,且不是从群中跳转过来的
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text="抱歉,我不知道你的期望商品列表,请在商品沟通群中联系我,谢谢！",
        )
        return

    # 普通 /start
    keyboard = [
        [InlineKeyboardButton("🛍 查看商品", callback_data="show_products")],
        [InlineKeyboardButton("💬 群内继续交流", callback_data="back_to_group")],
    ]
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text=f"@{user.full_name}，欢迎来到私聊客服，请选择：",
        reply_markup=InlineKeyboardMarkup(keyboard),
    )

async def show_products_in_group(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    群内“查看商品”回调——展示本群组商品，并附加“私聊我”按钮。
    """
    query = update.callback_query
    await query.answer()
    group_id = query.message.chat.id
    products = GROUP_PRODUCTS.get(group_id, DEFAULT_PRODUCTS)
    await send_product_list(
        chat_id=group_id,
        products=products,
        context=context,
        origin_group=group_id,
        from_query=query
    )

async def send_product_list(chat_id: int, products: dict, context: ContextTypes.DEFAULT_TYPE,
                            origin_group: int, from_query=None) -> None:
    """
    统一发送商品列表，并在群内附加“私聊”按钮。
    from_query: 如果不为 None，则挂在回调消息下回复。
    """
    keyboard = []
    # 列出商品购买按钮
    for pid, info in products.items():
        keyboard.append([
            InlineKeyboardButton(
                f"{info['name']} – ¥{info['price']}",
                callback_data=f"buy_{pid}_{origin_group}"
            )
        ])
    # 如果是在群里，还加一个“私聊我”按钮
    if from_query:
        # 构造深度链接，携带群组 ID
        payload = f"group_{origin_group}"
        link = f"https://t.me/{context.bot.username}?start={urllib.parse.quote(payload)}"
        keyboard.append([InlineKeyboardButton("📩 私聊我购买", url=link)])

    reply_markup = InlineKeyboardMarkup(keyboard)
    text = "请选择您要购买的商品："
    if from_query:
        await context.bot.send_message(
            chat_id=chat_id,
            text=text,
            reply_markup=reply_markup,
            reply_to_message_id=from_query.message.message_id,
        )
    else:
        await context.bot.send_message(
            chat_id=chat_id,
            text=text,
            reply_markup=reply_markup,
        )

group_last_welcome = {}
GROUP_COOLDOWN = 30

async def welcome_new_member(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    now = time.time()

    if chat_id in group_last_welcome and now - group_last_welcome[chat_id] < GROUP_COOLDOWN:
        return  # 冷却中，不欢迎

    group_last_welcome[chat_id] = now
    names = [member.full_name for member in update.message.new_chat_members]
    await update.message.reply_text(f"👋 欢迎 {'、'.join(names)} 加入本群！")

async def send_payment(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    用户点击“购买”后生成订单并发送支付按钮。
    callback_data 格式： buy_<product_id>_<origin_group>
    """
    query = update.callback_query
    await query.answer()
    data = query.data.split("_", 2)
    product_id, origin_group = data[1], int(data[2])
    products = GROUP_PRODUCTS.get(origin_group, DEFAULT_PRODUCTS)
    product = products[product_id]

    user_id = query.from_user.id
    order_id = f"ORD{random.randint(10000, 99999)}"
    ORDERS[order_id] = {
        "user_id": user_id,
        "product_id": product_id,
        "group_id": origin_group,
        "status": "pending",
        "msg_id": query.message.message_id
    }

    payment_link = f"https://yourdomain.com/pay?order_id={order_id}"
    keyboard = [
        [InlineKeyboardButton("💳 前往支付", url=payment_link)],
        [InlineKeyboardButton("🔐 打开 @wallet 钱包", url="https://t.me/wallet")],
        [InlineKeyboardButton("✅ 我已完成付款", callback_data=f"check_{order_id}_{user_id}")]
    ]
    await context.bot.send_message(
        chat_id=query.message.chat.id,
        text=(
            f"@{query.from_user.username or query.from_user.first_name}，您选择了【{product['name']}】\n"
            f"金额：¥{product['price']}\n\n"
            "请点击下方按钮完成支付："
        ),
        reply_markup=InlineKeyboardMarkup(keyboard),
        reply_to_message_id=query.message.message_id
    )

async def check_payment(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    用户点击“我已完成付款”后检查状态。
    callback_data 格式： check_<order_id>_<bind_user_id>
    """
    query = update.callback_query
    await query.answer()
    _, order_id, bind_uid = query.data.split("_")
    bind_uid = int(bind_uid)

    # 权限校验
    if query.from_user.id != bind_uid:
        await query.answer("❌ 这不是您的订单，无法操作。", show_alert=True)
        return

    order = ORDERS.get(order_id)
    if not order:
        await query.answer("❌ 这不是您的订单，无法操作。", show_alert=True)
        return

    # 模拟检查
    await context.bot.send_message(
        chat_id=query.message.chat.id,
        text="正在检查付款状态，请稍后…",
        reply_to_message_id=order["msg_id"]
    )
    await asyncio.sleep(2)

    if random.choice([True, False]):
        order["status"] = "paid"
        products = GROUP_PRODUCTS.get(order["group_id"], DEFAULT_PRODUCTS)
        product = products[order["product_id"]]
        await context.bot.send_message(
            chat_id=query.message.chat.id,
            text=(
                f"✅ @{query.from_user.username or query.from_user.first_name}，"
                f"您的付款已到账！已为您开通【{product['name']}】服务。"
            ),
            reply_to_message_id=order["msg_id"]
        )
    else:
        await context.bot.send_message(
            chat_id=query.message.chat.id,
            text="⚠️ 尚未检测到您的付款，请稍后重试。",
            reply_to_message_id=order["msg_id"]
        )

async def back_to_group(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    私聊中“群内继续交流”按钮回调，发回群组提示。
    """
    query = update.callback_query
    await query.answer()
    # 假设我们知道用户来自哪个群组，可在 ORDERS 或上下文记录
    # 这里只演示发送一条普通提示
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text="请在群组中继续操作，谢谢！"
    )

async def default_reply(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user  # 通常比 callback_query 更通用安全
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text=f"@{user.username or user.first_name}，请输入 /start 重新开始。"
    )

async def on_post_init(app: Application) -> None:
    # 注册命令，支持自动补全
    await app.bot.set_my_commands([
        BotCommand("start", "启动欢迎"),
        BotCommand("help",  "帮助信息"),
    ])

def main() -> None:
    app = (
        ApplicationBuilder()
        .token(BOT_TOKEN)
        .post_init(on_post_init)
        .build()
    )
    # 群组内命令
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CallbackQueryHandler(show_products_in_group, pattern="^show_products$"))
    app.add_handler(CallbackQueryHandler(send_payment, pattern="^buy_"))
    app.add_handler(CallbackQueryHandler(check_payment, pattern="^check_"))

    # 私聊模式
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CallbackQueryHandler(back_to_group, pattern="^back_to_group$"))

    # 默认回复
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, default_reply))
    # 新人入群欢迎
    app.add_handler(MessageHandler(filters.StatusUpdate.NEW_CHAT_MEMBERS, welcome_new_member))

    print("🤖 Bot 启动中…")
    app.run_polling()
    print("🤖 Bot 启动成功…")

if __name__ == "__main__":
    main()
