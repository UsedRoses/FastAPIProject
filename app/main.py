from common.init_app import register_exceptions, init_middlewares, register_routers, app
from common.public_configuration.public_settings import settings

register_exceptions(app)
init_middlewares(app)
register_routers(app, settings.ROUTER_DIR)


