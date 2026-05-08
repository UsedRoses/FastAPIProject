from common.init_app import init_middlewares, register_routers, app
from configuration.settings import settings

# register_exceptions(app)
init_middlewares(app)
register_routers(app, settings.ROUTER_DIR)


