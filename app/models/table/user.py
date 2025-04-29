from tortoise import fields
from tortoise.models import Model


class ZUser(Model):
    """
    中台用户信息
    """
    id = fields.BigIntField(pk=True, description="自增主键ID")
    username = fields.CharField(max_length=60, null=True, description="用户名")
    email = fields.CharField(max_length=60, description="用户邮箱")

    class Meta:
        table = "user"
        table_description = "中台用户信息"

class SocialAccount(Model):
    """
    中台用户信息
    """
    id = fields.BigIntField(pk=True, description="自增主键ID")
    user_id = fields.BigIntField(unique=True, description="用户编号")
    username = fields.CharField(max_length=60, null=True, description="用户名")
    email = fields.CharField(max_length=60, description="用户邮箱")

    class Meta:
        table = "social_account"
        table_description = "中台用户信息"