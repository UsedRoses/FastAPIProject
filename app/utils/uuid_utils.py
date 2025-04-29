import uuid
import string

BASE62 = string.digits + string.ascii_letters

def base62_encode(num, length=8):
    s = []
    base = 62
    while num:
        num, rem = divmod(num, base)
        s.append(BASE62[rem])
    return ''.join(reversed(s)).zfill(length)

def generate_base62_uuid_short_id(length=8):
    u = uuid.uuid4()
    # 取 uuid 的 int 值（前 64 位）
    short_int = u.int >> 64
    return base62_encode(short_int, length)

