def get_aliyun_log_access():
    access_key_id, access_key = get_access_key()
    return {
        'access_key_id': access_key_id,
        'access_key': access_key,
    }


def get_access_key():
    file_ram_path = '/nas/zbase/security-credentials/ali_ram'

    access_key_id = "LTAI5tSBGyaBxbCmnBRPNKG9"
    access_key_secret = "PIc8HNGu8ANknf9MDnh8ifA3uV0P98"
    try:
        with open(file_ram_path, 'r') as f:
            content = f.read()
            access_key_id, access_key_secret = content.split('\n')[0:2]
    except Exception as e:
        pass
    finally:
        return access_key_id, access_key_secret