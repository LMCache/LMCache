import redis
from typing import Optional
from lmcache.storage_backend.connector.base_connector import RemoteConnector

class RedisConnector(RemoteConnector):
    def __init__(self, host: str, port: int):
        self.connection = redis.Redis(host=host, port=port)

    def exists(self, key: str) -> bool:
        return self.connection.exists(key)

    def get(self, key: str) -> Optional[bytes]:
        return self.connection.get(key)

    def set(self, key: str, obj: bytes) -> None:
        self.connection.set(key, obj)

    def list(self):
        cursor = 0
        all_keys = []
    
        while True:
            cursor, keys = self.connection.scan(cursor=cursor, match='*')
            all_keys.extend(keys)
            if cursor == 0:
                break
    
        return [key.decode('utf-8') for key in all_keys]

