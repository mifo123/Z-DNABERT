import os
import shutil
import hashlib
import logging

class ModelManager:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def copy_file(self, source_path: str, target_path: str):
        hash1 = self.hash_file(source_path)
        hash2 = self.hash_file(target_path)
        if hash1 != hash2:
            self.logger.info('copying file to input directory')
            target_dir = os.path.dirname(target_path)
            if target_dir:
                os.makedirs(target_dir, exist_ok=True)
            shutil.copy2(source_path, target_path)
        else:
            self.logger.info("file hasn't changed")

    def hash_file(self, file_path: str):
        if not os.path.exists(file_path):
            return None
        h = hashlib.sha1()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest().encode()[:40]
