import os
import time

UPLOAD_FOLDER = "static/uploads"
MAX_AGE = 12 * 60 * 60  # 12 hours in seconds

now = time.time()

for filename in os.listdir(UPLOAD_FOLDER):
    filepath = os.path.join(UPLOAD_FOLDER, filename)

    if os.path.isfile(filepath):
        file_age = now - os.path.getmtime(filepath)
        if file_age > MAX_AGE:
            try:
                os.remove(filepath)
                print(f"Deleted: {filename}")
            except Exception as e:
                print(f"Failed to delete {filename}: {e}")
