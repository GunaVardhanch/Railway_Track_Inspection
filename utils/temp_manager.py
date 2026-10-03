import os
import tempfile
from contextlib import contextmanager
from typing import Generator

@contextmanager
def temporary_video_file(uploaded_file) -> Generator[str, None, None]:
    """
    Atomic context manager that handles either a direct video file path or
    streams uploaded video BytesIO to a NamedTemporaryFile (.mp4)
    and guarantees file cleanup/unlinking upon exit.
    """
    if isinstance(uploaded_file, str) and os.path.exists(uploaded_file):
        yield uploaded_file
        return

    suffix = ".mp4"
    if hasattr(uploaded_file, "name") and uploaded_file.name:
        ext = os.path.splitext(uploaded_file.name)[1].lower()
        if ext in [".mp4", ".avi", ".mov", ".mkv"]:
            suffix = ext

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    temp_path = temp_file.name

    try:
        # Seek to start if file-like
        if hasattr(uploaded_file, "seek"):
            uploaded_file.seek(0)

        # Stream chunks to prevent high memory consumption with large video uploads
        chunk_size = 1024 * 1024  # 1 MB chunk
        while True:
            chunk = uploaded_file.read(chunk_size)
            if not chunk:
                break
            temp_file.write(chunk)

        temp_file.flush()
        temp_file.close()

        import gc
        gc.collect()

        yield temp_path

    finally:
        # Close handle if not already closed
        try:
            temp_file.close()
        except Exception:
            pass

        # Unlink file cleanly
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass
