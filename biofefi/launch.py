import subprocess
import os
import logging


def main():
    app_path = os.path.join(os.path.dirname(__file__), "ui.py")
    try:
        subprocess.run(["streamlit", "run", app_path])
    except KeyboardInterrupt:
        logging.info("Shutting down BioFEFI...")
