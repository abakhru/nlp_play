#!/usr/bin/env python

"""
- (Google Text-to-Speech)
pip install gTTS PyObjC AppKit
"""
import subprocess
import uuid
from pathlib import Path

from gtts import gTTS


class Speech2TextDemo:

    def __init__(self):
        # text = "Hello! My name is Amit."
        self.text = input('Enter the text to convert: ')
        self.temp_file = Path(f'/tmp/{uuid.uuid4()}.wav')
        self.convert_text_to_speech()
        self.play()

    def convert_text_to_speech(self):
        tts = gTTS(self.text)
        tts.save(self.temp_file)

    def play(self):
        subprocess.check_output(f'vlc --play-and-exit {self.temp_file}', shell=True)
        self.temp_file.unlink()


if __name__ == '__main__':
    p = Speech2TextDemo()
