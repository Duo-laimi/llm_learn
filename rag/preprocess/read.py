import os
from tiktoken import get_encoding

from .register import register


class ReadFiles:
    def __init__(self, path, encoder=None):
        self.path = path
        self.file_list = self.get_files()
        if encoder is None:
            encoder = get_encoding("cl100k_base")
        self.encoder = encoder

    @staticmethod
    def get_suffix(path: str):
        idx = path.rindex(".")
        suffix = path[idx:]
        return suffix

    def get_files(self):
        if os.path.isfile(self.path):
            return [self.path]
        file_list = []
        supported_suffix = set(register.get_supported_types())
        for root, _, names in os.walk(self.path):
            for name in names:
                file_path = os.path.join(root, name)
                suffix = self.get_suffix(name)
                if suffix in supported_suffix:
                    file_list.append(file_path)
        return file_list

    def get_content(self, max_token_len: int = 600, conver_content: int = 150):
        docs = []
        for file in self.file_list:
            content = register.read_file_content(file)
            chunk_content = self.get_chunk(content, max_token_len=max_token_len, cover_content=conver_content)
            docs.extend(chunk_content)
        return docs

    # 文档切分函数，滑动窗口切分，每次切分保留一定重叠区域
    def get_chunk(self, text: str, max_token_len: int = 600, cover_content: int = 150):
        chunk_text = []
        curr_len = 0
        curr_chunk = ""
        lines = text.split("\n")
        for line in lines:
            line = line.replace(" ", "")
            line_len = len(self.encoder.encode(line))
            if line_len > max_token_len:
                print(f"Warning: line_len = {line_len}.")
            if curr_len + line_len <= max_token_len:
                curr_chunk += line + "\n"
                curr_len += line_len + 1
            else:
                chunk_text.append(curr_chunk)
                curr_chunk = curr_chunk[-cover_content:] + line
                curr_len = line_len + cover_content
        if curr_len > 0:
            chunk_text.append(curr_chunk)
        return chunk_text

