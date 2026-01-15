

class FileRegister:
    def __init__(self):
        self._read_func = {}

    def register(self, suffix: str):
        def decorator(func):
            self._read_func[suffix] = func
            return func
        return decorator

    def __getitem__(self, item: str):
        if item not in self._read_func:
            raise ValueError("Unsupported file type.")
        return self._read_func[item]

    def get_supported_types(self):
        return list(self._read_func.keys())

    def read_file_content(self, file_path: str):
        idx = file_path.rindex(".")
        suffix = file_path[idx:]
        if suffix not in self._read_func:
            raise ValueError("Unsupported file type.")
        read_func = self._read_func[suffix]
        return read_func(file_path)

    @staticmethod
    def import_register_item(module):
        import importlib
        importlib.import_module(module)


register = FileRegister()
register.import_register_item("rag.preprocess.supported")


__all__ = ["register"]