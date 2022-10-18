import logging
import os

OK = '\033[92m'
WARNING = '\033[93m'
FAIL = '\033[91m'
END = '\033[0m'

PINK = '\033[95m'
BLUE = '\033[94m'
GREEN = OK
RED = FAIL
WHITE = END
YELLOW = WARNING


class MainLogger(logging.Logger):
    
    def __init__(self, level):
        """
        Initialize the logger with the name "root".
        """
        logging.Logger.__init__(self, "main", level)

        c_handler = logging.StreamHandler()
        c_handler.setLevel(logging.INFO)

        self.addHandler(c_handler)
        self.f_handler = None

    def set_output_folder(self, log_dir):
        if self.f_handler is not None:
            self.removeHandler(self.f_handler)

        log_file = os.path.join(log_dir, 'file.log')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        self.f_handler = logging.FileHandler(log_file, mode='a')
        self.f_handler.setLevel(logging.INFO)   
        self.addHandler(self.f_handler)

mainlogger = MainLogger(logging.WARNING)

class colorlogger():
    def __init__(self, log_dir, log_name='train_logs.txt'):
        # set log
        self._logger = logging.getLogger(log_name)
        self._logger.setLevel(logging.INFO)
        log_file = os.path.join(log_dir, log_name)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        file_log = logging.FileHandler(log_file, mode='a')
        file_log.setLevel(logging.INFO)
        console_log = logging.StreamHandler()
        console_log.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "{}%(asctime)s{} %(message)s".format(GREEN, END),
            "%m-%d %H:%M:%S")
        file_log.setFormatter(formatter)
        console_log.setFormatter(formatter)
        self._logger.addHandler(file_log)
        self._logger.addHandler(console_log)

    def debug(self, msg):
        self._logger.debug(str(msg))

    def info(self, msg):
        self._logger.info(str(msg))

    def warning(self, msg):
        self._logger.warning(WARNING + 'WRN: ' + str(msg) + END)

    def critical(self, msg):
        self._logger.critical(RED + 'CRI: ' + str(msg) + END)

    def error(self, msg):
        self._logger.error(RED + 'ERR: ' + str(msg) + END)

