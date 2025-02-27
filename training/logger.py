import os
from datetime import datetime

class Logger:
    # class for logging
    def __init__(self, log_file):
        # create log file on disk
        self.log_file = log_file
        with open(self.log_file, "w") as file:
            file.write(f"Log file created: {datetime.now().isoformat()}\n")

    def log(self, message):
        # method for logging
        print(message)

        # get the timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # add timestamp to message
        log_message = f"[{timestamp}] {message}\n"

        # write message to file
        with open(self.log_file, "a") as file:
            file.write(log_message)