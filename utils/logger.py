import os

current_status = None
log_file_path = None

def status(msg):
    """ Set the current status
    """

    global current_status

    clear_status()
    print(msg, end='', flush=True)
    current_status = msg

def log(msg):
    """ Print a line to stdout and to the log file
    """

    #clear_status()
    log_current_status()
    log_to_file(msg)
    print(msg)

def log_current_status():
    """ log the current status
    """

    global current_status

    if current_status:
        print('')
        log_to_file(current_status)
        current_status = None

def set_logfile(path):
    global log_file_path

    os.makedirs(os.path.dirname(path), exist_ok=True)
    log_file_path = path

def clear_status():
    global current_status

    if current_status:
        print('\r\x1b[0K', end='') # Clear the current line
        current_status = None

def log_to_file(msg):
    global log_file_path

    if log_file_path:
        with open(log_file_path, 'a') as f:
            print(msg, file=f)
