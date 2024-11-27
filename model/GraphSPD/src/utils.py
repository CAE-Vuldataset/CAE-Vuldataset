import sys
import time

# Logger: redirect the stream on screen and to file.
class Logger(object):
    def __init__(self, filename = "log.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        pass

start_time = time.time()
def RunTime():
    t_sec = time.time() - start_time
    t_min, t_sec = divmod(t_sec, 60)
    t_hr, t_min = divmod(t_min, 60)
    t_day, t_hr = divmod(t_hr, 24)
    pTime = f' [TIME: '
    if t_day: pTime += f'{int(t_day)} day'
    if t_hr: pTime += f'{int(t_hr)} hr '
    if t_min: pTime += f'{int(t_min)} min '
    pTime += f'{round(t_sec,3)} sec]'
    return pTime