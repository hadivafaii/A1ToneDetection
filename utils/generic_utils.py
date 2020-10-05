from datetime import datetime


def now():
    return datetime.now().strftime("[%Y_%m_%d_%H-%M]")
