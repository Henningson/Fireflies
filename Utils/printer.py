class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class Printer:
    def Warning(message):
        print(bcolors.WARNING + "WARNING: " + message + bcolors.ENDC)

    def Error(message):
        print(bcolors.FAIL + "ERROR: " + message + bcolors.ENDC)

    def OKG(message):
        print(bcolors.OKGREEN + "ERROR: " + message + bcolors.ENDC)

    def OKC(message):
        print(bcolors.OKCYAN + "ERROR: " + message + bcolors.ENDC)

    def OKB(message):
        print(bcolors.OKCYAN + "ERROR: " + message + bcolors.ENDC)

    def KV(key, value):
        print("{0}{1}: {2}{3}{4}".format(bcolors.OKCYAN, key, bcolors.OKGREEN, value, bcolors.ENDC))

    def KV2(key, value):
        print("{0}{1}: {2}{3}{4}".format(bcolors.OKCYAN, key, bcolors.OKBLUE, value, bcolors.ENDC))

    def Header(message):
        print("{0}{1}{2}".format(bcolors.BOLD, message, bcolors.ENDC))