import os
import platform
import subprocess
import time


def isUsedPort(port):
    if os.popen('netstat -na | grep :' + str(port)).readlines():
        return True
    else:
        return False


def checkPort(startPort, number=4):
    machine = platform.platform().lower()
    if 'windows-' in machine:
        def isUsedPort(port):
            if os.popen('netstat -an | findstr :' + str(port)).readlines():
                return True
            else:
                return False
    elif 'linux-' in machine:
        def isUsedPort(port):
            if os.popen('netstat -na | grep :' + str(port)).readlines():
                return True
            else:
                return False
    else:
        print('Error, sorry, platform is unknown')
        exit(-1)

    free = False
    while not free:
        pointPort = startPort
        free = True
        for i in range(number):
            if isUsedPort(pointPort):
                free = False
                startPort = pointPort + number
                break
            else:
                pointPort = pointPort + 1
    return startPort


def kill_process(pid):
    os.popen(f'sudo kill -9 {int(pid)}')
    print(f'kill{pid}')


def get_pids(port):
    pid = os.popen("sudo netstat -anp|grep %s |awk '{print $7}'" % (str(port))).read().split('/')[0]
    return pid


def killServers(ports_list):
    for port in ports_list:
        pid = get_pids(port)
        kill_process(pid)


# 杀死占用端口号的ps进程
# ps_ports = ["23846", "23847", "23848", "23849", "23850", "23851"]
# kill_process(*get_pid(*ps_ports))


def startServer(startPort=2000, num=6, gpu_id=0):
    startPortList = []
    for i in range(num):
        interval = 3
        startPort = checkPort(startPort, interval)
        startPortList.append(startPort)
        subprocess.Popen(['docker', 'run', '-e', 'SDL_VIDEODRIVER=offscreen', '-it', '-p',
                          str(startPort) + '-' + str(startPort + interval - 1) + ':' +
                          str(startPort) + '-' + str(startPort + interval - 1),
                          '--runtime=nvidia', '--gpus', f'"device={gpu_id}"',
                          "carlasim/carla:0.9.11.1",
                          '/bin/bash', 'CarlaUE4.sh', '-opengl',
                          '-world-port=' + str(startPort)], shell=False,
                         stdout=subprocess.PIPE)  # subprocess.PIPE
        print(f'started \t {i} server | world port:{startPort}|traffic manager port:{startPort + interval}')
        startPort += (interval + 1)
        time.sleep(1)

    print(startPortList)
    return startPortList


if __name__ == '__main__':
    # startServer(startPort=2308, num=1, gpu_id=3)

    # startServer(startPort=2080, num=1, gpu_id=0)
    # startServer(startPort=2090, num=1, gpu_id=0)
    startServer(startPort=2070, num=1, gpu_id=1)
    startServer(startPort=2060, num=1, gpu_id=1)
    startServer(startPort=2050, num=1, gpu_id=1)
    # startServer(startPort=2800, num=4, gpu_id=0)
    # startServer(startPort=2700, num=4, gpu_id=0)
    # startServer(startPort=2600, num=4, gpu_id=0)
    # startServer(startPort=2500, num=4, gpu_id=3)
    # startServer(startPort=2400, num=4, gpu_id=0)
    startServer(startPort=2300, num=4, gpu_id=3)
    startServer(startPort=2200, num=4, gpu_id=2)
    startServer(startPort=2100, num=4, gpu_id=3)
    startServer(startPort=2000, num=4, gpu_id=2)

    # startServer(startPort=2016, num=2, gpu_id=2)
    # startServer(startPort=2024, num=2, gpu_id=3)
    # print(kill_process(get_pids(2004)))p
    # ports_list=[2004,2006,2008,2010]
    # for port in ports_list:
    #     pid = get_pids(port)
    #     print(pid)
    import sys

    sys.exit(0)
