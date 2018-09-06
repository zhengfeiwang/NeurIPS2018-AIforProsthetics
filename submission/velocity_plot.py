import os
import matplotlib.pyplot as plt

log_dir = './'
log_files = sorted([log for log in os.listdir(log_dir) if '.txt' in log])
print('log files:', log_files)

plt.figure()
plt.grid(True)
plt.title('submit velocity distribution')
plt.xlabel('timestamp')
plt.ylabel('velocity')

target_x = [i + 1 for i in range(300)]
target_y = [3.0] * 300
plt.plot(target_x, target_y, ':', color='red', label='target - 2700')

for log in log_files:
    lines = None
    score = None
    with open(log, 'r') as fp:
        lines = fp.readlines()
        score = lines[606].split(': ')[1].replace('\n', '')
        lines = lines[6:306]

    print('log file:', log, 'score:', score)

    x_axis = []
    y_axis = []

    for line in lines:
        line = line.replace('\n', '')
        timestamp = int(line.split(' score')[0].replace(' ', '').split('=')[1])
        velocity = float(line.split('velocity=')[1])
        x_axis.append(timestamp)
        y_axis.append(velocity)

    plt.plot(x_axis, y_axis, label=log.split('.')[0].split('log_')[1] + ' - ' + score)

plt.legend()
plt.savefig('velocity_distribution.png', dpi=300)
