import matplotlib.pyplot as plt

with open('./scores.csv', 'r') as file:
    _, *names = file.readline().strip().split(';')
    names = [str(name[1:-1]) for name in names]

    e, w_e, c, w_c = [], [], [], []
    for line in file:
        k, *values = line.strip().split(';')
        e.append(float(values[0]))
        w_e.append(float(values[1]))
        c.append(float(values[2]))
        w_c.append(float(values[3]))
    fig = plt.figure(figsize=(10, 5))
    ep, = plt.plot(range(1, 11), e, label=names[0])
    w_ep, = plt.plot(range(1, 11), w_e, label=names[1])
    cp, = plt.plot(range(1, 11), c, label=names[2])
    w_cp, = plt.plot(range(1, 11), w_c, label=names[3])
    plt.legend([ep, w_ep, cp, w_cp], names)
    plt.xlabel('k')
    plt.xticks(range(1, 11))
    plt.ylabel('Accuracy')
    fig.savefig('scores_plot.png')

