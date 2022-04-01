# coding: utf-8
import matplotlib.pyplot as plt
import sys

start_length = len("DEBUG - ")
def parse_log_file(identifier, file='to_plot.log'):
    with open(file) as f:
        values = []
        for line in f.readlines():
            if identifier in line and line[:start_length] == 'DEBUG - ':
                value = float(line[start_length:].split(' ')[1])
                values.append(value)

    return values


def plot_q_losses(file='to_plot.log'):
    plt.plot(parse_log_file('q1_loss', file=file), label='q1_loss')
    plt.plot(parse_log_file('q2_loss', file=file), label='q2_loss')
    plt.legend()


def plot_q_grads(file='to_plot.log'):
    plt.plot(parse_log_file('q1_grad', file=file), label='q1_grad')
    plt.plot(parse_log_file('q2_grad', file=file), label='q2_grad')
    plt.legend()


if __name__=='__main__':
    if len(sys.argv) >= 2:
        file = sys.argv[1]
    else:
        file = 'to_plot.log'

    for identifier in ['reward', 'trajectory_reward', 'actor_loss', 'value_loss', 'actor_grad',
                       'value_grad', 'mus', 'sigmas', 'action', 'log_probs', 'value', 'q']:
        plt.plot(parse_log_file(identifier, file=file), label=identifier)
        plt.legend()
        plt.show()

    plot_q_losses(file=file)
    plt.title('q_losses')
    plt.show()
    plot_q_grads(file=file)
    plt.title('q_grads')
    plt.show()

