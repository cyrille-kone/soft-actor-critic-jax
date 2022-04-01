#coding: utf-8
import matplotlib.pyplot as plt

def parse_log_file(identifier, file='to_plot.log'):
    with open(file) as f:
        values = []
        for line in f.readlines():
            if identifier in line:
                value = float(line[len("DEBUG - "):].split(' ')[1])
                values.append(value)

    return values


def plot_reward():
    plt.plot(parse_log_file('reward'))

def plot_actor_loss():
    plt.plot(parse_log_file('actor_loss'))

def plot_q_losses():
    plt.plot(parse_log_file('q1_loss'), label='q1_loss')
    plt.plot(parse_log_file('q2_loss'), label='q2_loss')
    plt.legend()

def plot_value_loss():
    plt.plot(parse_log_file('value_loss'))

def plot_actor_grad():
    plt.plot(parse_log_file('actor_grad'))

def plot_q_grads():
    plt.plot(parse_log_file('q1_grad'), label='q1_grad')
    plt.plot(parse_log_file('q2_grad'), label='q2_grad')
    plt.legend()

def plot_value_grad():
    plt.plot(parse_log_file('value_grad'))

def plot_mus():
    plt.plot(parse_log_file('mus'))

def plot_sigmas():
    plt.plot(parse_log_file('sigmas'))

def plot_actions():
    plt.plot(parse_log_file('actions'))

if __name__=='__main__':
    for identifier in ['reward', 'trajectory_reward', 'actor_loss', 'value_loss', 'actor_grad',
                       'value_grad', 'mus', 'sigmas', 'action', 'log_probs']:
        plt.plot(parse_log_file(identifier), label=identifier)
        plt.legend()
        plt.show()

    plot_q_losses()
    plt.title('q_losses')
    plt.show()
    plot_q_grads()
    plt.title('q_grads')
    plt.show()

