import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def sample_jester_data(file_name, context_dim = 32, num_actions = 8, num_contexts = 19181,
    shuffle_rows=True, shuffle_cols=False):
    """Samples bandit game from (user, joke) dense subset of Jester dataset.
    Args:
       file_name: Route of file containing the modified Jester dataset.
       context_dim: Context dimension (i.e. vector with some ratings from a user).
       num_actions: Number of actions (number of joke ratings to predict).
       num_contexts: Number of contexts to sample.
       shuffle_rows: If True, rows from original dataset are shuffled.
       shuffle_cols: Whether or not context/action jokes are randomly shuffled.
    Returns:
       dataset: Sampled matrix with rows: (context, rating_1, ..., rating_k).
       opt_vals: Vector of deterministic optimal (reward, action) for each context.
    """
    np.random.seed(0)
    with tf.gfile.Open(file_name, 'rb') as f:
       dataset = np.load(f)
    if shuffle_cols:
       dataset = dataset[:, np.random.permutation(dataset.shape[1])]
    if shuffle_rows:
       np.random.shuffle(dataset)
    dataset = dataset[:num_contexts, :]
    assert context_dim + num_actions == dataset.shape[1], 'Wrong data dimensions.'
    opt_actions = np.argmax(dataset[:, context_dim:], axis=1)
    opt_rewards = np.array([dataset[i, context_dim + a] for i, a in enumerate(opt_actions)])

    return dataset, opt_rewards, opt_actions


class UCB():
    def __init__(self, num_features, num_arms):
        self.A = np.array([np.identity(num_features) for _ in range(0, num_arms)])
        self.b = np.zeros((num_arms, num_features))


def train_model(ucb, dataset, opt_rewards, opt_actions, alpha = 0.2):
    best_regret = []
    best_alpha = alpha
    best_avg_regret = float('inf')
    for j in range(0,5):
        lr = alpha/(5**j)
        diff = 0
        for i in range(0, 18000):
            payoff = np.zeros((8,1))
            x_t_a = np.reshape(dataset[i,:32], (-1, 1))
            r = dataset[i, 32:]
            theta = np.zeros((8, 32))
            for arm in range(0, 8):
                inv_A = np.linalg.inv(ucb.A[arm, :])
                theta[arm,:] =  np.matmul(inv_A, ucb.b[arm, :])
                payoff[arm] = np.matmul(theta[arm,:].T, x_t_a) + lr * np.sqrt(np.matmul(np.matmul(x_t_a.T, inv_A), x_t_a))
            best_arm = np.random.choice(np.where(payoff == payoff.max())[0])
            ucb.A[best_arm,:] = ucb.A[best_arm,:] + np.dot(x_t_a, x_t_a.T)
            ucb.b[best_arm,:] = ucb.b[best_arm,:] + r[best_arm] * x_t_a.T

            if opt_actions[i] != best_arm: diff += 1
            if i % 1000 == 999:
                print(i)

        regret = []
        for i in range(18000, dataset.shape[0]):
            payoff = np.zeros((8,1))
            x_t_a = np.reshape(dataset[i,:32], (-1, 1))
            r = dataset[i, 32:]
            theta = np.zeros((8, 32))
            for arm in range(0, 8):
                inv_A = np.linalg.inv(ucb.A[arm, :])
                theta[arm,:] =  np.matmul(inv_A, ucb.b[arm, :])
                payoff[arm] = np.matmul(theta[arm,:].T, x_t_a) + lr * np.sqrt(np.matmul(np.matmul(x_t_a.T, inv_A), x_t_a))
            best_arm = np.random.choice(np.where(payoff == payoff.max())[0])
            regret.append(opt_rewards[i] - r[best_arm])


        avg_regret = float(sum(regret))/len(regret)
        if avg_regret < best_avg_regret:
            best_avg_regret = avg_regret
            best_alpha = lr
            best_regret  = regret

    print(best_alpha)
    print(best_avg_regret)

    return best_regret

def plot_regret(regret):
    plt.plot(range(0,len(regret)), regret)
    plt.title('Regrets vs Testing')
    plt.xlabel('Testing Set')
    plt.ylabel('Regrets')
    plt.show()

def main():
    dataset, opt_rewards, opt_actions = sample_jester_data('jester_data_40jokes_19181users.npy')
    ucb = UCB(32, 8)
    regret = train_model(ucb, dataset, opt_rewards, opt_actions)
    plot_regret(regret)



if __name__ == "__main__":
    main()
