import warnings

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from dataclasses import dataclass
from neuronav.utils import softmax


@dataclass
class AgentParams:
    poltype: str
    beta: float
    epsilon: float
    learning_rate: float
    gamma: float


def plot_grid_experiment_results(results, num_eps):
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    fig = plt.figure(figsize=(8.25, 2), dpi=200)
    for idx, an in enumerate(results):
        mu = np.array(results[an]).mean(0)
        std = np.array(results[an]).std(0)
        plt.plot(mu, label=an, color=colors[idx], linewidth=2)
        plt.fill_between(
            np.arange(len(mu)), (mu - std), (mu + std), color=colors[idx], alpha=0.1
        )
    fig.axes[0].spines["top"].set_visible(False)
    fig.axes[0].spines["right"].set_visible(False)
    plt.vlines(x=num_eps // 2, ymin=0, ymax=100, colors="gray", ls=":", lw=2)
    plt.xlabel("Episodes", fontsize=16)
    plt.ylabel("Time-steps", fontsize=16)
    plt.legend(fontsize=12, bbox_to_anchor=(1, 0.85))
    plt.show()


def plot_graph_experiment_results(
    agent_names, result_dict, result_name, save_fig=False
):
    fig, axs = plt.subplots(1, 4, figsize=(20, 2), dpi=(350))
    num_conditions = len(result_dict.keys())
    for i in range(len(agent_names)):
        means = [np.mean(result, axis=0)[i] for result in result_dict.values()]
        stds = [np.std(result, axis=0)[i] for result in result_dict.values()]
        axs[i].bar(result_dict.keys(), means, yerr=stds)
        axs[i].axis([-0.5, num_conditions - 0.5, 0, 1])
        axs[i].set_title(agent_names[i], fontsize=16)
        axs[i].spines["top"].set_visible(False)
        axs[i].spines["right"].set_visible(False)
        axs[i].set_xticklabels(result_dict.keys(), fontsize=14)
        axs[i].set_ylabel("Revaluation Score", fontsize=14)
    if save_fig:
        fig.savefig(f"{result_name}.pdf", bbox_inches="tight")


def plot_revaluation(scores, agent_names):
    scores_mean = np.mean(scores, axis=0)
    scores_std = np.std(scores, axis=0)
    plt.axis([-1, len(scores[0]), 0, 1])
    plt.bar([name for name in agent_names], scores_mean, yerr=scores_std)


def get_scores(condition_func, agent_dict, num_reps=1, plot=False):
    all_scores = []
    for _ in range(num_reps):
        all_scores.append(condition_func(agent_dict.values()))
    all_scores = np.array(all_scores)
    if plot:
        plot_revaluation(all_scores, agent_dict.keys())
    return all_scores


def init_agents(env, agent_types, agent_params):
    agents = [
        agent_type(
            env.state_size,
            env.action_space.n,
            gamma=agent_params.gamma,
            lr=agent_params.learning_rate,
            poltype=agent_params.poltype,
            beta=agent_params.beta,
            epsilon=agent_params.epsilon,
        )
        for agent_type in agent_types
    ]
    return agents


def calc_revaluation(prefs_a, prefs_b):
    scores = []
    temp = 5
    for idx, pref in enumerate(prefs_a):
        a = softmax(pref[2] / temp)
        b = softmax(prefs_b[idx][2] / temp)
        score = np.mean(np.abs(a - b)[0])
        scores.append(score)
    return scores


def plot_values_and_policy(agent, env, start_pos, plot_title=None):
    arrows = [
        [0, 0.5, 0, -0.5],
        [-0.5, 0, 0.5, 0],
        [0, -0.5, 0, 0.5],
        [0.5, 0, -0.5, 0],
    ]

    fig, ax = plt.subplots()

    V = agent.Q.mean(0)
    im = ax.imshow(
        V.reshape(env.grid_size, env.grid_size), cmap="RdBu", vmin=-1.0, vmax=1.0
    )
    for i in range(env.grid_size):
        for j in range(env.grid_size):
            use_dir = agent.Q.argmax(0).reshape(env.grid_size, env.grid_size)[i, j]
            use_arrow = arrows[use_dir].copy()
            use_arrow[0] += j
            use_arrow[1] += i
            if [i, j] not in env.blocks:
                if (i, j) == tuple(start_pos):
                    use_alpha = 1.0
                else:
                    use_alpha = 0.5
                ax.arrow(
                    use_arrow[0],
                    use_arrow[1],
                    use_arrow[2],
                    use_arrow[3],
                    head_width=0.33,
                    head_length=0.33,
                    fc="k",
                    ec="k",
                    alpha=use_alpha,
                )
            else:
                box = patches.Rectangle(
                    (j - 0.33, i - 0.33), 0.66, 0.66, color="black", alpha=0.25
                )
                ax.add_patch(box)
    plt.colorbar(im)
    if plot_title != None:
        ax.set_title(plot_title)
    fig.show()
