from __future__ import print_function

import itertools
from collections import namedtuple


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets

EpisodeStats = namedtuple("Stats",["episode_lengths", "episode_rewards"])

def render_episode(env, Q):
    screens = []
    state = env.reset()
    rewards = 0
    # First screen
    screens.append((env.render(mode='ansi'), None, 0))
    for t in itertools.count():
        best_action = np.argmax(Q[state])
        next_state, reward, done, _ = env.step(best_action)
        rewards += reward
        # Append screens
        screens.append((env.render(mode='ansi'), best_action, rewards))
        if done:
            break
        state = next_state
    return screens

def plot_episode_stats(stats, smoothing_window=10, noshow=False):

    # Plot the episode reward over time
    fig2 = plt.figure(figsize=(10,5))
    rewards_smoothed = pd.Series(stats.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
    plt.plot(rewards_smoothed)
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward (Smoothed)")
    plt.title("Episode Reward over Time (Smoothed over window size {})".format(smoothing_window))
    if noshow:
        plt.close(fig2)
    else:
        plt.show(fig2)

    # Plot the episode length over time
    fig1 = plt.figure(figsize=(10,5))
    plt.plot(stats.episode_lengths)
    plt.xlabel("Episode")
    plt.ylabel("Episode Length")
    plt.title("Episode Length over Time")
    if noshow:
        plt.close(fig1)
    else:
        plt.show(fig1)

    # Plot last rewards over time
    last_amount = min(100,len(stats.episode_rewards))
    fig4 = plt.figure(figsize=(10, 5))
    rewards_smoothed = pd.Series(stats.episode_rewards[-last_amount:])
    plt.plot(rewards_smoothed)
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward")
    plt.title("Last {} episode Rewards over Time".format(last_amount))
    if noshow:
        plt.close(fig4)
    else:
        plt.show(fig4)

    # Plot time steps and episode number
    fig3 = plt.figure(figsize=(10,5))
    plt.plot(np.cumsum(stats.episode_lengths), np.arange(len(stats.episode_lengths)))
    plt.xlabel("Time Steps")
    plt.ylabel("Episode")
    plt.title("Episode per time step")
    if noshow:
        plt.close(fig3)
    else:
        plt.show(fig3)


    return fig1, fig2, fig3, fig4

def play_ansi_sequence(ansi_sequence):
    def f(x):
        (screen, action, rewards) = ansi_sequence[x]
        print("\r{}.".format(screen), end="")
        return ("action:",action," rewards:",rewards)
    interact(f, x = widgets.IntSlider(min=0, max=len(ansi_sequence)-1, step=1, value=0))