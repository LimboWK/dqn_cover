# also checkout this: https://goodboychan.github.io/python/reinforcement_learning/pytorch/udacity/2021/05/07/DQN-LunarLander.html

import gym
from  minDQN import Agent
from utils import plotLearning, show_video, show_video_of_model
import numpy as np
import torch as T

if __name__ == "__main__":
    env = gym.make("LunarLander-v2")
    agent = Agent(gamma=0.99, epsilon=1.0, batch_size=16, n_actions=4, eps_end=0.01,
        input_dims=[8], lr=0.001)
    scores, eps_history = [], []
    n_games = 500

    for i in range(n_games):
        score = 0
        done = False
        obs = env.reset()
        while not done:
            action = agent.choose_action(obs)
            new_obs, reward, done, info = env.step(action)
            score += reward
            agent.store_transition(obs, action, reward, new_obs, done)
            agent.learn()
            obs = new_obs
        scores.append(score)
        eps_history.append(agent.epsilon)
        avg_score = np.mean(scores)

        print("episode", i, "score %.2f" % score, "average_score %.2f" % avg_score, 
            "epsilon %.2f" % agent.epsilon
        )
        
    x = [i+1 for i in range(n_games)]
    filename = "lunar_lander_2020.png"
    plotLearning(x, scores, eps_history, filename)

    # save the model dict
    model_name = f"Q_eval_N{n_games}_checkpoint.pth"
    T.save(agent.Q_eval.state_dict(), f"/Users/kun.wan/workdir/gdsp/dqn_cover/base/dqn_play/checkpoints/{model_name}")
    
    show_video_of_model(agent=agent, 
                        env=gym.make("LunarLander-v2"),
                        )

    

    # visualization of the model
    """
    agent = Agent(state_size=8, action_size=4, seed=0)
    show_video_of_model(agent, 'LunarLander-v2')
    show_video('LunarLander-v2')
    """
