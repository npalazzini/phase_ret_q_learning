import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import random

from phase_ret_env import environment, print_modulus_raw

tf.random.set_seed(1991)
np.random.seed(1991)
random.seed(1991)

# Configuration paramaters for the whole setup
gamma = 0.99  # Discount factor for past rewards
epsilon = 1.0  # Epsilon greedy parameter
epsilon_min = 0.1  # Minimum epsilon greedy parameter
epsilon_max = 1.0  # Maximum epsilon greedy parameter
epsilon_interval = (epsilon_max - epsilon_min)  # Rate at which to reduce chance of random action being taken
batch_size = 32  # Size of batch taken from replay buffer
max_steps_per_episode = 2000

N_EPISODES=100

num_actions = 9

def create_q_model():
    inputs = layers.Input(shape=(200, 200, 1))

    layer1 = layers.Conv2D(32, 8, strides=4, activation="relu")(inputs)
    layer2 = layers.Conv2D(64, 4, strides=2, activation="relu")(layer1)
    layer3 = layers.Conv2D(64, 3, strides=1, activation="relu")(layer2)

    layer4 = layers.Flatten()(layer3)

    layer5 = layers.Dense(512, activation="relu")(layer4)
    action = layers.Dense(num_actions, activation="linear")(layer5)

    return keras.Model(inputs=inputs, outputs=action)


# The first model makes the predictions for Q-values which are used to
# make a action.
model = create_q_model()
# Build a target model for the prediction of future rewards.
model_target = create_q_model()


"""
## Train
"""

optimizer = keras.optimizers.Adam(learning_rate=0.00025, clipnorm=1.0)

env=environment(HIO_ITERATIONS=60,
                ER_ITERATIONS=10,
                SQUARE_ROOT=True,
                MASK=True,
                POISSON=False,
                photons=0,
                IMPOSE_REALITY=True,
                HIO_BETA=0.9,
                R_COEFF=1,
                num_actions=num_actions,
                error_target=0)

# Experience replay buffers
action_history = []
state_history = []
state_next_history = []
rewards_history = []
done_history = []
frame_count = 0

# Number of frames to take random action and observe output
epsilon_random_frames = 10000
# Number of frames for exploration
epsilon_greedy_frames = 180000.0
# Maximum replay length
max_memory_length = 10000
# How often train the model
update_after_actions = 4
# How often to update the target network
update_target_network = 2000
# Using huber loss for stability
loss_function = keras.losses.Huber()

output_file=open("OUTPUT/RLdata.dat", "w", buffering=1)

for episode in range(N_EPISODES):

    error_file=open("OUTPUT/error.dat", "w", buffering=1)

    state = np.array(env.reset())

    episode_reward = 0
    minimum=1 #huge value
    best_dens=env.data
    best_supp=env.support

    for timestep in range(max_steps_per_episode):

        frame_count += 1

        # Use epsilon-greedy for exploration
        if frame_count < epsilon_random_frames or epsilon > np.random.rand(1)[0]:
            # Take random action
            action = np.random.choice(num_actions)
        else:
            # Predict action Q-values
            # From environment state
            state_tensor = tf.convert_to_tensor(state)
            state_tensor = tf.expand_dims(state_tensor, 0)
            action_probs = model(state_tensor, training=False)
            # Take best action
            action = tf.argmax(action_probs[0]).numpy()

        # Decay probability of taking random action
        epsilon -= epsilon_interval / epsilon_greedy_frames
        epsilon = max(epsilon, epsilon_min)

        # Apply the sampled action in our environment
        state_next, reward, done, error = env.step(action)
        state_next = np.array(state_next)

        episode_reward += reward

        # Save actions and states in replay buffer
        action_history.append(action)
        state_history.append(state)
        state_next_history.append(state_next)
        done_history.append(done)
        rewards_history.append(reward)
        state = state_next

        error_file.write(str(error)+"\t"+str(action)+"\n")

        if frame_count % update_after_actions == 0 and len(done_history) > batch_size:

            # Get indices of samples for replay buffers
            indices = np.random.choice(range(len(done_history)), size=batch_size)

            # Using list comprehension to sample from replay buffer
            state_sample = np.array([state_history[i] for i in indices])
            state_next_sample = np.array([state_next_history[i] for i in indices])
            rewards_sample = [rewards_history[i] for i in indices]
            action_sample = [action_history[i] for i in indices]
            done_sample = tf.convert_to_tensor([float(done_history[i]) for i in indices])

            # Build the updated Q-values for the sampled future states
            # Use the target model for stability
            future_rewards = model_target.predict(state_next_sample)
            # Q value = reward + discount factor * expected future reward
            updated_q_values = rewards_sample + gamma * tf.reduce_max(future_rewards, axis=1)

            # If final frame set the last value to -1
            updated_q_values = updated_q_values * (1 - done_sample) - done_sample

            # Create a mask so we only calculate loss on the updated Q-values
            masks = tf.one_hot(action_sample, num_actions)

            with tf.GradientTape() as tape:
                # Train the model on the states and updated Q-values
                q_values = model(state_sample)

                # Apply the masks to the Q-values to get the Q-value for action taken
                q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
                # Calculate loss between new Q-value and old Q-value
                loss = loss_function(updated_q_values, q_action)

            # Backpropagation
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

        if frame_count % update_target_network == 0:
            # update the the target network with new weights
            model_target.set_weights(model.get_weights())

        # Limit the state and reward history
        if len(rewards_history) > max_memory_length:
            del rewards_history[:1]
            del state_history[:1]
            del state_next_history[:1]
            del action_history[:1]
            del done_history[:1]

        if error<minimum:
            minimum=error
            best_dens=env.data
            best_supp=env.support

        if done:
            print("Epoch:", episode, ", steps: ",timestep, ", minimum: {:.8f}".format(minimum), ", reward: {}".format(reward) )
            break

        if timestep==max_steps_per_episode-1:
            print("Epoch:", episode, ", steps: ",timestep, ", minimum: {:.8f}".format(minimum), ", reward: {}".format(reward) )

    error_file.close()
    output_file.write(str(episode)+"\t"+str(timestep)+"\t"+str(minimum)+"\t"+str(episode_reward)+"\n")
    print("\n")
    print_modulus_raw(best_dens, "OUTPUT/best_density/best_density"+str(episode).zfill(2)+".pgm", 8)
    print_modulus_raw(best_supp, "OUTPUT/best_density/support"+str(episode).zfill(2)+".pgm", 8)

output_file.close()
