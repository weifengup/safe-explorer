import tensorflow as tf
import numpy as np
import time
from datetime import datetime
from safe_explorer.core.config import Config
from safe_explorer.core.replay_buffer import ReplayBuffer
from safe_explorer.core.tensorboard import TensorBoard
from safe_explorer.safety_layer.constraint_model import ConstraintModel

import warnings

# 取消特定类型的警告，例如 DeprecationWarning
warnings.filterwarnings("ignore", category=DeprecationWarning)

# 或者取消所有警告
warnings.filterwarnings("ignore")

class SafetyLayer:
    def __init__(self, env):
        self._env = env
        self._config = Config.get().safety_layer.trainer
        self._num_constraints = env.get_num_constraints()
        self._initialize_constraint_models()
        self._replay_buffer = ReplayBuffer(self._config.replay_buffer_size)
        self._writer = TensorBoard.get_writer()
        self._train_global_step = 0
        self._eval_global_step = 0

    def _initialize_constraint_models(self):
        self._models = [ConstraintModel(self._env.observation_space.get("agent_position").shape[0],
                                        self._env.action_space.shape[0])
                        for _ in range(self._num_constraints)]
        self._optimizers = [tf.train.AdamOptimizer(learning_rate=self._config.lr)
                            for _ in range(self._num_constraints)]

    def _sample_steps(self, num_steps):
        episode_length = 0
        observation = self._env.reset()

        for step in range(num_steps):
            action = np.random.rand(self._env.action_space.shape[0])
            c = self._env.get_constraint_values()
            observation_next, _, done, _ = self._env.step(action)
            c_next = self._env.get_constraint_values()

            self._replay_buffer.add({
                "action": action,
                "observation": observation["agent_position"],
                "c": c,
                "c_next": c_next
            })

            observation = observation_next
            episode_length += 1

            if done or (episode_length == self._config.max_episode_length):
                observation = self._env.reset()

    def _evaluate_batch(self, batch, sess):
        observation = batch["observation"]
        action = batch["action"]
        c = batch["c"]
        c_next = batch["c_next"]

        gs = [x(observation) for x in self._models]

        c_next_predicted = [c[:, i] + tf.reduce_sum(tf.multiply(gs[i], action), axis=1)
                            for i in range(self._num_constraints)]
        losses = [tf.reduce_mean(tf.square(c_next[:, i] - c_next_predicted[i])) for i in range(self._num_constraints)]

        return losses

    def _update_batch(self, batch, sess):
        batch = self._replay_buffer.sample(self._config.batch_size)
        losses = self._evaluate_batch(batch, sess)

        train_ops = [optimizer.minimize(loss) for optimizer, loss in zip(self._optimizers, losses)]
        sess.run(tf.global_variables_initializer())

        return np.array([sess.run(loss) for loss in losses])

    def evaluate(self, sess):
        self._sample_steps(self._config.evaluation_steps)

        # self._eval_mode()
        losses = [self._evaluate_batch(batch, sess) for batch in
                  self._replay_buffer.get_sequential(self._config.batch_size)]

        losses = np.mean(np.concatenate(losses).reshape(-1, self._num_constraints), axis=0)

        self._replay_buffer.clear()

        self._eval_global_step += 1

        # self._train_mode()

        print("Validation completed, average loss:", losses)

    def train(self):
        start_time = time.time()

        print("==========================================================")
        print("Initializing constraint model training...")
        print("----------------------------------------------------------")
        print("Start time:", datetime.fromtimestamp(start_time))
        print("==========================================================")

        sess = tf.Session()
        init = tf.global_variables_initializer()
        sess.run(init)

        for epoch in range(self._config.epochs):
            self._sample_steps(self._config.steps_per_epoch)

            losses = np.mean(np.concatenate([self._update_batch(batch, sess) for batch in
                                             self._replay_buffer.get_sequential(self._config.batch_size)]).reshape(-1,
                                                                                                                   self._num_constraints),
                             axis=0)

            self._replay_buffer.clear()

            self._train_global_step += 1

            print("Finished epoch", epoch, "with losses:", losses)
            print("Running validation ...")
            self.evaluate(sess)
            print("----------------------------------------------------------")

        self._writer.close()
        sess.close()

        print("==========================================================")
        print("Finished training constraint model. Time spent:", int(time.time() - start_time), "secs")
        print("==========================================================")
