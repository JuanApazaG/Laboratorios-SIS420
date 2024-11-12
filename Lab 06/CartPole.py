import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

def train(episodes):
    # Crear el entorno CartPole-v1
    env = gym.make("CartPole-v1")

    # Parámetros de Q-Learning
    learning_rate = 0.1           # Tasa de aprendizaje
    discount_factor = 0.95         # Factor de descuento
    epsilon = 1.0                 # Probabilidad inicial de exploración (acciones aleatorias)
    epsilon_decay_rate = 0.0001   # Tasa de decaimiento de epsilon
    min_epsilon = 0.01            # Epsilon mínimo
    num_bins = 20                 # Número de divisiones para discretizar el espacio de observación
    rng = np.random.default_rng() # Generador de números aleatorios

    # Divisiones para discretizar el espacio de estados
    obs_space_high = env.observation_space.high
    obs_space_low = env.observation_space.low
    obs_space_high[1], obs_space_low[1] = 0.5, -0.5  # Limitar la velocidad del carrito
    obs_space_high[3], obs_space_low[3] = 0.5, -0.5  # Limitar la velocidad angular
    bins = [np.linspace(low, high, num_bins) for low, high in zip(obs_space_low, obs_space_high)]

    # Función para discretizar observaciones continuas
    def discretize_state(state):
        state_idx = tuple(np.digitize(s, b) - 1 for s, b in zip(state, bins))
        return state_idx

    # Inicializar la tabla Q
    q_table = np.zeros([num_bins] * len(env.observation_space.low) + [env.action_space.n])

    # Almacenar recompensas por episodio
    rewards_por_episode = np.zeros(episodes)

    # Bucle de entrenamiento
    for i in range(episodes):
        # Reiniciar el entorno con renderización en ciertos episodios
        if (i + 1) % 100 == 0:
            env.close()
            env = gym.make("CartPole-v1", render_mode="human")
        else:
            env.close()
            env = gym.make("CartPole-v1")

        state, _ = env.reset()
        state = discretize_state(state)
        total_reward = 0

        done = False
        while not done:
            # Exploración o explotación
            if rng.random() < epsilon:
                action = env.action_space.sample()  # Exploración
            else:
                action = np.argmax(q_table[state])  # Explotación

            # Realizar acción y obtener el nuevo estado y recompensa
            next_state, reward, done, _, _ = env.step(action)
            next_state = discretize_state(next_state)

            # Actualizar la tabla Q
            best_next_action = np.argmax(q_table[next_state])
            q_table[state][action] += learning_rate * (reward + discount_factor * q_table[next_state][best_next_action] - q_table[state][action])

            state = next_state
            total_reward += reward

        # Almacenar recompensa y ajustar epsilon
        rewards_por_episode[i] = total_reward
        epsilon = max(min_epsilon, epsilon - epsilon_decay_rate)

        # Mostrar progreso cada 50 episodios
        if (i + 1) % 50 == 0:
            print(f"Episodio: {i + 1} - Recompensa: {np.mean(rewards_por_episode[max(0, i-49):i+1])}")

    # Cierra el entorno
    env.close()

    # Imprimir la tabla Q final
    print("Tabla Q resultante después del entrenamiento:")
    print(q_table)

    # Calcular recompensas acumuladas para análisis
    suma_rewards = np.zeros(episodes)
    for t in range(episodes):
        suma_rewards[t] = np.sum(rewards_por_episode[max(0, t - 50):(t + 1)])

    # Gráfico de recompensas acumuladas
    plt.plot(suma_rewards)
    plt.xlabel('Episodios')
    plt.ylabel('Suma de recompensas acumuladas')
    plt.title('Evolución de las recompensas acumuladas durante el entrenamiento')
    plt.show()

# Ejecución del entrenamiento
train(15000)
