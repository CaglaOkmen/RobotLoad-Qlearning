import numpy as np
import random
from gymnasium import Env, spaces
import os
import matplotlib.pyplot as plt
from matplotlib import colors
import imageio

# Robotu Ortamı Olusturma
class RobotEnv(Env):
    def __init__(self):
        super().__init__()
        
        # ASCII Harita
        self.desc = np.asarray([
            "+-------------+",
            "| : : : |B| : |",
            "| : | : : | : |",
            "|M: : : : | :M|",
            "| : : : | : : |",
            "|M: : :M| : :M|",
            "| : : : : : : |",
            "|M: | : : | :M|",
            "+-------------+",
        ], dtype='c')
        
        # Yasaklı Pozisyonlar (Makineler)
        self.mach_locs = [(2, 0), (4, 0), (6, 0), (4, 3), (2, 6), (4, 6), (6, 6)]
        # Yuk Konumları
        self.product_locs = [(2, 1), (4, 1), (6, 1), (4, 2), (2, 5), (4, 5), (6, 5)]
        self.num_products = len(self.product_locs)
        # Baslangıc ve Teslimat Noktası (B)
        self.start_pos = (0, 4)

        # Parametreler
        self.num_rows = 7 
        self.num_columns = 7
        self.max_capacity = 2  # Robot en fazla 2 yuk tasıyabilir
        self.num_load_states = self.max_capacity + 1 # Robot Yuk durumu: 0, 1 veya 2 
        self.num_mask_states = 2 ** self.num_products # Her Yuk icin 2 durum (alındı/alınmadı)
        self.num_states = self.num_rows * self.num_columns * self.num_mask_states * self.num_load_states # Toplam Durum Sayısı
        print(f"Toplam Durum Sayısı: {self.num_states}")
        self.action_space = spaces.Discrete(6)  # 0:Asagı, 1:Yukarı, 2:Saga, 3:Sola 4:Load, 5: Unload
        self.observation_space = spaces.Discrete(self.num_states)
        self.s = None

        # Gecis Tablosu
        self.initialize_transition_table()

        # Gorselleştirme Ayarları
        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        # Renkler: Zemin, Makine, Robot, Ürun, Başlangic/Boşaltım
        self.cmap = colors.ListedColormap(['white', '#424242', '#FFD54F', '#29B6F6', '#E0E0E0']) 
        self.bounds = [0, 1, 2, 3, 4, 5]
        self.norm = colors.BoundaryNorm(self.bounds, self.cmap.N)

    def encode(self, row, col, mask, load):
        i = row
        i *= self.num_columns
        i += col
        i *= self.num_mask_states
        i += mask
        i *= self.num_load_states
        i += load
        return i

    def decode(self, i):
        out = []
        out.append(i % self.num_load_states)
        i //= self.num_load_states
        out.append(i % self.num_mask_states) 
        i //= self.num_mask_states
        out.append(i % self.num_columns)
        i //= self.num_columns
        out.append(i)
        return reversed(out) # row, col, mask, load

    # Tum durumlar ve eylemler icin gecis olasılıklarını hesaplar
    def initialize_transition_table(self):
        self.P = {state: {action: [] for action in range(6)} for state in range(self.num_states)}
        for row in range(self.num_rows):
            for col in range(self.num_columns):
                for mask in range(self.num_mask_states):
                    for load in range(self.num_load_states):  
                        state = self.encode(row, col, mask, load) # Tum Durumlar
                        for action in range(6): # Tum eylemler 
                            self.process_action(state, row, col, mask, load, action)

    # Ajanın yaptıgı eylem sonucu ne olacagını belirler
    def process_action(self, state, row, col, mask, load, action):
        new_row, new_col = row, col
        new_mask = mask
        new_load = load
        reward = -1 # Adım maliyeti
        terminated = False
        
        if action == 0:   # Aşagı
            new_row = min(row + 1, self.num_rows - 1)
            # Yasaklı bolge kontrolu
            if (new_row, new_col) in self.mach_locs:
                new_row, new_col = row, col
                reward = -5
        elif action == 1: # Yukarı
            new_row = max(row - 1, 0)
            if (new_row, new_col) in self.mach_locs:
                new_row, new_col = row, col
                reward = -5
        elif action == 2: # Saga
            if self.desc[1 + row, 2 * col + 2] == b":": # Duvar kontrolu
                new_col = min(col + 1, self.num_columns - 1)
                if (new_row, new_col) in self.mach_locs:
                    new_row, new_col = row, col
                    reward = -5
        elif action == 3: # Sola
            if self.desc[1 + row, 2 * col] == b":":
                new_col = max(col - 1, 0)
                if (new_row, new_col) in self.mach_locs:
                    new_row, new_col = row, col
                    reward = -5
        elif action == 4: # Yukleme
            # Robot bir yuk noktasında mi
            if (row, col) in self.product_locs:
                prod_idx = self.product_locs.index((row, col))
                is_collected = (mask >> prod_idx) & 1 
                
                # Bu yuk daha once alınmamış ve robotun kapasitesi varsa yukle
                if not is_collected and load < self.max_capacity:
                    new_mask = mask | (1 << prod_idx)
                    new_load += 1
                    reward = 2 # Yukleme Başarılı Ödulu
                else:
                    reward = -10 # Hatalı Yukleme Cezası
            else:
                reward = -10 # Boş yerde yukleme cezası

        elif action == 5: # Boşaltma
            # Robot Bosaltma (B) noktasında
            if (row, col) == self.start_pos:
                if load > 0:
                    if load == 1:
                        reward = 5 # Tek yuk boşaltma odulu
                    else:
                        reward = 15 # Cift yuk boşaltma odulu
                    new_load = 0 # Yuku sıfırla
                    if new_mask == (1 << self.num_products) - 1: # Tum yukler toplandıysa
                        reward += 25
                        terminated = True # Oyun Bitti
                else:
                    reward = -10 # Yuk yokken boşaltmaya cezası
            else:
                reward = -10 # Yanlış yerde boşaltmaya cezası
        new_state = self.encode(new_row, new_col, new_mask, new_load)
        self.P[state][action].append((1.0, new_state, reward, terminated))

    # Verilen eylemi uygular
    def step(self, action):
        transitions = self.P[self.s][action]
        _, next_state, reward, terminated = transitions[0]
        self.s = next_state
        return next_state, reward, terminated, False, {}

    # Oyunu Sıfırlar 
    def reset(self):
        start_row, start_col = self.start_pos
        self.s = self.encode(start_row, start_col, 0, 0)
        return self.s, {}
    
    # Ortamı Gorsellestirir
    def render(self, step_num=0, total_reward=0, episode_num=1):
        robot_row, robot_col, mask, load = self.decode(self.s)
        grid_data = np.zeros((self.num_rows, self.num_columns))
        
        # Baslangic/Boşaltma Noktası (Gri)
        grid_data[self.start_pos] = 5

        # Makineler (Siyah)
        for machine in self.mach_locs:
            grid_data[machine] = 1
            
        # Yukler Maske bitlerine gore (Mavi)
        for i, pos in enumerate(self.product_locs):
            is_collected = (mask >> i) & 1
            if not is_collected:
                grid_data[pos] = 3

        # Robot (Sarı)
        grid_data[robot_row, robot_col] = 2

        self.ax.clear()
        self.ax.imshow(grid_data, cmap=self.cmap, norm=self.norm)
        
        # Izgaralar ve Duvarlar
        self.ax.grid(which='major', color='gray', linestyle='-', linewidth=0.5)
        self.ax.set_xticks(np.arange(-.5, self.num_columns, 1))
        self.ax.set_yticks(np.arange(-.5, self.num_rows, 1))
        self.ax.set_xticklabels([])
        self.ax.set_yticklabels([])
        
        for row in range(self.num_rows):
            for col in range(self.num_columns - 1):
                if self.desc[1 + row, 2 * col + 2] == b"|":
                    self.ax.vlines(x=col + 0.5, ymin=row - 0.5, ymax=row + 0.5, colors='black', linewidth=4)
        for spine in self.ax.spines.values():
            spine.set_edgecolor('black'); spine.set_linewidth(2)
        
        # Başlık Bilgisi
        collected_total = bin(mask).count('1')
        title_text = (f"Ep {episode_num} | Adım: {step_num} | Skor: {total_reward}\n"
                      f"Toplam Toplanan: {collected_total}/7 | Robot Yuku: {load}/{self.max_capacity}")
        self.ax.set_title(title_text, fontsize=11, fontweight='bold')
        
        plt.draw()
        plt.pause(0.001) # Daha akıcı olması icin sureyi azalttım

# Q-Learning Ajan Egitimi
def train_robot(env, episodes):
    # Parametreler
    alpha = 0.1 # Ogrenme Orani
    gamma = 0.99 # Gelecek odaklilik
    epsilon = 0.9 # Kesif Orani
    epsilon_decay = 0.03 # Epsilon azalma miktarı
    epsilon_min = 0.05 # Minimum Epsilon
    
    Q = np.zeros((env.num_states, env.action_space.n))
    print(f"Egitim Başladı... Toplam Durum Sayısı: {env.num_states}")
    reward_list = []
    step_list = []
    epsilon_list = []
    for ep in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        step_count = 0
        while not done:
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[state])
            next_state, reward, done, _, _ = env.step(action)
            # Eylem Sonucuna Göre Q-Learning Guncellemesi
            best_next_action = np.max(Q[next_state])
            Q[state, action] += alpha * (reward + gamma * best_next_action - Q[state, action])
            state = next_state
            total_reward += reward
            step_count += 1
        if (ep + 1) % 1000 == 0:
            reward_list.append(total_reward) # Toplam odul
            step_list.append(step_count) # Adım sayısı
        if (ep + 1) % 5000 == 0:
            epsilon_list.append(epsilon) # Epsilon degeri
            if epsilon > epsilon_min:
                epsilon -= epsilon_decay
            print(f"Egitim Ep {ep + 1}/{episodes} tamamlandı.")

    # Reward grafigi
    plt.plot(np.arange(len(reward_list))*1000, reward_list)
    plt.xlabel("Episode")
    plt.ylabel("Reward (1000 episode aralıklı)")
    plt.title("Training Reward")
    plt.savefig("Reward_Train.png", dpi=150)

    # Adım sayısı grafigi
    plt.plot(np.arange(len(step_list))*1000, step_list)
    plt.xlabel("Episode")
    plt.ylabel("Adım Sayısı (1000 episode aralıklı)")
    plt.title("Training Step Count")
    plt.savefig("Step_Count.png", dpi=150)
    
    # Epsilon grafigi
    plt.plot(np.arange(len(epsilon_list))*5000, epsilon_list)
    plt.xlabel("Episode")
    plt.ylabel("Epsilon Değeri (5000 episode aralıklı)")
    plt.title("Epsilon Azalma Grafiği")
    plt.savefig("Epsilon_Decay.png", dpi=150)
    return Q

# Robotun Test
def test_robot(env, Q, episodes=3):
    frames = []
    for ep in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        step = 0
        
        print(f"\nTEST EPISODE {ep + 1}")
        env.render(step_num=step, total_reward=total_reward, episode_num=ep+1)
        
        while not done:
            action = np.argmax(Q[state])
            next_state, reward, done, _, _ = env.step(action)
            total_reward += reward
            step += 1
            state = next_state
            
            env.render(step_num=step, total_reward=total_reward, episode_num=ep+1)
            
            # GIF Frame
            env.fig.canvas.draw()
            image_data = np.frombuffer(env.fig.canvas.tostring_rgb(), dtype='uint8')
            side_length = int(np.sqrt(image_data.size / 3))
            image_data = image_data.reshape((side_length, side_length, 3))
            frames.append(image_data)
        
        print(f"Toplam Ödul: {total_reward}")
    plt.close()
    # GIF Kaydet
    imageio.mimsave('robot.gif', frames, loop=0, fps=10)

# Calısma Ortaminin Yonetimi
if __name__ == "__main__":
    env = RobotEnv()
    q_table = train_robot(env, episodes=400000)

    test_robot(env, q_table, episodes=5)
