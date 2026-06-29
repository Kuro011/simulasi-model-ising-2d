import numpy as np
import matplotlib.pyplot as plt
import random
import os

# Mengatur seed agar hasil dapat direproduksi
random.seed(42)
np.random.seed(42)

def metropolis_step(grid, T):
    N = grid.shape[0]
    x, y = random.randint(0, N-1), random.randint(0, N-1)
    
    s_neighbors = (
        grid[(x + 1) % N, y] +
        grid[(x - 1) % N, y] +
        grid[x, (y + 1) % N] +
        grid[x, (y - 1) % N]
    )
    
    delta_E = 2 * grid[x, y] * s_neighbors
    
    if delta_E <= 0 or random.random() < np.exp(-delta_E / T):
        grid[x, y] *= -1
        
    return grid

def run_simulation(N=20, temp=1.0, n_steps=200000):
    grid = np.random.choice([-1, 1], size=(N, N))
    magnetization_history = []
    
    for step in range(n_steps):
        grid = metropolis_step(grid, temp)
        if step % 100 == 0:
            magnetization = np.mean(grid)
            magnetization_history.append(magnetization)
            
    return grid, magnetization_history

if __name__ == "__main__":
    GRID_SIZE = 20
    MONTE_CARLO_STEPS = 200000
    # Tiga studi kasus sesuai instruksi modul
    temperatures = [1.0, 2.27, 4.0]
    
    # Membuat plot ukuran besar (2 baris, 3 kolom)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    for i, T in enumerate(temperatures):
        print(f"Menjalankan simulasi untuk T = {T}...")
        final_grid, M_history = run_simulation(N=GRID_SIZE, temp=T, n_steps=MONTE_CARLO_STEPS)
        
        # Baris 1: Visualisasi Kisi 2D
        axes[0, i].imshow(final_grid, cmap='binary', vmin=-1, vmax=1)
        axes[0, i].set_title(f"Kisi Akhir (T = {T})")
        axes[0, i].axis('off')
        
        # Baris 2: Grafik Magnetisasi
        axes[1, i].plot(M_history, color='blue', linewidth=1)
        axes[1, i].set_title(f"Magnetisasi (T = {T})")
        axes[1, i].set_xlabel("Langkah Monte Carlo (x100)")
        axes[1, i].set_ylabel("<M>")
        axes[1, i].set_ylim(-1.1, 1.1)
        axes[1, i].grid(True, linestyle='--', alpha=0.5)
        
    plt.tight_layout()
    
    # Simpan hasil gabungan ke folder docs
    os.makedirs('docs', exist_ok=True)
    plt.savefig('docs/hasil_simulasi_all_cases.png', dpi=300)
    print("Simulasi semua kasus selesai dan gambar berhasil disimpan!")
    plt.show()