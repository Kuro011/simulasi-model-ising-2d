import numpy as np
import matplotlib.pyplot as plt
import random
import os

# Mengatur seed agar hasil simulasi dapat direproduksi dengan sama
random.seed(42)
np.random.seed(42)

def metropolis_step(grid, T):
    """
    Melakukan satu kali uji coba pembalikan spin berdasarkan kriteria Metropolis.
    Menggunakan kondisi batas periodik dengan operasi modulo (%).
    """
    N = grid.shape[0]
    # Pilih indeks kisi (koordinat x, y) secara acak
    x, y = random.randint(0, N-1), random.randint(0, N-1)
    
    # Hitung jumlah dari 4 spin tetangga terdekat (Atas, Bawah, Kanan, Kiri)
    # Operasi % N memastikan jika koordinat berada di ujung kisi, ia akan berputar ke ujung satunya
    s_neighbors = (
        grid[(x + 1) % N, y] +
        grid[(x - 1) % N, y] +
        grid[x, (y + 1) % N] +
        grid[x, (y - 1) % N]
    )
    
    # Hitung perubahan energi lokal (delta E) secara efisien jika spin di (x,y) dibalik
    delta_E = 2 * grid[x, y] * s_neighbors
    
    # Aturan penerimaan Metropolis (Metropolis Criterion)
    if delta_E <= 0 or random.random() < np.exp(-delta_E / T):
        grid[x, y] *= -1  # Diterima: Balik arah spin (+1 jadi -1, atau sebaliknya)
        
    return grid

def run_simulation(N=20, temp=1.0, n_steps=200000):
    """
    Menjalankan simulasi Model Ising 2D dimulai dari konfigurasi acak (Hot Start).
    """
    print(f"Memulai simulasi Model Ising 2D ({N}x{N}) pada T = {temp} untuk {n_steps} langkah...")
    
    # Inisialisasi Hot Start: Kisi spin acak bernilai +1 atau -1
    grid = np.random.choice([-1, 1], size=(N, N))
    
    magnetization_history = []
    
    # Loop utama untuk melangkah secara Monte Carlo
    for step in range(n_steps):
        grid = metropolis_step(grid, temp)
        
        # Hitung dan catat nilai rata-rata magnetisasi setiap 100 langkah
        if step % 100 == 0:
            magnetization = np.mean(grid)
            magnetization_history.append(magnetization)
            
    print("Simulasi selesai!")
    return grid, magnetization_history

if __name__ == "__main__":
    # 1. Parameter Konfigurasi Simulasi
    GRID_SIZE = 20          # Ukuran kisi N x N
    TEMPERATURE = 1.0       # Suhu Rendah (Fase Feromagnetik)
    MONTE_CARLO_STEPS = 200000  # Minimal 200.000 langkah
    
    # 2. Menjalankan Fungsi Simulasi
    final_grid, M_history = run_simulation(N=GRID_SIZE, temp=TEMPERATURE, n_steps=MONTE_CARLO_STEPS)
    
    # 3. Visualisasi Hasil Menggunakan Matplotlib Berdampingan
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot Kiri: Keadaan Akhir Kisi Spin (Menggunakan peta warna biner)
    axes[0].imshow(final_grid, cmap='binary', vmin=-1, vmax=1)
    axes[0].set_title(f"Keadaan Akhir Kisi Spin (T = {TEMPERATURE:.2f})")
    axes[0].axis('off')  # Menghilangkan penomoran sumbu koordinat agar fokus pada visual kisi
    
    # Plot Kanan: Kurva Riwayat Magnetisasi Rata-rata sepanjang waktu simulasi
    axes[1].plot(M_history, color='blue', linewidth=1.5, label=r'$\langle M \rangle$')
    axes[1].set_title(f"Riwayat Magnetisasi Rata-rata (T = {TEMPERATURE:.2f})")
    axes[1].set_xlabel("Langkah Monte Carlo (x100)")
    axes[1].set_ylabel("Rata-rata Magnetisasi <M>")
    axes[1].set_ylim(-1.1, 1.1)
    axes[1].grid(True, linestyle='--', alpha=0.6)
    axes[1].legend()
    
    plt.tight_layout()
    
    # 4. Membuat folder docs jika belum ada untuk menyimpan hasil grafik otomatis
    os.makedirs('docs', exist_ok=True)
    graph_path = 'docs/hasil_simulasi_kasus1.png'
    plt.savefig(graph_path, dpi=300)
    print(f"Grafik hasil simulasi berhasil disimpan di: {graph_path}")
    
    # Menampilkan plot di layar
    plt.show()