import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path
import numpy as np
try:  # Allow running as a module or as a script
    from .it2anfis import train_anfis, evalmyanfis
except ImportError:  # pragma: no cover - fallback when executed directly
    from it2anfis import train_anfis, evalmyanfis


def run_training(path_var, epoch_var, mf_var, step_var, dec_var, inc_var, b_var, result_label):
    """Run ANFIS training with the selected parameters."""
    data_path = Path(path_var.get())
    if not data_path.is_file():
        messagebox.showerror("Error", "Debe seleccionar un archivo de datos válido")
        return
    try:
        data = np.loadtxt(data_path)
        epoch_n = int(epoch_var.get())
        mf_n = int(mf_var.get())
        step_size = float(step_var.get())
        decrease_rate = float(dec_var.get())
        increase_rate = float(inc_var.get())
        B = float(b_var.get())
    except ValueError:
        messagebox.showerror("Error", "Parámetros numéricos inválidos")
        return

    it2anfis, y_anfis, RMSE = train_anfis(
        data, epoch_n, mf_n, step_size, decrease_rate, increase_rate, B
    )
    y_anfis = evalmyanfis(it2anfis, data[:, :-1])
    rmse = np.sqrt(np.sum((y_anfis[:, 0] - data[:, -1]) ** 2) / data.shape[0])
    result_label.config(text=f"RMSE: {rmse:.4f}")


def select_file(path_var):
    """Open a file dialog for selecting the training data file."""
    file_path = filedialog.askopenfilename(title="Seleccione archivo de datos")
    if file_path:
        path_var.set(file_path)


def main():
    root = tk.Tk()
    root.title("Entrenamiento IT2-ANFIS")

    path_var = tk.StringVar()
    epoch_var = tk.StringVar(value="100")
    mf_var = tk.StringVar(value="2")
    step_var = tk.StringVar(value="0.1")
    dec_var = tk.StringVar(value="0.5")
    inc_var = tk.StringVar(value="1.1")
    b_var = tk.StringVar(value="0.5")

    tk.Label(root, text="Archivo de datos:").grid(row=0, column=0, sticky="e")
    tk.Entry(root, textvariable=path_var, width=30).grid(row=0, column=1)
    tk.Button(root, text="Seleccionar", command=lambda: select_file(path_var)).grid(row=0, column=2)

    tk.Label(root, text="Épocas:").grid(row=1, column=0, sticky="e")
    tk.Entry(root, textvariable=epoch_var).grid(row=1, column=1)

    tk.Label(root, text="Funciones de membresía:").grid(row=2, column=0, sticky="e")
    tk.Entry(root, textvariable=mf_var).grid(row=2, column=1)

    tk.Label(root, text="Paso inicial:").grid(row=3, column=0, sticky="e")
    tk.Entry(root, textvariable=step_var).grid(row=3, column=1)

    tk.Label(root, text="Tasa de disminución:").grid(row=4, column=0, sticky="e")
    tk.Entry(root, textvariable=dec_var).grid(row=4, column=1)

    tk.Label(root, text="Tasa de incremento:").grid(row=5, column=0, sticky="e")
    tk.Entry(root, textvariable=inc_var).grid(row=5, column=1)

    tk.Label(root, text="Parámetro B:").grid(row=6, column=0, sticky="e")
    tk.Entry(root, textvariable=b_var).grid(row=6, column=1)

    result_label = tk.Label(root, text="")
    result_label.grid(row=8, column=0, columnspan=3)

    tk.Button(
        root,
        text="Entrenar",
        command=lambda: run_training(
            path_var, epoch_var, mf_var, step_var, dec_var, inc_var, b_var, result_label
        ),
    ).grid(row=7, column=0, columnspan=3, pady=5)

    root.mainloop()


if __name__ == "__main__":
    main()
