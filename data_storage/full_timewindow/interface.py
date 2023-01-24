import tkinter as tk
import time
from tkinter import messagebox


if __name__ == '__main__':

    root = tk.Tk()
    root.title("Guide")
    root.geometry("1920x1080")

    label = tk.Label(root, text="Please, perform a continuous: ", font=("Arial", 25), pady=50)
    label.pack()
    label = tk.Label(root, text="TEST", font=("Arial", 80), fg="darkblue")
    label.pack()

    label = tk.Label(root, text=" ", font=("Arial", 25), pady=100)
    label.pack()

    label = tk.Label(root, text="Next interaction in: ", font=("Arial", 25), pady=50)
    label.pack()

    temp = 10
    str_temp = tk.StringVar()
    primitive_timer = tk.Label(root, textvariable=str_temp, font=("Arial", 80), fg="darkgreen")

    # experiment_timer = tk.Label(root, textvariable=str_temp, font=("Arial", 80), pady=50)

    while temp > -1:
        str_temp.set(str(temp))

        if temp < 6:
            primitive_timer.config(fg="red")

        primitive_timer.pack()
        root.update()
        time.sleep(1)

        if temp == 0:
            messagebox.showinfo("Time Countdown", "Time's up ")
        temp -= 1

    root.mainloop()
