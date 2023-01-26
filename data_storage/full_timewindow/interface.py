import tkinter as tk
import time
from tkinter import messagebox
from datetime import timedelta
from collections import namedtuple
import numpy as np
import random


if __name__ == '__main__':

    root = tk.Tk()
    root.title("Guide")
    root.geometry("1920x1080")

    primitives = ["PULL", "PUSH", "SHAKE", "TWIST"]

    xtime = 60

    while True:
        times = np.random.choice(range(3, 10), size=random.randint(6, 20), replace=True)
        if sum(times) == xtime:
            break

    experiment = []
    Test = namedtuple("time", "primitive")
    for t in times:
        experiment.append(Test(t, random.choice(primitives)))

    print(experiment)
    exit(0)

    label = tk.Label(root, text="Please, perform a continuous: ", font=("Arial", 25), pady=30)
    label.pack()

    str_primitive = tk.StringVar()
    str_primitive.set(random.choice(primitives))
    label = tk.Label(root, textvariable=str_primitive, font=("Arial", 100), fg="darkblue")
    label.pack()

    label = tk.Label(root, text=" ", font=("Arial", 25), pady=60)
    label.pack()

    label = tk.Label(root, text="Next interaction in: ", font=("Arial", 25), pady=30)
    label.pack()

    temp = 10
    str_temp = tk.StringVar()
    primitive_timer = tk.Label(root, textvariable=str_temp, font=("Arial", 80), fg="darkgreen")
    primitive_timer.pack()

    label = tk.Label(root, text=" ", font=("Arial", 25), pady=60)
    label.pack()

    label = tk.Label(root, text="Experiment will end in : ", font=("Arial", 25), pady=10)
    label.pack()

    str_time = tk.StringVar()

    experiment_timer = tk.Label(root, textvariable=str_time, font=("Arial", 25), fg="black")
    experiment_timer.pack()

    while temp > -1:
        str_temp.set(str(timedelta(seconds=temp)))
        str_time.set(str(timedelta(seconds=xtime)))

        if temp < 6:
            primitive_timer.config(fg="red")
        else:
            primitive_timer.config(fg="darkgreen")

        root.update()
        time.sleep(1)

        if temp == 0:
            temp = 10
            str_primitive.set(random.choice(primitives))

        if xtime == 0:
            messagebox.showinfo("Experiment Ended", "We got everything we need :)\nThank you!")

            root.destroy()

        temp -= 1
        xtime -= 1

    root.mainloop()
