#!/usr/bin/env python3

import tkinter as tk
import time
from tkinter import messagebox
from datetime import timedelta
from collections import namedtuple
import numpy as np
import random
import rospy
from std_msgs.msg import String
from pygame import mixer
from config.definitions import ROOT_DIR


if __name__ == '__main__':

    root = tk.Tk()
    root.title("Guide")
    root.geometry("1920x1080")
    mixer.init()
    sound = mixer.Sound(ROOT_DIR + "/data_storage/full_timewindow/beep-07a.wav")

    primitives = ["PUXAR", "EMPURRAR", "ABANAR", "TORCER"]
    xtime = 15
    min_t = 2
    max_t = 6

    Stamp = namedtuple("Stamp", "time primitive")

    def generate_times(total, min_t, max_t):
        # número possível de ações
        min_n = (total + max_t - 1) // max_t
        max_n = total // min_t
        while True:
            n = random.randint(min_n, max_n)
            # começa com tempo mínimo
            times = [min_t] * n
            remaining = total - n * min_t
            # distribui o tempo restante
            while remaining > 0:
                i = random.randint(0, n - 1)
                if times[i] < max_t:
                    times[i] += 1
                    remaining -= 1
            if sum(times) == total:
                random.shuffle(times)
                return times

    def generate_primitives(n, primitives):
        seq = []
        last = None
        for _ in range(n):
            choices = [p for p in primitives if p != last]
            p = random.choice(choices)
            seq.append(p)
            last = p
        return seq

    times = generate_times(xtime, min_t, max_t)
    prims = generate_primitives(len(times), primitives)

    experiment = [Stamp(t, p) for t, p in zip(times, prims)]

    print("experiment")
    print(experiment)
    # exit(0)

    label = tk.Label(root, text="Please, perform a continuous: ", font=("Arial", 25), pady=30)
    label.pack()

    str_primitive = tk.StringVar()

    label_str = tk.Label(root, textvariable=str_primitive, font=("Arial", 100), fg="darkblue")
    label_str.pack()

    label = tk.Label(root, text=" ", font=("Arial", 25), pady=60)
    label.pack()

    label = tk.Label(root, text="Next interaction in: ", font=("Arial", 25), pady=30)
    label.pack()

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

    pub = rospy.Publisher('ground_truth', String, queue_size=10)
    rospy.init_node('full_timewindow_interface', anonymous=True)
    rate = rospy.Rate(100)  # 100hz
    pub.publish("START")

    for i, stamp in enumerate(experiment):
        label_str.config(font=("Arial", 100))
        str_primitive.set(stamp.primitive)
        temp = int(stamp.time)

        while True:
            if temp <= 0:
                if xtime <= 0:
                    pub.publish("END")
                    messagebox.showinfo("Experiment Ended", "We got everything we need :)\nThank you!")
                    root.destroy()
                break

            rate.sleep()
            temp -= 0.01
            xtime -= 0.01
            message = str(stamp.primitive)
            if temp >= 0:
                    pub.publish(message)

            str_temp.set(str(timedelta(seconds=int(temp))))
            str_time.set(str(timedelta(seconds=int(xtime))))

            if int(temp) < 3:
                if (temp - int(temp)) < 0.1:
                    sound.play()
                if i < len(experiment) - 1:
                    str_primitive.set(stamp.primitive + "     =>     " + experiment[i+1].primitive)
                    label_str.config(font=("Arial", 80))

                primitive_timer.config(fg="red")
            else:
                primitive_timer.config(fg="darkgreen")

            root.update()

    # pub.publish("END")
    root.mainloop()
