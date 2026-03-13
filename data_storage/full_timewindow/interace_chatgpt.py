# O problema esta na publicaacao da mensagem "END" que so acontece depois de clicar ok!
# Este script do cahtgpt tenta corrigir isso. o script "interface.py" esta quase perfeito, mas ainda precisa de melhorar muita coisa.
# Este scritp do chat vai ajudar.



#!/usr/bin/env python3

import tkinter as tk
from tkinter import messagebox
from datetime import timedelta
from collections import namedtuple
import random
import rospy
from std_msgs.msg import String
from pygame import mixer
from config.definitions import ROOT_DIR


primitives = ["PUXAR", "EMPURRAR", "ABANAR", "TORCER"]
xtime = 15
min_t = 2
max_t = 6

Stamp = namedtuple("Stamp", "time primitive")


def generate_times(total, min_t, max_t):

    min_n = (total + max_t - 1) // max_t
    max_n = total // min_t

    while True:

        n = random.randint(min_n, max_n)
        times = [min_t] * n
        remaining = total - n * min_t

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


if __name__ == "__main__":

    rospy.init_node("full_timewindow_interface", anonymous=True)

    root = tk.Tk()
    root.title("Guide")
    root.geometry("1920x1080")

    mixer.init()
    sound = mixer.Sound(ROOT_DIR + "/data_storage/full_timewindow/beep-07a.wav")

    # -------- gerar experiência --------

    times = generate_times(xtime, min_t, max_t)
    prims = generate_primitives(len(times), primitives)

    experiment = [Stamp(t, p) for t, p in zip(times, prims)]

    print("experiment")
    print(experiment)

    # -------- GUI --------

    label = tk.Label(root, text="Please, perform a continuous:", font=("Arial", 25), pady=30)
    label.pack()

    str_primitive = tk.StringVar()
    label_str = tk.Label(root, textvariable=str_primitive, font=("Arial", 100), fg="darkblue")
    label_str.pack()

    label = tk.Label(root, text="Next interaction in:", font=("Arial", 25), pady=30)
    label.pack()

    str_temp = tk.StringVar()
    primitive_timer = tk.Label(root, textvariable=str_temp, font=("Arial", 80), fg="darkgreen")
    primitive_timer.pack()

    label = tk.Label(root, text="Experiment will end in:", font=("Arial", 25), pady=10)
    label.pack()

    str_time = tk.StringVar()
    experiment_timer = tk.Label(root, textvariable=str_time, font=("Arial", 25))
    experiment_timer.pack()

    # -------- ROS --------

    pub = rospy.Publisher("ground_truth", String, queue_size=10)

    rate = rospy.Rate(100)

    pub.publish("START")

    start_time = rospy.Time.now().to_sec()

    primitive_index = 0
    primitive_start = start_time

    last_beep_second = None

    while not rospy.is_shutdown():

        now = rospy.Time.now().to_sec()
        elapsed = now - start_time

        remaining_total = xtime - elapsed

        if remaining_total <= 0:
            break

        current = experiment[primitive_index]

        elapsed_primitive = now - primitive_start
        remaining_primitive = current.time - elapsed_primitive

        if remaining_primitive <= 0:
            primitive_index += 1

            if primitive_index >= len(experiment):
                break

            primitive_start = now
            current = experiment[primitive_index]
            remaining_primitive = current.time

        # -------- publicar ação --------

        pub.publish(current.primitive)

        # -------- atualizar GUI --------

        str_primitive.set(current.primitive)

        str_temp.set(str(timedelta(seconds=int(remaining_primitive))))
        str_time.set(str(timedelta(seconds=int(remaining_total))))

        if remaining_primitive < 3:

            sec = int(remaining_primitive)

            if sec != last_beep_second:
                sound.play()
                last_beep_second = sec

            primitive_timer.config(fg="red")

            if primitive_index < len(experiment) - 1:
                next_prim = experiment[primitive_index + 1].primitive
                str_primitive.set(current.primitive + "  =>  " + next_prim)

        else:
            primitive_timer.config(fg="darkgreen")
            last_beep_second = None

        root.update()

        rate.sleep()

    # -------- fim --------

    pub.publish("END")

    messagebox.showinfo("Experiment Ended", "We got everything we need :)\nThank you!")

    root.destroy()