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

    # primitives = ["PULL", "PUSH", "SHAKE", "TWIST"]
    primitives = ["PUXAR", "EMPURRAR", "ABANAR", "TORCER"]

    xtime = 60

    mixer.init()
    sound = mixer.Sound(ROOT_DIR + "/data_storage/full_timewindow/beep-07a.wav")
    # sound = mixer.Sound("beep-06.wav")

    while True:
        times = np.random.choice(range(5, 12), size=random.randint(5, 12), replace=True)
        # times = np.random.choice(range(5, 15), size=random.randint(4, 12), replace=True)
        if sum(times) == xtime:
            break

    experiment = []
    Stamp = namedtuple("Stamp", "time primitive")

    for t in times:
        experiment.append(Stamp(t, random.choice(primitives)))

    print("experiment")
    print(experiment)

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

    pub.publish("END")
    root.mainloop()
