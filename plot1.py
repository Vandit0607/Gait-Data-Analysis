#!/usr/bin/python3
#Aviral Upadhyay
#Vandit Maheshwari
#Version 1.0
#Date May 7th, 2020



import pandas as pd
import matplotlib .pyplot as plt



def plot():
	df = pd.read_csv("CSV/GaCo01_01.csv")
	df2 = pd.read_csv("CSV/GaPt03_01.csv")
	df = df[0:2000]
	df2 = df2[0:2000]

	plt.plot(df["Time(sec)"],df["Total_force_left"],label = "Control Group")
	plt.xlabel("Time in sec")
	plt.ylabel("Total force left foot")
	plt.plot(df2["Time(sec)"],df2["Total_force_left"],label = "Patient Group")
	plt.legend()
	plt.title("Left Foot Force Control Group V Patient")
	plt.show()


	plt.plot(df["Time(sec)"],df["Total_force_right"],label = "Control Group")
	plt.xlabel("Time in sec")
	plt.ylabel("Total force right foot")
	plt.plot(df2["Time(sec)"],df2["Total_force_right"],label = "Patient Group")
	plt.legend()
	plt.title("right Foot Force Control Group V Patient")
	plt.show()

	"""
	for i in range(1,9):
		xx = "VGRF_left_s" + str(i)
		plt.plot(df["Time(sec)"],df[xx])


	plt.xlabel("Time in sec")
	plt.ylabel("Forces-Left")
	plt.title("Left Foor Control Group")
	plt.show()

	for i in range(1,9):
		xx = "VGRF_right_s" + str(i)
		plt.plot(df["Time(sec)"],df[xx])


	plt.xlabel("Time in sec")
	plt.ylabel("Forces-Right")
	plt.title("Right Foot Control Group")
	plt.show()


	for i in range(1,9):
		xx = "VGRF_left_s" + str(i)
		plt.plot(df2["Time(sec)"],df2[xx])


	plt.xlabel("Time in sec")
	plt.ylabel("Forces-Left")
	plt.title("Left Foor Patient Group")
	plt.show()

	for i in range(1,9):
		xx = "VGRF_right_s" + str(i)
		plt.plot(df2["Time(sec)"],df2[xx])


	plt.xlabel("Time in sec")
	plt.ylabel("Forces-Right")
	plt.title("Right Foot Patient Group")
	plt.show()"""