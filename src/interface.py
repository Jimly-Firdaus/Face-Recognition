from tkinter import filedialog
from tkinter import *
from tkinter import ttk
from PIL import ImageTk, Image
import os
import tkinter as tk
from tkinter import filedialog
from tkinter.filedialog import askopenfile
from PIL import Image, ImageTk
import main

# Window config
root = tk.Tk()
# root.configure(bg="#9acd32")
root.geometry("1024x600")
root.resizable(False,False)
root.title("Face Recognition App")

# Typography
font = ('times', 18, 'bold')
rgFont = ('times', 10)
# Title Section
# Title Frame
titleFrame = ttk.Frame(root, width=100)
titleFrame.pack()
# Title Contents
titleContent = ttk.Label(titleFrame, text="Face Recognition", font=('Times 25 bold'))
titleContent.grid(row=1, column=0)


# Body Section
# Body Frame
bodyFrame = ttk.Frame(root, width=100)
bodyFrame.pack(pady=30)

# LHS Section
# Choose dataset label & button
chooseDatasetLabel = tk.Label(bodyFrame, text='Choose a dataset folder', width=25, font = font)
chooseDatasetLabel.grid(row=1,column=0, pady=10)
chooseDatasetBtn = tk.Button(bodyFrame, text="Browse", width=20, command = lambda:uploadDatasetFile())
chooseDatasetBtn.grid(row=2, column=0)
datasetStatusLabel = tk.Label(bodyFrame, text='No File Choosen', width=25, font = rgFont)
datasetStatusLabel.grid(row=3, column=0)

# Choose test image label & button
# Choose dataset label & button
chooseTestImgLabel = tk.Label(bodyFrame, text='Choose a test image', width=25, font= font)
chooseTestImgLabel.grid(row=4,column=0, pady=10)
chooseImgTestBtn = tk.Button(bodyFrame, text="Browse", width=20, command = lambda:uploadTestFile())
chooseImgTestBtn.grid(row=5, column=0)
testFaceStatusLabel = tk.Label(bodyFrame, text='No File Choosen', width=60, font = rgFont)
testFaceStatusLabel.grid(row=6, column=0)

# Result Prompt
resultLabelTitle = tk.Label(bodyFrame, text='Result', width=10)
resultLabelTitle.grid(row=7, column=0, pady=10)

resultLabelContent = tk.Label(bodyFrame, text='None', width=10)
resultLabelContent.grid(row=8, column=0)

# Run Button
runBtn = tk.Button(bodyFrame, text="Start Recognition!", width=20, command = lambda:startRecognize())
runBtn.grid(row = 9, column=1, columnspan=2)

# RHS Section
# Test Image Section
# Test Image Label
testImgLabel = tk.Label(bodyFrame, text='Test Image', width=30)
testImgLabel.grid(row=1, column=1)
# Test Image Container & Image
imagePath = "assets/baseImage.jfif"
originalImage = Image.open(imagePath)
resizedImage = originalImage.resize((256,256))
testImg = ImageTk.PhotoImage(resizedImage)
testImgContainer1 = tk.Button(bodyFrame, image=testImg)
testImgContainer1.grid(row=2, column=1, rowspan=5, padx=10)

# Result Image Label
resultImgLabel = tk.Label(bodyFrame, text='Closest Image', width=30)
resultImgLabel.grid(row=1, column=2)
# Result Image Container & Image
testImgContainer2 = tk.Button(bodyFrame, image=testImg)
testImgContainer2.grid(row=2, column=2, rowspan=5)

# Execution Time Prompt
currTime = '00:00'
executionTimeLabel = tk.Label(bodyFrame, text='Execution Time: ' + currTime)
executionTimeLabel.grid(row=7, column=1, pady=20)

# executionTimeContainer = tk.Label(bodyFrame, text='00:00')
# executionTimeContainer.grid(row=6, column=1)


# Callbacks
def uploadDatasetFile():
    global datasetDirectory
    datasetDirectory = filedialog.askdirectory()
    datasetStatusLabel = tk.Label(bodyFrame, text = datasetDirectory, width=60, font = rgFont)
    datasetStatusLabel.grid(row=3, column=0)


def uploadTestFile():
    global sampleDirectory, imgTest
    # f_types = [('Jpg Files', '*.jpg','Png Files', '*.png', 'Jpeg Files', '*.jpeg')]
    sampleDirectory = filedialog.askopenfilename()
    testFaceStatusLabel = tk.Label(bodyFrame, text = sampleDirectory, width=60, font = rgFont)
    testFaceStatusLabel.grid(row=6, column=0)
    fetchedImg = Image.open(sampleDirectory)
    resizeImg = fetchedImg.resize((256,256))
    imgTest = ImageTk.PhotoImage(resizeImg)
    testImgReplace = tk.Button(bodyFrame, image = imgTest)
    testImgReplace.grid(row=2, column=1, rowspan=5, padx=10)

# def uploadTestFile():
    # global imgTest
    # f_types = [('Jpg Files', '*.jpg', '*.png', '*.jpeg')]
    # filename = filedialog.askopenfilename(filetypes=f_types)
    # fetchedImg = Image.open(filename)
    # resizeImg = fetchedImg.resize((256,256))
    # imgTest = ImageTk.PhotoImage(resizeImg)
    
    # b2 =tk.Button(root,image=imgTest) # using Button 
    # b2.grid(row=3,column=1)

def startRecognize():
    global resultImage
    finalResultIndex, matchPercentage = main.run(datasetDirectory, sampleDirectory)
    files = [os.path.join(datasetDirectory, p) for p in os.listdir(datasetDirectory)]
    filename = files[finalResultIndex]
    fetchedImg = Image.open(filename)
    resizeImg = fetchedImg.resize((256,256))
    resultImage = ImageTk.PhotoImage(resizeImg)
    resultReplace = tk.Button(bodyFrame, image = resultImage)
    resultReplace.grid(row=2, column=2, rowspan=5, padx=10)
    testFaceStatusLabel = tk.Label(bodyFrame, text = filename, width=60, font = rgFont)
    testFaceStatusLabel.grid(row=6, column=0)
    stringResult = f"Accuration Percentage : %.2f" % (matchPercentage)
    resultLabelContent = tk.Label(bodyFrame, text=stringResult, width=30)
    resultLabelContent.grid(row=8, column=0)


root.mainloop()