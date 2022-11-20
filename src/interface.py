from tkinter import filedialog
from tkinter import *
from tkinter import ttk
from PIL import ImageTk, Image
import time
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import main
import cv2
import os

# Window config
root = tk.Tk()
# root.configure(bg="#9acd32")
root.geometry("1100x600")
root.resizable(False,False)
root.title("Face Recognition App")

# Typography
font = ('inter', 18, 'bold')
rgFont = ('inter', 10)
frameStyle = ttk.Style()
frameStyle.configure('body.TFrame', background='#B6DFB5')
colorPallete = "#B6DFB5"
# Title Section
# Title Frame
titleFrame = ttk.Frame(root, width=100, height=50, style='body.TFrame', border=1)
titleFrame.pack(fill="both", expand=1)
# Title Contents
titleContent = ttk.Label(titleFrame, text="Face Recognition",font=('Inter 25 bold'), background=colorPallete)
titleContent.place(relx=0.5, rely=0.5, anchor=CENTER)


# Body Section
# Body Frame

bodyFrame = ttk.Frame(root, width=100, style='body.TFrame')
bodyFrame.pack(fill="both", expand=1)

# LHS Section
# Choose dataset label & button
btnImg1 = "./assets/datasetBtn.png"
datasetBtn = Image.open(btnImg1)
datasetBtn = datasetBtn.resize((152, 36))
datasetBtn = ImageTk.PhotoImage(datasetBtn)
chooseDatasetLabel = tk.Label(bodyFrame, text='Choose a dataset folder', width=25, font = font, background=colorPallete)
chooseDatasetLabel.grid(row=1,column=0, pady=10)
chooseDatasetBtn = tk.Button(bodyFrame, command = lambda:uploadDatasetFile(), width=170, height=36, image=datasetBtn, borderwidth=0, background=colorPallete)
chooseDatasetBtn.grid(row=2, column=0)
datasetStatusLabel = tk.Label(bodyFrame, text='No File Choosen', width=25, font = rgFont, background=colorPallete)
datasetStatusLabel.grid(row=3, column=0)

# Choose test image label & button
# Choose dataset label & button
btnImg2 = "./assets/testImgBtn.png"
testImgBtn = Image.open(btnImg2)
testImgBtn = testImgBtn.resize((152, 36))
testImgBtn = ImageTk.PhotoImage(testImgBtn)
chooseTestImgLabel = tk.Label(bodyFrame, text='Choose a test image', width=25, font= font, background=colorPallete)
chooseTestImgLabel.grid(row=4,column=0, pady=10)
chooseImgTestBtn = tk.Button(bodyFrame, command = lambda:uploadTestFile(), image=testImgBtn, width=170, height=36, borderwidth=0, background=colorPallete)
chooseImgTestBtn.grid(row=5, column=0)
testFaceStatusLabel = tk.Label(bodyFrame, text='No File Choosen', width=60, font = rgFont, background=colorPallete)
testFaceStatusLabel.grid(row=6, column=0)

# Result Prompt
resultLabelTitle = tk.Label(bodyFrame, text='Result', width=10, background=colorPallete)
resultLabelTitle.grid(row=7, column=0, pady=10)

resultLabelContent = tk.Label(bodyFrame, text='None', width=10, background=colorPallete)
resultLabelContent.grid(row=8, column=0)

# Camera Button (custom test face)
btnImg4 = "./assets/realTimeBtn.png"
runImgBtn2 = Image.open(btnImg4)
runImgBtn2 = runImgBtn2.resize((152, 36))
runImgBtn2 = ImageTk.PhotoImage(runImgBtn2)
runBtn = tk.Button(bodyFrame, text="Realtime test face!", command = lambda:captureTestFaceWithCam(), image=runImgBtn2, width=170, height=36, borderwidth=0, pady=10, background=colorPallete)
runBtn.grid(row = 9, column=1, columnspan=2)

# Run Button
btnImg3 = "./assets/startBtn.png"
runImgBtn = Image.open(btnImg3)
runImgBtn = runImgBtn.resize((152, 36))
runImgBtn = ImageTk.PhotoImage(runImgBtn)
runBtn = tk.Button(bodyFrame, command = lambda:startRecognize(), image=runImgBtn, width=170, height=36, borderwidth=0, pady=10, background=colorPallete)
runBtn.grid(row = 10, column=1, columnspan=2)

# RHS Section
# Test Image Section
# Test Image Label
testImgLabel = tk.Label(bodyFrame, text='Test Image', width=30, background=colorPallete)
testImgLabel.grid(row=1, column=1)
# Test Image Container & Image
imagePath = "./assets/baseImage.jfif"
originalImage = Image.open(imagePath)
resizedImage = originalImage.resize((256,256))
testImg = ImageTk.PhotoImage(resizedImage)
testImgContainer1 = tk.Button(bodyFrame, image=testImg, borderwidth=0)
testImgContainer1.grid(row=2, column=1, rowspan=5, padx=10)

# Result Image Label
resultImgLabel = tk.Label(bodyFrame, text='Closest Image', width=30, background=colorPallete)
resultImgLabel.grid(row=1, column=2)
# Result Image Container & Image
testImgContainer2 = tk.Button(bodyFrame, image=testImg, borderwidth=0)
testImgContainer2.grid(row=2, column=2, rowspan=5)

# Execution Time Prompt
currTime = 0
executionTimeLabel = tk.Label(bodyFrame, text="Execution Time: " + str(currTime) + " seconds", background=colorPallete)
executionTimeLabel.grid(row=7, column=1, pady=20)


# Callbacks
def uploadDatasetFile():
    global datasetDirectory
    datasetDirectory = filedialog.askdirectory()
    datasetDirectoryStr = datasetDirectory
    datasetDirectoryStr = os.path.basename(datasetDirectoryStr)
    datasetStatusLabel = tk.Label(bodyFrame, text = datasetDirectoryStr, width=60, font = rgFont, background=colorPallete)
    datasetStatusLabel.grid(row=3, column=0)

def uploadTestFile():
    global sampleDirectory, imgTest, testFaceStatusLabel
    sampleDirectory = filedialog.askopenfilename()
    fetchedImg = Image.open(sampleDirectory)
    resizeImg = fetchedImg.resize((256,256))
    imgTest = ImageTk.PhotoImage(resizeImg)
    testImgReplace = tk.Button(bodyFrame, image = imgTest, borderwidth=0)
    testImgReplace.grid(row=2, column=1, rowspan=5, padx=10)
    sampleDirectoryStr = os.path.basename(sampleDirectory)
    testFaceStatusLabel = tk.Label(bodyFrame, text = sampleDirectoryStr, width=60, font = rgFont, background=colorPallete)
    testFaceStatusLabel.grid(row=6, column=0)

def closeCam():
    global endCam, frame, countImage, camera
    endCam = True
    if (endCam):
        imageName = "../bin/sample_image_" + str(countImage) + ".jpg"
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        cv2.imwrite(imageName, frame)
        countImage += 1
        successMsg = tk.Label(bodyFrame, text="Closed camera! Please select the test face image!", width=40, background=colorPallete)
        successMsg.grid(row = 11, column=1, columnspan=10)
        camera.release()
        cv2.destroyAllWindows()
        runBtn = tk.Button(bodyFrame, command = lambda:captureTestFaceWithCam(), image=runImgBtn2, width=170, height=36, borderwidth=0, pady=10, background=colorPallete)
        runBtn.grid(row = 9, column=1, columnspan=2)

def captureTestFaceWithCam():
    global btnImg4, runImgBtn4
    btnImg4 = "./assets/capture&close.png"
    runImgBtn4 = Image.open(btnImg4)
    runImgBtn4 = runImgBtn4.resize((152, 36))
    runImgBtn4 = ImageTk.PhotoImage(runImgBtn4)
    runBtn = tk.Button(bodyFrame, command = lambda:closeCam(), image=runImgBtn4, width=170, height=36, borderwidth=0, pady=10, background=colorPallete)
    runBtn.grid(row = 9, column=1, columnspan=2)
    global endCam, frame, countImage, camera
    camera = cv2.VideoCapture(0)
    endCam = False
    countImage = 1
    captured = False
    while (not endCam):
        timelapse = 0
        while (timelapse < 200):
            # start cam
            global frame
            success, frame = camera.read()
            try:
                frame = cv2.resize(frame, (256, 256))
                frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                imageFromFrame = ImageTk.PhotoImage(Image.fromarray(frame))
            except:
                pass
            testImgReplace = tk.Label(bodyFrame, image = imageFromFrame)
            testImgReplace.grid(row=2, column=1, rowspan=5, padx=10)
            testImgReplace.configure(image=imageFromFrame)
            testImgReplace.image = imageFromFrame
            testImgReplace.update()
            timelapse += 1 # time limit

            if (timelapse <= 100 and captured):
                if (timelapse == 50):
                    captured = False
                    successMsg = tk.Label(bodyFrame, text="", width=20, background=colorPallete)
                    successMsg.grid(row = 11, column=1, columnspan=3)
                else:
                    successMsg = tk.Label(bodyFrame, text="Test face successfully taken!", width=40, background=colorPallete)
                    successMsg.grid(row = 11, column=1, columnspan=3)

        # auto capture
        if (timelapse == 200):
            imageName = "../bin/sample_image_" + str(countImage) + ".jpg"
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            cv2.imwrite(imageName, frame)
            captured = True
            countImage += 1

def startRecognize():
    global resultImage, currTime, startTime
    startTime = time.time()
    finalResultPath, matchPercentage = main.run(datasetDirectory, sampleDirectory)
    currTime = round(time.time() - startTime, 3)
    executionTimeLabel = tk.Label(bodyFrame, text="Execution Time: " + str(currTime) + " seconds", background=colorPallete)
    executionTimeLabel.grid(row=7, column=1, pady=20)
    filename = finalResultPath
    fetchedImg = Image.open(filename)
    resizeImg = fetchedImg.resize((256,256))
    resultImage = ImageTk.PhotoImage(resizeImg)
    resultReplace = tk.Button(bodyFrame, image = resultImage, borderwidth=0)
    resultReplace.grid(row=2, column=2, rowspan=5, padx=10)
    filenameStr = os.path.basename(filename)
    testFaceStatusLabel = tk.Label(bodyFrame, text = filenameStr, width=60, font = rgFont, background=colorPallete)
    testFaceStatusLabel.grid(row=6, column=0)
    if (matchPercentage == 0):
        stringResult = "No match!"
    else:
        stringResult = f"Accuration Percentage : %.2f" % (matchPercentage)
    resultLabelContent = tk.Label(bodyFrame, text=stringResult, width=30, background=colorPallete)
    resultLabelContent.grid(row=8, column=0)


root.mainloop()