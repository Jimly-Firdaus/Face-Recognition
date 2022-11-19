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

# Camera Button (custom test face)
runBtn = tk.Button(bodyFrame, text="Realtime test face!", width=20, command = lambda:captureTestFaceWithCam())
runBtn.grid(row = 9, column=1, columnspan=2)

# Run Button
runBtn = tk.Button(bodyFrame, text="Start Recognition!", width=20, command = lambda:startRecognize())
runBtn.grid(row = 10, column=1, columnspan=2)

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
currTime = 0
executionTimeLabel = tk.Label(bodyFrame, text="Execution Time: " + str(currTime) + " seconds")
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
    sampleDirectory = filedialog.askopenfilename()
    testFaceStatusLabel = tk.Label(bodyFrame, text = sampleDirectory, width=60, font = rgFont)
    testFaceStatusLabel.grid(row=6, column=0)
    fetchedImg = Image.open(sampleDirectory)
    resizeImg = fetchedImg.resize((256,256))
    imgTest = ImageTk.PhotoImage(resizeImg)
    testImgReplace = tk.Button(bodyFrame, image = imgTest)
    testImgReplace.grid(row=2, column=1, rowspan=5, padx=10)


def closeCam():
    global endCam, frame, countImage, camera
    endCam = True
    if (endCam):
        imageName = "./sample/sample_image_" + str(countImage) + ".jpg"
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        cv2.imwrite(imageName, frame)
        countImage += 1
        successMsg = tk.Label(bodyFrame, text="Closed camera! Please select the test face image!", width=40)
        successMsg.grid(row = 11, column=1, columnspan=10)
        camera.release()
        cv2.destroyAllWindows()
        runBtn = tk.Button(bodyFrame, text="Realtime test face!", width=20, command = lambda:captureTestFaceWithCam())
        runBtn.grid(row = 9, column=1, columnspan=2)

def captureTestFaceWithCam():
    runBtn = tk.Button(bodyFrame, text="Capture & close camera!", width=20, command = lambda:closeCam())
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
            frame = cv2.resize(frame, (256, 256))
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            imageFromFrame = ImageTk.PhotoImage(Image.fromarray(frame))
            testImgReplace = tk.Label(bodyFrame, image = imageFromFrame)
            testImgReplace.grid(row=2, column=1, rowspan=5, padx=10)
            testImgReplace.configure(image=imageFromFrame)
            testImgReplace.image = imageFromFrame
            testImgReplace.update()
            timelapse += 1 # time limit

            if (timelapse <= 100 and captured):
                if (timelapse == 50):
                    captured = False
                    successMsg = tk.Label(bodyFrame, text="", width=20)
                    successMsg.grid(row = 11, column=1, columnspan=3)
                else:
                    successMsg = tk.Label(bodyFrame, text="Test face successfully taken!", width=40)
                    successMsg.grid(row = 11, column=1, columnspan=3)

        # auto capture
        if (timelapse == 200):
            imageName = "./sample/sample_image_" + str(countImage) + ".jpg"
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            cv2.imwrite(imageName, frame)
            captured = True
            countImage += 1

def startRecognize():
    global resultImage, currTime, startTime
    startTime = time.time()
    finalResultPath, matchPercentage = main.run(datasetDirectory, sampleDirectory)
    currTime = round(time.time() - startTime, 3)
    executionTimeLabel = tk.Label(bodyFrame, text="Execution Time: " + str(currTime) + " seconds")
    executionTimeLabel.grid(row=7, column=1, pady=20)
    filename = finalResultPath
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