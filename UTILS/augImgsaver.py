from keras.preprocessing.image import ImageDataGenerator
from build.randomNamemaker import randomName
from tkinter import filedialog, messagebox
from keras import preprocessing as pre
from imutils.paths import list_images
from PIL import Image, ImageTk
import tkinter.ttk as ttk
import tkinter as tk
import numpy as np
import pickle
import time
import PIL
import cv2
import os

class canvas:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title('augGo2')
        self.window.geometry('900x400+70+70')

        title = tk.Label(self.window, text = 'go2 with data augmentation')
        image = tk.Label(self.window, text = 'applied image')

        image.place(y = 20, x = 650)
        title.place(y = 10, x = 150)

        self.genBtn = tk.Button(self.window, text = 'generate', command = self.generate)
        self.exitBtn = tk.Button(self.window, text = 'exit', command = self.EXIT)
        self.loadBtn = tk.Button(self.window, text = 'load files', command = self.loadImgs)
        self.applyBtn = tk.Button(self.window, text = 'apply', command = self.apply)

        self.loadBtn.place(y = 160, x = 20, width = 100)
        self.applyBtn.place(y = 160 ,x = 120, width = 100)
        self.genBtn.place(y = 160, x = 220, width = 100)
        self.exitBtn.place(y = 160, x = 320, width = 100)

        self.frame = tk.Frame(self.window, width = 270, height = 160,  relief = 'solid', bd = 2)
        self.imgFrame = tk.Frame(self.window, width = 420, height = 350,  relief = 'solid', bd = 2)
        self.frame.place(y = 190, x = 10)
        self.imgFrame.place(y = 40, x = 460)

        self.fileScrollH = tk.Scrollbar(self.frame)
        self.fileScrollH.pack(side = 'right', fill = 'y')
        self.fileListbox = tk.Listbox(self.frame, selectmode = 'single', width = 270, height = 160, yscrollcommand = self.fileScrollH.set)
        self.fileListbox.place(y = 190, x = 10)

        self.horizonCheck = tk.BooleanVar()
        self.verticalCheck = tk.BooleanVar()

        self.horizonCheck.set(False)
        self.verticalCheck.set(False)

        #* horizonCheck.get -> horizontalCheck flag recieve 

        rotText = tk.StringVar()
        zoomText = tk.StringVar()
        widthText = tk.StringVar()
        heightText = tk.StringVar()
        shearText = tk.StringVar()
        noiText = tk.StringVar()

        self.horizon = tk.Checkbutton(self.window, text = ' horizontal flip', var = self.horizonCheck)
        self.vertical = tk.Checkbutton(self.window, text = ' vertical flip', var = self.verticalCheck)
        self.rotationRange = tk.Entry(self.window, justify = 'center', textvariable = rotText)
        self.zoomRange = tk.Entry(self.window, justify = 'center', textvariable = zoomText)
        self.widthRange = tk.Entry(self.window, justify = 'center', textvariable = widthText)
        self.heightRange = tk.Entry(self.window, justify = 'center', textvariable = heightText)
        self.shearRange = tk.Entry(self.window, justify = 'center', textvariable = shearText)
        self.numofImg = tk.Entry(self.window, justify = 'center', textvariable = noiText)

        rotationLb = tk.Label(self.window, text = 'rotation range')
        zoomLb = tk.Label(self.window, text = 'zoom range')
        widthLb = tk.Label(self.window, text = 'width shift range')
        heightLb = tk.Label(self.window, text = 'height shift range')
        shearLb = tk.Label(self.window, text = 'shear range')
        saveImgLb = tk.Label(self.window, text = '# of save image')

        rotText.set(0)
        zoomText.set(0.0)
        widthText.set(0.0)
        heightText.set(0.0)
        shearText.set(0.0)
        noiText.set(100)

        self.rotationRange.place(y = 70, x = 165, width = 50)
        rotationLb.place(y = 70, x = 30)

        self.shearRange.place(y = 70, x = 365, width = 50)
        shearLb.place(y = 70, x = 225)

        self.widthRange.place(y = 100, x = 165, width = 50)
        widthLb.place(y = 100, x = 30)

        self.heightRange.place(y = 100, x = 365, width = 50)
        heightLb.place(y = 100, x = 225)

        self.zoomRange.place(y = 130, x = 165, width = 50)
        zoomLb.place(y = 130, x = 30)

        self.numofImg.place(y = 130, x = 365, width = 50)
        saveImgLb.place(y = 130, x = 225)

        self.vertical.place(y = 40, x = 50)
        self.horizon.place(y = 40, x = 250)
        self.window.mainloop()

    def apply(self):
        try:
            testImg = self.imgs[0]
            print(type(testImg))

        except Exception as e:
            text = f'[ERR 2] {e} - images load first... :('
            self.log(text)

            print(text)
            messagebox.showerror('error occured', 'please load images')
            pass

        try:
            hori, ver, rot, zoom, width, height, shear = self.getParams()
            generator = ImageDataGenerator(rotation_range = rot, width_shift_range = width, height_shift_range= height,
                                            zoom_range=zoom, horizontal_flip=hori, vertical_flip=ver, shear_range=shear)

            testImg = pre.image.img_to_array(testImg)
            testImg = np.expand_dims(testImg, axis = 0)

            for (idx, batch) in enumerate(generator.flow(testImg, batch_size=1)):
                img = pre.image.array_to_img(batch[0])
                print(type(img))
                img.save('dd.jpg')
                if idx % 1 == 0:
                    break

            # image = PIL.Image.fromarray(img)
            image = ImageTk.PhotoImage(image = img)

            firstImg = tk.Label(self.imgFrame, image = image)
            firstImg.pack(fill = 'both')

            text = '[INFO] apply comlete! :D'
            self.log(text)
            print(text)

        except Exception as e:
            text = f'[ERR 4] {e} - apply failed... :('
            self.log(text)

            print(text)
            pass

    def loadImgs(self):
        try:
            imgPath = filedialog.askdirectory(title = 'Select your Images')
            self.imgPaths = list(sorted(list_images(imgPath)))
            self.showimgs = []
            self.imgs = []

            for img in self.imgPaths:
                print(img)
                image = cv2.imread(img)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                npimg = image.copy()

                image = PIL.Image.fromarray(image)
                image = ImageTk.PhotoImage(image = image)
                self.imgs.append(npimg)
                self.showimgs.append(image)

            print(type(self.imgs[0]))
            firstImg = tk.Label(self.imgFrame, image = self.showimgs[0])
            firstImg.pack(fill = 'both')

            text = '[INFO] images load complete! :D'
            self.log(text)
            print(text)

        except Exception as e:
            text = f'[ERR 1] {e} - invalid image directory... :('
            self.log(text)

            print(text)
            messagebox.showerror('error occured', 'invalid image directory')
            pass

    def generate(self):
        try:
            savePath = filedialog.askdirectory(title = 'Select your save path')
            print(savePath)

            text = '[INFO] save path ready! :D'
            self.log(text)
            print(text)
            try:
                noi = int(self.numofImg.get())

            except Exception as e:
                text = f'[ERR 3] {e} - invalid parameters... :('
                self.log(text)

                print(text)
                messagebox.showerror('error occured', 'invalid parameters included')
                pass

            try:
                hori, ver, rot, zoom, width, height, shear = self.getParams()
                generator = ImageDataGenerator(rotation_range = rot, width_shift_range = width, height_shift_range= height,
                                                zoom_range=zoom, horizontal_flip=hori, vertical_flip=ver, shear_range=shear)

                for (idx, oriImg) in enumerate(self.imgs):

                    imgPath = self.imgPaths[idx]
                    dirName = imgPath.split(os.path.sep)[-2]
                    os.makedirs(f'{savePath}/{dirName}', exist_ok=True)

                    dirs = "/".join(imgPath.split(os.path.sep)[:-1])
                    fileCnt = len(os.listdir(dirs))
                    print(dirName, fileCnt)
                    print(int(noi/fileCnt))

                    oriImg = pre.image.img_to_array(oriImg)
                    oriImg = np.expand_dims(oriImg, axis = 0)
                    cnt = int(noi / fileCnt) + int(noi % fileCnt) if (idx + 1) == fileCnt else int(noi / fileCnt)

                    for (idx2, batch) in enumerate(generator.flow(oriImg, batch_size=1)):
                        genImg = pre.image.array_to_img(batch[0])
                        genImg.save(f'{savePath}/{dirName}/{time.time()}_{randomName(12)}{idx2}.jpg')

                        if (idx2+1) % cnt == 0:
                            break
                
                text = '[INFO] generate ok! :D'
                self.log(text)
                print(text)

            except Exception as e:
                text = f'[ERR 5] {e} - generate failed... :('
                self.log(text)
                messagebox.showerror('error occured', 'generate failed...')

                print(text)
                pass

        except Exception as e:
            text = f'[ERR 6] {e} - invalid save path... :('
            self.log(text)

            print(text)
            pass

    def getParams(self):
        try:
            hori = self.horizonCheck.get()
            ver = self.verticalCheck.get()

            rot = int(self.rotationRange.get())
            zoom = float(self.zoomRange.get())
            width = float(self.widthRange.get())
            height = float(self.heightRange.get())
            shear = float(self.shearRange.get())

            return hori, ver, rot, zoom, width, height, shear

        except Exception as e:
            text = f'[ERR 3] {e} - get params failed... :('
            self.log(text)

            print(text)
            messagebox.showerror('error occured', 'invalid argument included')
            pass
        
    def log(self, text):
        TIME = time.localtime(time.time())
        YYYY = TIME.tm_year
        MM = TIME.tm_mon
        DD = TIME.tm_mday
        H = TIME.tm_hour
        M = TIME.tm_min
        S = TIME.tm_sec
        nowTime = f'[{YYYY:04d}-{MM:02d}-{DD:02d} {H:02d}:{M:02d}:{S:02d}] - '

        os.makedirs('build/log', exist_ok=True)
        with open('build/log/logs.txt', 'a') as f:
            text = nowTime + text + '\n'
            f.write(text)
    
    def EXIT(self):
        if messagebox.askokcancel('Quit', 'Do you want to quit?'):
            exit()

if __name__ == '__main__':
    canvas()
    
