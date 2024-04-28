from tkinter import*
from tkinter import ttk
from PIL import Image,ImageTk

import os

class Face_Recognition_System:
    def __init__(self,root):
        self.root=root
        self.root.geometry("1366x768+0+0")
        self.root.title("Vaani")

        bg1=Image.open(r"Images_GUI\bg3.jpg")
        bg1=bg1.resize((1366,768),Image.ANTIALIAS)
        self.photobg1=ImageTk.PhotoImage(bg1)

        bg_img = Label(self.root,image=self.photobg1)
        bg_img.place(x=0,y=0,width=1366,height=768)


        title_lb1 = Label(bg_img,text="Vaani",font=("Chiller",30,"bold"),bg="Black",fg="white")
        title_lb1.place(x=0,y=0,width=1366,height=45)


        # gs_logo=Image.open(r"Images_GUI/gs.png")
        # gs_logo=gs_logo.resize((250,250),Image.ANTIALIAS)
        # self.gs_logo1=ImageTk.PhotoImage(gs_logo)
        # gs_img = Label(self.root,image=self.gs_logo1)
        # gs_img.place(x=50,y=50,width=250,height=250)



        att_img_btn=Image.open(r"Images_GUI/att.jpg")
        att_img_btn=att_img_btn.resize((250,250),Image.ANTIALIAS)
        self.att_img1=ImageTk.PhotoImage(att_img_btn)

        att_b1 = Button(bg_img,command=self.attendance_pannel,image=self.att_img1,cursor="hand2",)
        att_b1.place(x=590,y=250,width=250,height=250)

        att_b1_1 = Button(bg_img,command=self.attendance_pannel,text="Vaani",cursor="hand2",font=("tahoma",15,"bold"),bg="white",fg="navyblue")
        att_b1_1.place(x=590,y=500,width=250,height=45)

        exi_img_btn=Image.open(r"Images_GUI/exi.jpg")
        exi_img_btn=exi_img_btn.resize((90,90),Image.ANTIALIAS)
        self.exi_img1=ImageTk.PhotoImage(exi_img_btn)

        exi_b1 = Button(bg_img,command=self.Close,image=self.exi_img1,cursor="hand2",)
        exi_b1.place(x=1200,y=620,width=90,height=90)

        exi_b1_1 = Button(bg_img,command=self.Close,text="Exit",cursor="hand2",font=("tahoma",15,"bold"),bg="white",fg="navyblue")
        exi_b1_1.place(x=1200,y=703,width=90,height=45)

    
    def attendance_pannel(self):
        self.new_window=Toplevel(self.root)
        # self.app=Attendance(self.new_window)
        import app
        # self.app=os.system('python app.py')


    


    def Close(self):
        root.destroy()
    
    





if __name__ == "__main__":
    root=Tk()
    obj=Face_Recognition_System(root)
    root.mainloop()
