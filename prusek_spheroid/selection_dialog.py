from tkinter import Toplevel, Label, Button, Frame
from PIL import Image, ImageTk
import cv2 as cv


class SelectionDialog:
    def __init__(self, master, counter, total_files, user_decision_lock):
        self.master = master
        self.counter = counter
        self.total_files = total_files
        self.cv_image1 = None
        self.cv_image2 = None
        self.cv_mask1 = None
        self.cv_mask2 = None
        self.image_save_path = None
        self.mask_save_path = None
        self.user_decision_lock = user_decision_lock

        self.selection_dialog = None
        self.photo1 = None
        self.photo2 = None
        self.image_label1 = None
        self.image_label2 = None

        self.create_selection_dialog()

    def create_selection_dialog(self):
        if self.selection_dialog:
            self.selection_dialog.destroy()

        initial_width = 1040
        initial_height = 520
        self.selection_dialog = Toplevel(self.master)
        self.selection_dialog.title(f"Processed image: {self.counter}/{self.total_files}")
        self.selection_dialog.geometry(f"{initial_width}x{initial_height}")

        frame = Frame(self.selection_dialog)
        frame.pack(fill="both", expand=True)

        self.photo1 = ImageTk.PhotoImage(Image.new("RGB", (500, 500), "gray"))
        self.photo2 = ImageTk.PhotoImage(Image.new("RGB", (500, 500), "gray"))

        frame1 = Frame(frame)
        frame1.pack(side="left", padx=10)
        Label(frame1, text="WITHOUT holes").pack()
        self.image_label1 = Label(frame1, image=self.photo1)
        self.image_label1.pack()
        Button(frame1, text="Save WITHOUT Holes", command=lambda: self.save_and_proceed(self.cv_image1, self.cv_mask1)).pack()

        frame2 = Frame(frame)
        frame2.pack(side="right", padx=10)
        Label(frame2, text="WITH holes").pack()
        self.image_label2 = Label(frame2, image=self.photo2)
        self.image_label2.pack()
        Button(frame2, text="Save WITH Holes", command=lambda: self.save_and_proceed(self.cv_image2, self.cv_mask2)).pack()

        self.selection_dialog.bind("<Configure>", self.resize_images)

    def save_and_proceed(self, image, mask):
        if not cv.imwrite(self.image_save_path, image):
            print(f"FAILED to save image: {self.image_save_path}")
        if not cv.imwrite(self.mask_save_path, mask):
            print(f"FAILED to save mask: {self.mask_save_path}")
        if self.user_decision_lock.locked():
            self.user_decision_lock.release()

    def update_selection_dialog(self, img_without, img_with, mask_without, mask_with, image_path, mask_path, counter):
        self.counter = counter
        self.image_save_path = image_path
        self.mask_save_path = mask_path
        self.cv_image1 = img_without
        self.cv_image2 = img_with
        self.cv_mask1 = mask_without
        self.cv_mask2 = mask_with

        # Update the window title with the new counter value
        if self.selection_dialog:
            self.selection_dialog.title(f"Processed image: {self.counter}/{self.total_files}")

        self.resize_images()

    def resize_images(self, event=None):
        if self.cv_image1 is not None and self.cv_image2 is not None:
            current_width = self.selection_dialog.winfo_width()
            new_size = self.calculate_new_size(current_width)
            self.resize_and_display_images(new_size[0], new_size[1])

    def resize_and_display_images(self, width, height):
        try:
            if self.selection_dialog.winfo_exists() and self.image_label1.winfo_exists() and self.image_label2.winfo_exists():
                img_pil1 = Image.fromarray(cv.cvtColor(self.cv_image1, cv.COLOR_BGR2RGB)).resize((width, height), Image.LANCZOS)
                img_pil2 = Image.fromarray(cv.cvtColor(self.cv_image2, cv.COLOR_BGR2RGB)).resize((width, height), Image.LANCZOS)

                self.photo1 = ImageTk.PhotoImage(img_pil1)
                self.photo2 = ImageTk.PhotoImage(img_pil2)
                self.image_label1.config(image=self.photo1)
                self.image_label2.config(image=self.photo2)
        except Exception as e:
            print(f"Error resizing images: {e}")

    @staticmethod
    def calculate_new_size(current_width):
        aspect_ratio = 1.0
        new_width = int(current_width * 0.45)
        new_height = int(new_width * aspect_ratio)
        return new_width, new_height

    def destroy_dialog(self):
        """Closes and destroys the selection dialog window."""
        if self.selection_dialog:
            self.selection_dialog.destroy()
            self.selection_dialog = None