import os
import tkinter as tk
import customtkinter as ctk
import numpy as np
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from process_images import Process_Images, GrayScale_Images, DepthMap_Images
from Featuring import extract_keypoints_with_depth
from PointCloud import save_keypoints_as_pointcloud
from modeling import pointcloud_to_mesh
import open3d as o3d

class Create_Main_Window:
    def __init__(self, root):
        self.root = root
        self.root.title("3D Reconstruction App")
        root.geometry(str(round(root.winfo_screenwidth()/1.28)) + "x" + str(round(root.winfo_screenheight()/1.234))+"+100+50")

        # Configure grid layout
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_rowconfigure(0, weight=1)

        # Left panel for buttons
        self.left_panel = ctk.CTkFrame(self.root, width=200)
        self.left_panel.grid(row=0, column=0, padx=10, pady=10, sticky="ns")
        self.left_panel.grid_propagate(False)

        # Right panel for thumbnails and 3D view
        self.right_panel = ctk.CTkFrame(self.root)
        self.right_panel.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

        # Configure right panel grid
        self.right_panel.grid_rowconfigure(0, weight=1)
        self.right_panel.grid_rowconfigure(1, weight=1)
        self.right_panel.grid_columnconfigure(0, weight=1)

        # Thumbnail area (top right)
        self.thumbnail_frame = ctk.CTkScrollableFrame(self.right_panel, label_text="Selected Images")
        self.thumbnail_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")

        # 3D view area (bottom right)
        self.view_3d_frame = ctk.CTkFrame(self.right_panel)
        self.view_3d_frame.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")

        # Add buttons to left panel
        self.load_button = ctk.CTkButton(
            self.left_panel,
            text="Load Images",
            command=self.load_images
        )
        self.load_button.pack(pady=10, padx=10, fill="x")

        self.reconstruct_button = ctk.CTkButton(
            self.left_panel,
            text="Reconstruct",
            command=self.reconstruct
        )
        self.reconstruct_button.pack(pady=10, padx=10, fill="x")

        # Variables
        self.image_paths = []
        self.thumbnails = []

        # Initialize 3D view
        self.init_3d_view()

    def load_images(self):
        filetypes = (
            ("Image files", "*.jpg *.jpeg *.png"),
            ("All files", "*.*")
        )

        paths = filedialog.askopenfilenames(
            title="Select images",
            initialdir=os.path.expanduser("~"),
            filetypes=filetypes
        )

        if paths:

            existing_paths = set(self.image_paths)
            for path in paths:
                if path not in existing_paths:
                    self.image_paths.append(path)
                    existing_paths.add(path)

            # self.image_paths = paths
            self.show_thumbnails()

    def show_thumbnails(self):
        # Clear existing thumbnails widgets but keep paths
        for widget in self.thumbnail_frame.winfo_children():
            widget.destroy()

        self.thumbnail_widgets = []

        # Display all thumbnails
        for path in self.image_paths:
            try:
                # Create frame for each thumbnail + button
                frame = ctk.CTkFrame(self.thumbnail_frame)
                frame.pack(fill="x", padx=5, pady=5)

                # Load and resize image
                img = Image.open(path)
                img.thumbnail((100, 100))
                ctk_img = ctk.CTkImage(light_image=img, size=img.size)

                # Create image label
                img_label = ctk.CTkLabel(frame, image=ctk_img, text="")
                img_label.image = ctk_img  # Keep reference
                img_label.pack(side="left", padx=5)

                # Create delete button
                delete_btn = ctk.CTkButton(
                    frame,
                    text="X",
                    width=30,
                    command=lambda p=path: self.delete_image(p)
                )
                delete_btn.pack(side="right", padx=5)

                # Add path label (optional)
                path_label = ctk.CTkLabel(frame, text=os.path.basename(path))
                path_label.pack(side="left", padx=5, fill="x", expand=True)

                # Store reference
                self.thumbnail_widgets.append((frame, path))
            except Exception as e:
                print(f"Error loading image {path}: {e}")

    def delete_image(self, path_to_delete):
        # Remove from paths list
        self.image_paths = [p for p in self.image_paths if p != path_to_delete]

        # Update thumbnails display
        self.show_thumbnails()

    def reconstruct(self):
        if not self.image_paths:
            messagebox.showerror("Ошибка", "Не выбраны изображения")
            return

        print("Starting Image work...")
        print("Starting Image Processing...")
        image_processed_paths = Process_Images(self.image_paths)
        print("Create Image GrayScaling...")
        grayscale_image_paths = GrayScale_Images(image_processed_paths)
        print("Create Image Depth Map...")
        disparity, depthmap_image_path = DepthMap_Images(grayscale_image_paths)


        print("Starting featuring keypoints...")

        kp, des, pswd = extract_keypoints_with_depth('images\depthMaps\depth_map_final.png', disparity, 50, 0.01)
        save_keypoints_as_pointcloud(pswd)
        pcd = o3d.io.read_point_cloud("keypoints_cloud.ply")
        o3d.visualization.draw_geometries([pcd], window_name=f"keypoints length: {len(kp)}", width=800, height=600)

        print("Starting create model...")
        pointcloud_to_mesh("keypoints_cloud.ply")

        print("Show model...")
        self.update_3d_view()

    def init_3d_view(self):
        # Create matplotlib figure
        self.fig = Figure(figsize=(5, 4), dpi=100)
        self.ax = self.fig.add_subplot(111, projection='3d')

        # Create a placeholder 3D object ------------------------------------------------------
        # self.update_3d_view()

        # Embed figure in tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.view_3d_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def update_3d_view(self, ply_file="output_mesh.ply"):
        # Clear previous plot
        self.ax.clear()

        try:
            # Load 3D mesh using Open3D
            mesh = o3d.io.read_triangle_mesh(ply_file)

            # Convert mesh to numpy arrays
            vertices = np.asarray(mesh.vertices)
            triangles = np.asarray(mesh.triangles)

            # Extract coordinates
            x = vertices[:, 0]
            y = vertices[:, 1]
            z = vertices[:, 2]

            # Plot the mesh
            self.ax.plot_trisurf(x, y, z, triangles=triangles,
                            cmap='viridis', edgecolor='none')

            # Set labels
            self.ax.set_xlabel('X')
            self.ax.set_ylabel('Y')
            self.ax.set_zlabel('Z')
            self.ax.set_title('3D Model')

            # Auto-scale view
            self.ax.autoscale_view()

        except Exception as e:
            print(f"Error loading 3D model: {e}")
            # Show empty plot if error occurs
            self.ax.text(0.5, 0.5, 0.5, "No 3D model loaded",
                        ha='center', va='center')

        # Redraw canvas
        self.canvas.draw()














        # Create a simple 3D object (sphere for demonstration)
        # u = np.linspace(0, 2 * np.pi, 100)
        # v = np.linspace(0, np.pi, 100)
        # x = np.outer(np.cos(u), np.sin(v))
        # y = np.outer(np.sin(u), np.sin(v))
        # z = np.outer(np.ones(np.size(u)), np.cos(v))

        # Plot the surface
        # self.ax.plot_surface(x, y, z, color='b')
        # self.ax.set_title("3D Reconstruction")

        # Redraw
        # self.canvas.draw()



if __name__ == "__main__":
    root = ctk.CTk()
    app = Create_Main_Window(root)
    root.mainloop()