import matplotlib.pyplot as plt


class PathPrinter:
    def __init__(self, scale, landmark_radius):
        self.landmark_radius = landmark_radius
        self.SCALE = scale
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        plt.ion()

    def _draw(self, landmarks, start, goal, G, path, ax):
        ax.clear()

        if len(G) > 0:
            Gx, Gy = zip(*[(gx * self.SCALE, gy * self.SCALE) for gx, gy in G])
            ax.scatter(Gx, Gy, c="lightgray", s=10, label="RRT nodes")

        for lid, lx, ly in landmarks:
            lx_mm = lx * self.SCALE
            ly_mm = ly * self.SCALE
            circle = plt.Circle(
                (lx_mm, ly_mm),
                self.landmark_radius,
                color="red",
                fill=False,
                linestyle="--",
            )
            ax.add_patch(circle)
            ax.text(lx_mm, ly_mm, f"ID{lid}", color="red")

        ax.scatter(
            start[0] * self.SCALE,
            start[1] * self.SCALE,
            c="green",
            s=100,
            marker="o",
            label="Start",
        )
        ax.scatter(
            goal[0] * self.SCALE,
            goal[1] * self.SCALE,
            c="blue",
            s=100,
            marker="*",
            label="Goal",
        )

        if path is not None and len(path) > 1:
            px, py = zip(*[(px * self.SCALE, py * self.SCALE) for px, py in path])
            ax.plot(px, py, c="black", linewidth=2, label="Path")

        ax.set_xlabel("X [mm]")
        ax.set_ylabel("Y [mm]")
        ax.set_title("RRT Path Planning")
        ax.legend()
        ax.axis("equal")
        ax.grid(True)

    def save_path_image(self, landmarks, start, goal, G, path, filename="rrt_path.png"):
        fig, ax = plt.subplots(figsize=(8, 8))
        self._draw(landmarks, start, goal, G, path, ax)
        fig.savefig(filename, bbox_inches="tight")
        plt.close(fig)
        print(f"Path image saved as {filename}")

    def show_path_image(self, landmarks, start, goal, G, path):
        self._draw(landmarks, start, goal, G, path, self.ax)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001)
