import matplotlib.pyplot as plt


class PathPrinter:
    def __init__(self, scale, landmark_radius):
        self.landmark_radius = landmark_radius
        self.SCALE = scale
        pass

    def save_path_image(self, landmarks, start, goal, G, path, filename="rrt_path.png"):
        plt.figure(figsize=(8, 8))

        if len(G) > 0:
            Gx, Gy = zip(*[(gx * self.SCALE, gy * self.SCALE) for gx, gy in G])
            plt.scatter(Gx, Gy, c="lightgray", s=10, label="RRT nodes")

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
            plt.gca().add_patch(circle)
            plt.text(lx_mm, ly_mm, f"ID{lid}", color="red")

        plt.scatter(
            start[0] * self.SCALE,
            start[1] * self.SCALE,
            c="green",
            s=100,
            marker="o",
            label="Start",
        )
        plt.scatter(
            goal[0] * self.SCALE,
            goal[1] * self.SCALE,
            c="blue",
            s=100,
            marker="*",
            label="Goal",
        )

        if path is not None and len(path) > 1:
            px, py = zip(*[(px * self.SCALE, py * self.SCALE) for px, py in path])
            plt.plot(px, py, c="black", linewidth=2, label="Path")

        plt.xlabel("X [mm]")
        plt.ylabel("Y [mm]")
        plt.title("RRT Path Planning")
        plt.legend()
        plt.axis("equal")
        plt.grid(True)

        plt.savefig(filename)
        plt.close()
        print(f"Path image saved as {filename}")
