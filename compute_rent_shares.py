import yaml
import tkinter as tk
from tkinter import ttk, Scale, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import cvxpy as cp
import logging
from pathlib import Path  # For path handling


import argparse


# --- Start of the fix ---
# Define a custom representer for NumPy scalars
def numpy_scalar_representer(dumper, data):
    if isinstance(data, (np.float32, np.float64, np.int32, np.int64)):
        return dumper.represent_data(
            data.item()
        )  # Use .item() to get the Python scalar
    return dumper.represent_data(data)


# Add the representer to the default Dumper
yaml.add_representer(np.float64, numpy_scalar_representer)
yaml.add_representer(np.float32, numpy_scalar_representer)
yaml.add_representer(np.int64, numpy_scalar_representer)
yaml.add_representer(np.int32, numpy_scalar_representer)


# Define a custom constructor for NumPy scalars
def numpy_scalar_constructor(loader, node):
    # This assumes the scalar is a float, adjust if you have other types
    return np.array(loader.construct_pairs(node), dtype=np.float64).item()


# Add the constructor for the specific tag encountered in your YAML
yaml.add_constructor(
    "!!python/object/apply:numpy.core.multiarray.scalar",
    numpy_scalar_constructor,
    Loader=yaml.SafeLoader,
)

# --- End of the fix ---


parser = argparse.ArgumentParser()
parser.add_argument(
    "--base-dir",
    type=Path,
    default=Path(__file__).resolve().parent,
    help="Base directory for application data",
)
parser.add_argument(
    "--user-config-dir",
    type=Path,
    default=Path(__file__).resolve().parent / "user_configs_test_2",
    help="Directory containing user configuration YAML files",
)
parser.add_argument(
    "--global-config-path",
    type=Path,
    default=Path(__file__).resolve().parent / "global_config_2.yaml",
    help="Path to global configuration YAML file",
)
parser.add_argument(
    "--output-yaml-path",
    type=Path,
    default=Path(__file__).resolve().parent / "final_rent_allocations.yaml",
    help="Path to save final rent allocation YAML file",
)
parser.add_argument(
    "--bar-chart-path",
    type=Path,
    default=Path(__file__).resolve().parent / "rent_allocation_barplot.png",
    help="Path to save rent allocation bar chart image",
)
parser.add_argument(
    "--pie-chart-path",
    type=Path,
    default=Path(__file__).resolve().parent / "rent_allocation_piechart.png",
    help="Path to save rent allocation pie chart image",
)
parser.add_argument(
    "--beta-min-violation",
    type=float,
    default=10.0,
    help="Penalty coefficient for violating minimum rent constraints",
)

args = parser.parse_args()

BASE_DIR = args.base_dir
USER_CONFIG_DIR = args.user_config_dir
GLOBAL_CONFIG_PATH = args.global_config_path
OUTPUT_YAML_PATH = args.output_yaml_path
BAR_CHART_PATH = args.bar_chart_path
PIE_CHART_PATH = args.pie_chart_path
DEFAULT_BETA_MIN_VIOLATION = args.beta_min_violation


def safe_divide(numerator, denominator, default=0.0):
    """Safely divide two numbers, returning a default value if the denominator is zero."""
    if denominator == 0:
        return default
    return numerator / denominator


class RentAllocatorApp:
    def __init__(self, master):
        self.master = master
        master.title("Dynamic Rent Allocation")

        # --- Initialize Core Data ---
        self.total_rent = 0
        self.user_data = {}
        self.users = []
        self.ideals = {}
        self.mins = {}
        self.maxs = {}
        self.beta_min_violation = DEFAULT_BETA_MIN_VIOLATION

        self.final_allocations = None  # Will store the current optimized allocations
        self.proportional_initial_allocations = (
            {}
        )  # Store allocations based purely on ideals

        # --- Load Configuration ---
        if not self.load_all_configs():
            # If loading fails critically, messagebox and exit
            messagebox.showerror(
                "Configuration Error",
                "Failed to load critical configuration. Application will exit.",
            )
            master.destroy()
            return

        # Calculate proportional allocations once after loading configs
        self.proportional_initial_allocations = self._calculate_proportional_allocation(
            self.total_rent, self.ideals
        )

        # Initial adjustment value (alpha for optimization)
        self.global_adjustment_alpha = 0.0  # 0.0 for equal share, 1.0 for ideal-based

        # --- GUI Setup ---
        self._setup_gui()

        # Perform initial allocation calculation and UI update
        # Use the initial slider value (which should be 0)
        self.slider_changed(self.global_slider.get())

    def _load_yaml_config(self, filepath):
        try:
            with open(filepath, "r") as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logging.error(f"Error: File not found at {filepath}")
            return None
        except yaml.YAMLError as e:
            logging.error(f"Error loading YAML from {filepath}: {e}")
            return None

    def load_all_configs(self):
        global_config = self._load_yaml_config(GLOBAL_CONFIG_PATH)
        if not global_config or "TotalRent" not in global_config:
            logging.error(
                "Error: Could not load total rent from global_config.yaml or 'TotalRent' key is missing."
            )
            return False
        self.total_rent = global_config["TotalRent"]
        self.beta_min_violation = global_config.get(
            "BetaMinViolation", DEFAULT_BETA_MIN_VIOLATION
        )
        logging.info(
            f"Total rent: {self.total_rent}, Beta for min violation: {self.beta_min_violation}"
        )

        if not USER_CONFIG_DIR.exists():
            logging.warning(f"User config directory '{USER_CONFIG_DIR}' not found.")
            return False

        user_files = list(
            USER_CONFIG_DIR.glob("*.yaml")
        )  # Ensure it's a list for the check below
        if not user_files:
            logging.warning(
                f"No user configuration files found in '{USER_CONFIG_DIR}'."
            )
            return False

        for file_path in user_files:
            data = self._load_yaml_config(file_path)
            if data:
                username = file_path.stem  # Gets filename without extension
                self.user_data[username] = data

        if not self.user_data:
            logging.error("No valid user configurations were loaded.")
            return False

        self.users = list(self.user_data.keys())
        self.ideals = {
            user: data.get("ideal", 0) for user, data in self.user_data.items()
        }
        self.mins = {
            user: data.get("Minimum", 0) for user, data in self.user_data.items()
        }
        self.maxs = {
            user: data.get("Max", float("inf")) for user, data in self.user_data.items()
        }

        logging.info("User data loaded:")
        for user in self.users:
            logging.info(
                f"  {user}: ideal={self.ideals.get(user,0)}, Min={self.mins.get(user,0)}, Max={self.maxs.get(user, float('inf'))}"
            )
        return True

    def _calculate_proportional_allocation(self, total_rent, ideals):
        """Calculates rent allocation proportional to user ideals."""
        total_ideal_sum = sum(ideals.values())
        if not self.users:
            return {}  # No users to allocate to

        if (
            total_ideal_sum == 0
        ):  # Avoid division by zero; distribute equally if no ideals
            num_users = len(self.users)
            equal_share_val = total_rent / num_users if num_users > 0 else 0
            return {user: equal_share_val for user in self.users}
        return {
            user: total_rent * (ideal_val / total_ideal_sum)
            for user, ideal_val in ideals.items()
        }

    def _adjust_allocation_cvxpy(self, alpha_slider_value):
        """
        Adjust rent allocation using convex optimization that blends equal and preference-based shares,
        while respecting min and max constraints. Users with higher max rents are favored more as alpha increases.
        """
        if not self.users:
            logging.warning("No users defined for allocation.")
            return None

        users = self.users
        total_rent = self.total_rent
        num_users = len(users)
        equal_share = total_rent / num_users

        # Step 1: Compute preference scores based on max rent
        # Higher max => higher ability/willingness to pay => higher score
        pref_scores = {}
        score_sum = 0.0
        for u in users:
            max_r = self.maxs.get(u, float("inf"))
            # Ensure finite and meaningful weight
            score = max_r if max_r < float("inf") else 1.0
            pref_scores[u] = score
            score_sum += score

        # Normalize and distribute rent accordingly
        pref_shares = {
            u: (
                (pref_scores[u] / score_sum * total_rent)
                if score_sum > 0
                else equal_share
            )
            for u in users
        }

        # Step 2: Interpolate between equal and preference-based allocations
        blended_ideals = {
            u: (1 - alpha_slider_value) * equal_share
            + alpha_slider_value * pref_shares[u]
            for u in users
        }

        # Step 3: CVXPY variables and constraints
        alloc_vars = {u: cp.Variable(nonneg=True) for u in users}
        allocations_expr = cp.hstack([alloc_vars[u] for u in users])
        constraints = [cp.sum(allocations_expr) == total_rent]

        sum_min_rent = sum(self.mins.get(u, 0) for u in users)
        if sum_min_rent > total_rent:
            logging.warning(
                "Sum of minimum rents exceeds total rent. Dropping min constraints."
            )
            messagebox.showwarning(
                "Constraint Warning", "Minimum rent constraints dropped."
            )
            for u in users:
                constraints.append(alloc_vars[u] <= self.maxs.get(u, float("inf")))
        else:
            for u in users:
                constraints.append(alloc_vars[u] >= self.mins.get(u, 0))
                constraints.append(alloc_vars[u] <= self.maxs.get(u, float("inf")))

        # Step 4: Minimize deviation from blended ideals
        ideal_expr = cp.hstack([blended_ideals[u] for u in users])
        objective = cp.Minimize(cp.sum_squares(allocations_expr - ideal_expr))
        problem = cp.Problem(objective, constraints)

        try:
            problem.solve(solver=cp.ECOS)
            if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                allocations = {u: alloc_vars[u].value for u in users}
                rounded = {u: round(val, 2) for u, val in allocations.items()}

                # Adjust for rounding error
                diff = sum(rounded.values()) - total_rent
                if abs(diff) >= 0.01:
                    sorted_users = sorted(rounded, key=rounded.get, reverse=(diff > 0))
                    for u in sorted_users:
                        if abs(diff) < 0.01:
                            break
                        adj = min(abs(diff), 0.01)
                        rounded[u] -= adj if diff > 0 else -adj
                        diff = sum(rounded.values()) - total_rent

                return rounded
            else:
                logging.warning(f"Optimization status: {problem.status}")
                if problem.status == cp.INFEASIBLE:
                    for u in users:
                        if self.mins.get(u, 0) > self.maxs.get(u, float("inf")):
                            msg = (
                                f"Infeasible: User {u}'s minimum rent ({self.mins.get(u, 0)}) "
                                f"exceeds maximum ({self.maxs.get(u, float('inf'))})."
                            )
                            logging.error(msg)
                            messagebox.showwarning("Infeasible Constraints", msg)
                            break
                return None
        except cp.SolverError as e:
            logging.error(f"Solver failed: {e}")
            messagebox.showerror("Solver Error", f"The optimization solver failed: {e}")
            return None
        except Exception as e:
            logging.error(f"Unexpected error: {e}")
            messagebox.showerror(
                "Unexpected Error", f"An unexpected error occurred: {e}"
            )
            return None

    def _setup_gui(self):
        # Create a main frame to hold the canvas and scrollbar
        main_frame = ttk.Frame(self.master)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Create a Canvas
        self.canvas = tk.Canvas(main_frame)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Create a Scrollbar and link it to the Canvas
        self.scrollbar = ttk.Scrollbar(
            main_frame, orient=tk.VERTICAL, command=self.canvas.yview
        )
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.config(yscrollcommand=self.scrollbar.set)

        # Create a frame inside the canvas to hold all actual widgets
        # This is the frame that will be scrolled
        self.scrollable_frame = ttk.Frame(self.canvas)
        self.canvas.create_window(
            (0, 0),
            window=self.scrollable_frame,
            anchor="nw",
            tags="self.scrollable_frame",
        )

        # Configure the scrollable frame to expand with the canvas
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")),
        )
        # Bind mouse wheel scrolling
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)

        # --- Existing GUI Setup, now placed inside self.scrollable_frame ---

        # Allocation Display Frame
        allocation_frame = ttk.LabelFrame(
            self.scrollable_frame, text="Rent Details & Allocations"
        )
        allocation_frame.pack(padx=10, pady=10, fill="x")

        self.gui_allocation_labels = {}
        self.gui_user_info_labels = {}

        for user in self.users:
            max_val_display = (
                "inf"
                if self.maxs.get(user, float("inf")) == float("inf")
                else f"{self.maxs.get(user, 0):.2f}€"
            )
            user_info_text = (
                f"{user}: Raw ideal: {self.ideals.get(user, 0)} | "
                f"ideal-Proportional: {self.proportional_initial_allocations.get(user, 0):.2f}€ | "
                f"Min: {self.mins.get(user, 0):.2f}€ | Max: {max_val_display}"
            )
            self.gui_user_info_labels[user] = ttk.Label(
                allocation_frame, text=user_info_text
            )
            self.gui_user_info_labels[user].pack(pady=1, anchor="w")

            self.gui_allocation_labels[user] = ttk.Label(
                allocation_frame,
                text="Current Allocation: -- €",
                font=("Arial", 10, "bold"),
            )
            self.gui_allocation_labels[user].pack(pady=(0, 6), anchor="w")

        # Slider Frame
        slider_frame = ttk.LabelFrame(
            self.scrollable_frame, text="Allocation Adjustment"
        )
        slider_frame.pack(padx=10, pady=10, fill="x")
        slider_label_text = "Focus (0% = Equal Share, 100% = ideal-Proportional Share):"
        slider_label = ttk.Label(slider_frame, text=slider_label_text)
        slider_label.pack(pady=5)
        self.global_slider = Scale(
            slider_frame,
            from_=0,
            to=100,
            orient=tk.HORIZONTAL,
            length=400,
            resolution=1,
            command=lambda value: self.slider_changed(value),
        )
        self.global_slider.set(int(self.global_adjustment_alpha * 100))
        self.global_slider.pack(fill="x", padx=5, pady=5)

        # Plotting Area
        plot_frame = ttk.LabelFrame(self.scrollable_frame, text="Visualizations")
        plot_frame.pack(padx=10, pady=10, fill="both", expand=True)

        self.bar_fig, self.bar_ax = plt.subplots(figsize=(7.5, 4.5))  # Adjusted size
        self.bar_canvas = FigureCanvasTkAgg(self.bar_fig, master=plot_frame)
        self.bar_canvas_widget = self.bar_canvas.get_tk_widget()
        self.bar_canvas_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        self.pie_fig, self.pie_ax = plt.subplots(figsize=(5.5, 4.5))  # Adjusted size
        self.pie_canvas = FigureCanvasTkAgg(self.pie_fig, master=plot_frame)
        self.pie_canvas_widget = self.pie_canvas.get_tk_widget()
        self.pie_canvas_widget.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)

        # Save Button
        save_button = ttk.Button(
            self.scrollable_frame,
            text="Save Allocations & Plots",
            command=self.save_final_allocations,
        )
        save_button.pack(pady=10)

    def _on_mousewheel(self, event):
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def update_gui_allocation_labels(self):
        if self.final_allocations:
            for user, amount in self.final_allocations.items():
                if (
                    user in self.gui_allocation_labels
                ):  # Should always be true if users haven't changed
                    amount_val = (
                        amount if amount is not None else 0
                    )  # Handle potential None in allocation values
                    self.gui_allocation_labels[user].config(
                        text=f"Current Allocation: {float(amount_val):.2f} €"
                    )
        else:  # Clear labels or show error if allocation failed
            for user in self.users:
                if user in self.gui_allocation_labels:
                    self.gui_allocation_labels[user].config(
                        text="Current Allocation: Error / Unavailable"
                    )

    def update_plots(self):
        # Clear previous plots
        self.bar_ax.clear()
        self.pie_ax.clear()

        if (
            not self.final_allocations
            or not self.users
            or not all(
                isinstance(val, (int, float)) for val in self.final_allocations.values()
            )
        ):
            self.bar_ax.text(
                0.5,
                0.5,
                "Allocation data unavailable\nor invalid for plotting.",
                ha="center",
                va="center",
                fontsize=10,
                wrap=True,
            )
            self.bar_canvas.draw()
            self.pie_ax.text(
                0.5,
                0.5,
                "Allocation data unavailable\nor invalid for plotting.",
                ha="center",
                va="center",
                fontsize=10,
                wrap=True,
            )
            self.pie_canvas.draw()
            return

        # Bar Chart Update
        num_users = len(self.users)
        x_indices = np.arange(num_users)
        width = 0.28  # Adjusted width for better spacing

        prop_alloc_vals = [
            self.proportional_initial_allocations.get(u, 0) for u in self.users
        ]
        final_alloc_vals = [self.final_allocations.get(u, 0) for u in self.users]
        min_vals = [self.mins.get(u, 0) for u in self.users]
        max_vals_plot = [
            self.maxs.get(u, np.nan) for u in self.users
        ]  # Use NaN for inf for plotting logic

        self.bar_ax.bar(
            x_indices - width / 2,
            prop_alloc_vals,
            width,
            label="ideal-Proportional",
            color="lightblue",
            edgecolor="grey",
        )
        self.bar_ax.bar(
            x_indices + width / 2,
            final_alloc_vals,
            width,
            label="Final Allocation",
            color="salmon",
            edgecolor="grey",
        )

        for i, user in enumerate(self.users):
            # Min lines
            self.bar_ax.plot(
                [x_indices[i] - width, x_indices[i] + width],
                [min_vals[i], min_vals[i]],
                color="green",
                linestyle="--",
                lw=1.2,
                label="Min Rent" if i == 0 else "",
            )
            # Max lines (only if not inf)
            if not np.isnan(max_vals_plot[i]) and max_vals_plot[i] != float("inf"):
                self.bar_ax.plot(
                    [x_indices[i] - width, x_indices[i] + width],
                    [max_vals_plot[i], max_vals_plot[i]],
                    color="red",
                    linestyle=":",
                    lw=1.2,
                    label="Max Rent" if i == 0 else "",
                )

        self.bar_ax.set_xticks(x_indices)
        self.bar_ax.set_xticklabels(self.users, rotation=45, ha="right", fontsize=9)
        self.bar_ax.set_ylabel("Rent (€)", fontsize=10)
        self.bar_ax.set_title("Rent Allocations Overview", fontsize=12)
        self.bar_ax.legend(fontsize="small")
        self.bar_ax.grid(axis="y", linestyle="--", alpha=0.7)
        self.bar_fig.tight_layout()
        self.bar_canvas.draw()

        # Pie Chart Update
        valid_pie_data = [
            val for val in final_alloc_vals if val is not None and val > 1e-6
        ]  # Filter out None, zero or tiny values for pie
        pie_labels_data = [
            self.users[i]
            for i, val in enumerate(final_alloc_vals)
            if val is not None and val > 1e-6
        ]

        if not valid_pie_data or sum(valid_pie_data) == 0:
            self.pie_ax.text(
                0.5,
                0.5,
                "No positive allocations\nto display in pie chart.",
                ha="center",
                va="center",
                fontsize=10,
                wrap=True,
            )
        else:
            pie_display_labels = [
                f"{label}\n({self.final_allocations.get(label, 0):.2f}€)"
                for label in pie_labels_data
            ]
            self.pie_ax.pie(
                valid_pie_data,
                labels=pie_display_labels,
                autopct="%1.1f%%",
                startangle=90,
                textprops={"fontsize": 8},
            )
            self.pie_ax.set_title("Final Rent Share", fontsize=12)
        self.pie_ax.axis("equal")
        self.pie_fig.tight_layout()
        self.pie_canvas.draw()

    def slider_changed(self, new_value_str):
        try:
            self.global_adjustment_alpha = float(new_value_str) / 100.0
        except ValueError:
            logging.error(f"Invalid slider value: {new_value_str}")
            return

        new_allocations = self._adjust_allocation_cvxpy(self.global_adjustment_alpha)

        if new_allocations is not None:
            self.final_allocations = new_allocations
            logging.info(
                f"Slider updated. Alpha: {self.global_adjustment_alpha:.2f}. New Allocations computed: {self.final_allocations}, sum: {sum(self.final_allocations.values()):.2f}€"
            )
        else:
            # Optimization failed. self.final_allocations might retain the last valid one.
            # Or, if you want to force plots to show "unavailable": self.final_allocations = None
            logging.warning(
                f"Slider updated. Alpha: {self.global_adjustment_alpha:.2f}. Optimization failed. Plots/labels may not reflect desired changes or show last valid state."
            )
            # No messagebox here to avoid spamming if slider is dragged quickly; logging is sufficient.
            # The plots and labels will show "unavailable" if self.final_allocations is None.

        self.update_gui_allocation_labels()
        self.update_plots()

    def save_final_allocations(self):
        if self.final_allocations and all(
            isinstance(val, (int, float)) for val in self.final_allocations.values()
        ):
            try:
                with open(OUTPUT_YAML_PATH, "w") as f:
                    yaml.dump(
                        self.final_allocations,
                        f,
                        allow_unicode=True,
                        sort_keys=False,
                        default_flow_style=False,
                    )
                logging.info(f"Allocations saved to {OUTPUT_YAML_PATH}")

                self.bar_fig.savefig(BAR_CHART_PATH)
                logging.info(f"Bar chart saved to {BAR_CHART_PATH}")
                self.pie_fig.savefig(PIE_CHART_PATH)
                logging.info(f"Pie chart saved to {PIE_CHART_PATH}")
                messagebox.showinfo(
                    "Save Successful",
                    f"Allocations and plots saved to:\n{OUTPUT_YAML_PATH.name}\n{BAR_CHART_PATH.name}\n{PIE_CHART_PATH.name}",
                )
            except Exception as e:
                logging.error(f"Error saving allocations or plots: {e}")
                messagebox.showerror(
                    "Save Error", f"Failed to save allocations/plots: {e}"
                )
        else:
            logging.warning("Attempted to save but no valid allocations available.")
            messagebox.showwarning(
                "Save Failed",
                "No valid allocations to save. Please ensure the optimization was successful.",
            )


def main():
    root = tk.Tk()
    # Set a minimum size for the window for better layout
    root.minsize(800, 600)
    app = RentAllocatorApp(root)

    # Check if the app instance was successfully created and essential data loaded
    # (e.g. master window not destroyed due to critical init error)
    if not root.winfo_exists():
        logging.error(
            "Application window was destroyed during initialization. Exiting."
        )
        return

    # Only run mainloop if app initialization was successful enough
    root.mainloop()


if __name__ == "__main__":
    main()
