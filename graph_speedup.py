import matplotlib.pyplot as plt
import re
import numpy as np


def parse_log_file(filename):
    """
    Parse the quickhull log file and extract timing information for different stages.

    Returns:
    - point_counts: List of point counts
    - serial_sort_times: List of serial sort times
    - parallel_sort_times: List of parallel sort times
    - serial_algo_times: List of serial algorithm execution times
    - parallel_algo_times: List of parallel algorithm execution times
    - serial_total_times: List of serial total execution times
    - parallel_total_times: List of parallel total execution times
    """
    point_counts = []
    serial_sort_times = []
    parallel_sort_times = []
    serial_algo_times = []
    parallel_algo_times = []
    serial_total_times = []
    parallel_total_times = []

    with open(filename, "r") as file:
        for line in file:
            # Check for point count
            point_match = re.search(r"Running for (\d+) points\.\.\.", line)
            if point_match:
                point_counts.append(int(point_match.group(1)))

            # Parse serial times
            serial_sort_match = re.search(r"Serial sort time: (\d+)ms", line)
            if serial_sort_match:
                serial_sort_times.append(int(serial_sort_match.group(1)))

            serial_algo_match = re.search(
                r"Serial algorithm execution time: (\d+)ms", line
            )
            if serial_algo_match:
                serial_algo_times.append(int(serial_algo_match.group(1)))

            serial_total_match = re.search(
                r"Serial total execution time: (\d+)ms", line
            )
            if serial_total_match:
                serial_total_times.append(int(serial_total_match.group(1)))

            # Parse parallel times
            parallel_sort_match = re.search(r"Parallel sort time: (\d+)ms", line)
            if parallel_sort_match:
                parallel_sort_times.append(int(parallel_sort_match.group(1)))

            parallel_algo_match = re.search(
                r"Parallel algorithm execution time: (\d+)ms", line
            )
            if parallel_algo_match:
                parallel_algo_times.append(int(parallel_algo_match.group(1)))

            parallel_total_match = re.search(
                r"Parallel total execution time: (\d+)ms", line
            )
            if parallel_total_match:
                parallel_total_times.append(int(parallel_total_match.group(1)))

    return (
        point_counts,
        serial_sort_times,
        parallel_sort_times,
        serial_algo_times,
        parallel_algo_times,
        serial_total_times,
        parallel_total_times,
    )


def calculate_speedup(serial_times, parallel_times):
    """
    Calculate speedup as serial time / parallel time
    Handle potential zero division
    """
    speedup = []
    for s, p in zip(serial_times, parallel_times):
        # Avoid division by zero, use a small epsilon if parallel time is zero
        # Add a small value to prevent divide by zero and handle very small times
        speedup.append(max(s / (p + 0.001), 0.001))
    return speedup


def plot_detailed_speedup(filename):
    """
    Create a plot showing speedup for sort, algorithm, and total execution
    """
    # Parse the log file
    (
        point_counts,
        serial_sort_times,
        parallel_sort_times,
        serial_algo_times,
        parallel_algo_times,
        serial_total_times,
        parallel_total_times,
    ) = parse_log_file(filename)

    # Calculate speedups
    sort_speedup = calculate_speedup(serial_sort_times, parallel_sort_times)
    algo_speedup = calculate_speedup(serial_algo_times, parallel_algo_times)
    total_speedup = calculate_speedup(serial_total_times, parallel_total_times)

    # Set up the plot
    plt.figure(figsize=(14, 8))

    # Width of each bar group
    width = 0.25

    # X-axis positions
    x = range(len(point_counts))

    # Create bar plots
    plt.bar(
        [i - width for i in x],
        sort_speedup,
        width,
        label="Sort Speedup",
        color="blue",
        alpha=0.7,
    )
    plt.bar(
        [i for i in x],
        algo_speedup,
        width,
        label="Algorithm Speedup",
        color="green",
        alpha=0.7,
    )
    plt.bar(
        [i + width for i in x],
        total_speedup,
        width,
        label="Total Execution Speedup",
        color="red",
        alpha=0.7,
    )

    # Add speedup values on top of each bar
    for i, (s, a, t) in enumerate(zip(sort_speedup, algo_speedup, total_speedup)):
        plt.text(i - width, s, f"{s:.2f}x", ha="center", va="bottom", fontsize=8)
        plt.text(i, a, f"{a:.2f}x", ha="center", va="bottom", fontsize=8)
        plt.text(i + width, t, f"{t:.2f}x", ha="center", va="bottom", fontsize=8)

    # Customize the plot
    plt.xlabel("Number of Points", fontsize=12)
    plt.ylabel("Speedup (Serial Time / Parallel Time)", fontsize=12)
    plt.title(
        "Quickhull with optimized flag (-O3): Parallel Speedup Breakdown", fontsize=14
    )
    plt.xticks([i for i in x], point_counts, rotation=45)

    # Add a horizontal line at y=1 to show baseline
    plt.axhline(y=1, color="r", linestyle="--", alpha=0.5)

    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()

    # Save the plot
    plt.savefig("pictures/quickhull_speedup_optimized.png")

    # # Create a detailed text report
    # with open("quickhull_speedup_report.txt", "w") as f:
    #     f.write("Quickhull Parallel Speedup Detailed Analysis\n")
    #     f.write("===========================================\n\n")
    #     f.write("Points\t\tSort Speedup\tAlgo Speedup\tTotal Speedup\n")
    #     f.write("-------\t\t-----------\t------------\t-------------\n")
    #     for i in range(len(point_counts)):
    #         f.write(
    #             f"{point_counts[i]}\t\t{sort_speedup[i]:.2f}x\t\t{algo_speedup[i]:.2f}x\t\t{total_speedup[i]:.2f}x\n"
    #         )
    #
    # # Add some additional insights
    # f.write("\nInsights:\n")
    # f.write(f"Average Sort Speedup: {np.mean(sort_speedup):.2f}x\n")
    # f.write(f"Average Algorithm Speedup: {np.mean(algo_speedup):.2f}x\n")
    # f.write(f"Average Total Execution Speedup: {np.mean(total_speedup):.2f}x\n")


# Run the analysis
plot_detailed_speedup("quickhull_output_optimized.log")
