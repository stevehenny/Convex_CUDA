import matplotlib.pyplot as plt
import re


def parse_log_file(filename):
    """
    Parse the quickhull log file and extract timing information.

    Returns:
    - point_counts: List of point counts
    - parallel_sort_times: List of parallel sort times
    - parallel_algo_times: List of parallel algorithm execution times
    - parallel_total_times: List of parallel total execution times
    - serial_sort_times: List of serial sort times
    - serial_algo_times: List of serial algorithm execution times
    - serial_total_times: List of serial total execution times
    """
    point_counts = []
    parallel_sort_times = []
    parallel_algo_times = []
    parallel_total_times = []
    serial_sort_times = []
    serial_algo_times = []
    serial_total_times = []

    with open(filename, "r") as file:
        for line in file:
            # Check for point count
            point_match = re.search(r"Running for (\d+) points\.\.\.", line)
            if point_match:
                point_counts.append(int(point_match.group(1)))

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

    return (
        point_counts,
        parallel_sort_times,
        parallel_algo_times,
        parallel_total_times,
        serial_sort_times,
        serial_algo_times,
        serial_total_times,
    )


def plot_execution_times(filename):
    """
    Create a bar graph comparing serial and parallel execution times.
    """
    # Parse the log file
    (
        point_counts,
        parallel_sort_times,
        parallel_algo_times,
        parallel_total_times,
        serial_sort_times,
        serial_algo_times,
        serial_total_times,
    ) = parse_log_file(filename)

    # Set up the plot
    plt.figure(figsize=(12, 8))

    # Width of each bar group
    width = 0.15

    # X-axis positions
    x = range(len(point_counts))

    # Create bar plots
    plt.bar(
        [i - 1.5 * width for i in x],
        parallel_sort_times,
        width,
        label="Parallel Sort Time",
        color="blue",
        alpha=0.7,
    )
    plt.bar(
        [i - 0.5 * width for i in x],
        parallel_algo_times,
        width,
        label="Parallel Algo Time",
        color="green",
        alpha=0.7,
    )
    plt.bar(
        [i + 0.5 * width for i in x],
        parallel_total_times,
        width,
        label="Parallel Total Time",
        color="red",
        alpha=0.7,
    )

    plt.bar(
        [i + 1.5 * width for i in x],
        serial_sort_times,
        width,
        label="Serial Sort Time",
        color="cyan",
        alpha=0.7,
    )
    plt.bar(
        [i + 2.5 * width for i in x],
        serial_algo_times,
        width,
        label="Serial Algo Time",
        color="lime",
        alpha=0.7,
    )
    plt.bar(
        [i + 3.5 * width for i in x],
        serial_total_times,
        width,
        label="Serial Total Time",
        color="orange",
        alpha=0.7,
    )

    # Customize the plot
    plt.xlabel("Number of Points", fontsize=12)
    plt.ylabel("Execution Time (ms)", fontsize=12)
    plt.title(
        "Quickhull with optimized flag (-O3): Serial vs Parallel Execution Times",
        fontsize=14,
    )
    plt.xticks([i + width for i in x], point_counts, rotation=45)
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Tight layout to prevent cutting off labels
    plt.tight_layout()

    # Save the plot
    plt.savefig("pictures/quickhull_performance_optimized.png")


# Run the analysis
plot_execution_times("quickhull_output_optimized.log")
