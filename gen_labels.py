"""
Generate labels

This script runs the task identification on some data, then derives cluster labels from groups of segmentations.

Leopold Ehrlich
February 1 2026
"""

def main():
    # Load the data

    # Run the task identification

    # Run the label generation
    
    pass

def align_poi():
    """Given points where possible indicators of task switches happen, align it to when the task switch likely happened"""
    pass

def label_dist():
    """For given array of time points, produce the distributions around those points representing the possiblility of a task switch happening then"""
    pass

def extract_switches():
    """For given set of distributions, perform a weighted sum followed by a threshold such that the output is populated with high confidence task switches"""
    pass

def gen_label_vec():
    """Given an array of task switches, produce a vector labeling the appropriate clusters for each time step"""
    pass



if __name__ == "__main__":
    main()
